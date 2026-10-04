"""WhatsApp Business Platform (Cloud API) adapter — optional bonus.

Maps the agent's channel-agnostic messages onto WhatsApp:
  * ≤ 3 buttons  -> interactive "button" message (titles ≤ 20 chars)
  * 4–10 buttons -> interactive "list" message (rows ≤ 24 chars)
  * > 10 buttons -> list of the first 9 + "Menu"
  * field excerpt -> image message (uploaded crop) followed by the question
Incoming webhooks: text, interactive button/list replies and images (downloaded via the
Graph API media endpoint) are turned into agent events.

Configure: WA_TOKEN, WA_PHONE_NUMBER_ID, WA_VERIFY_TOKEN, WA_APP_SECRET.
Security: X-Hub-Signature-256 is verified on every webhook call. The phone number of the
midwife is mapped to a midwife ID (never stored on records).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import urllib.request
from typing import Optional

GRAPH = "https://graph.facebook.com/v20.0"


def _clip(s: str, n: int) -> str:
    s = s.replace("*", "")
    return s if len(s) <= n else s[: n - 1] + "…"


def to_whatsapp(to: str, m: dict) -> list[dict]:
    """Agent message -> list of Cloud API message payloads."""
    out = []
    text = m.get("text") or " "
    if m.get("image_url_public"):
        out.append({"messaging_product": "whatsapp", "to": to, "type": "image",
                    "image": {"link": m["image_url_public"]}})
    btns = m.get("buttons") or []
    if not btns:
        out.append({"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text[:4096]}})
    elif len(btns) <= 3:
        out.append({"messaging_product": "whatsapp", "to": to, "type": "interactive", "interactive": {
            "type": "button", "body": {"text": text[:1024]},
            "action": {"buttons": [{"type": "reply", "reply": {"id": b["id"][:256], "title": _clip(b["label"], 20)}}
                                   for b in btns]}}})
    else:
        rows = btns[:9] + ([{"id": "menu", "label": "🏠 Menu"}] if len(btns) > 10 else btns[9:10])
        out.append({"messaging_product": "whatsapp", "to": to, "type": "interactive", "interactive": {
            "type": "list", "body": {"text": text[:1024]},
            "action": {"button": "Choisir", "sections": [{"title": "Options", "rows": [
                {"id": b["id"][:200], "title": _clip(b["label"], 24)} for b in rows]}]}}})
    return out


def verify_signature(body: bytes, header: Optional[str], secret: Optional[str] = None) -> bool:
    secret = secret or os.environ.get("WA_APP_SECRET", "")
    if not secret or not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header[7:])


def parse_webhook(payload: dict) -> list[tuple[str, dict]]:
    """Cloud API webhook JSON -> [(sender_phone, agent_event)]. Images carry a media id."""
    events = []
    for entry in payload.get("entry", []):
        for ch in entry.get("changes", []):
            for msg in ch.get("value", {}).get("messages", []):
                frm, ty = msg.get("from"), msg.get("type")
                if ty == "text":
                    events.append((frm, {"type": "text", "text": msg["text"]["body"]}))
                elif ty == "interactive":
                    it = msg["interactive"]
                    r = it.get("button_reply") or it.get("list_reply") or {}
                    events.append((frm, {"type": "button", "id": r.get("id", ""), "label": r.get("title", "")}))
                elif ty == "image":
                    events.append((frm, {"type": "photo_media", "media_id": msg["image"]["id"]}))
    return events


def download_media(media_id: str) -> bytes:
    tok = os.environ["WA_TOKEN"]
    req = urllib.request.Request(f"{GRAPH}/{media_id}", headers={"Authorization": f"Bearer {tok}"})
    meta = json.loads(urllib.request.urlopen(req, timeout=30).read())
    req = urllib.request.Request(meta["url"], headers={"Authorization": f"Bearer {tok}"})
    return urllib.request.urlopen(req, timeout=60).read()


def send(payload: dict) -> dict:
    tok, pid = os.environ["WA_TOKEN"], os.environ["WA_PHONE_NUMBER_ID"]
    req = urllib.request.Request(f"{GRAPH}/{pid}/messages", data=json.dumps(payload).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())
