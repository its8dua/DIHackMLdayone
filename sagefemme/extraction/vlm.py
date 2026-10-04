"""Vision-LLM extraction backend (online layer).

Providers (pick with env VLM_PROVIDER):
  anthropic : Claude Messages API (ANTHROPIC_API_KEY, VLM_MODEL default claude-sonnet-4-5)
  openai    : any OpenAI-compatible endpoint (VLM_BASE_URL, VLM_API_KEY, VLM_MODEL) — e.g. a
              self-hosted Qwen2.5-VL / Llama-3.2-Vision through vLLM or Ollama, so that real
              patient data never leaves the health-system network.
Privacy: the page is ALWAYS redacted on-device (direct identifiers blacked out using the
template geometry) before it is sent. The model receives a schema for that page type only and
must return value + status + confidence per field, saying when it is unsure.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageDraw

from ..forms.layout import PAGE_TYPES, SPECS
from ..net import Network, OfflineError

STATUSES = ["KNOWN", "UNKNOWN", "NOT_PROVIDED", "ILLEGIBLE", "NOT_APPLICABLE", "NEEDS_REVIEW"]


@dataclass
class VReading:
    raw: object = None
    conf: float = 0.0
    dash: bool = False
    note: str = ""
    status_hint: Optional[str] = None


def redact(img: Image.Image, polys: list) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    for p in polys:
        d.polygon([tuple(map(float, xy)) for xy in p], fill=(0, 0, 0))
    return out


def page_schema(page_type: str) -> str:
    lines = []
    for s in SPECS:
        if s.page != page_type or s.kind == "pii":
            continue
        kind = "checkbox (true if ticked/crossed/hatched, else false)" if s.kind in ("check", "checkrow") else s.type
        lines.append(f"{s.id} | {s.fr} | {kind}")
    return "\n".join(lines)


SYSTEM = """You transcribe a photographed page of a French maternal-health paper registry (Morocco).
Handwriting may be French, Arabic or English. Some identifiers are blacked out on purpose: never guess them.
Return ONLY JSON: {"fields": {"<id>": {"value": <string|true|false|null>, "status": "<STATUS>", "confidence": <0..1>}}}
Rules:
- Transcribe exactly what is written (keep units like "g", "cm", "SA"; dates as written dd/mm/yyyy).
- status KNOWN = clearly read. NEEDS_REVIEW = you read something but are not sure (still give your best value).
  ILLEGIBLE = something is written but unreadable (value null). NOT_PROVIDED = left blank.
  NOT_APPLICABLE = a dash "—" or "/" is written. UNKNOWN = the paper says unknown ("inconnu", "?").
- confidence is your honest probability that the value is exactly right. Never hide doubt.
- Characters may be missing from a word (gaps): give your best reconstruction with status NEEDS_REVIEW.
- Checkboxes: value true/false, status KNOWN unless the mark is ambiguous.
- Do not add fields that are not listed. Do not output names, phone numbers, ID numbers or addresses."""


def _image_b64(img: Image.Image, max_side=2000) -> str:
    im = img.convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _post(url, headers, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse_response(text: str, page_type: str) -> dict[str, VReading]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in model output")
    data = json.loads(m.group())
    fields = data.get("fields", data)
    valid = {s.id: s for s in SPECS if s.page == page_type and s.kind != "pii"}
    out = {}
    for fid, spec in valid.items():
        f = fields.get(fid)
        if not isinstance(f, dict):
            out[fid] = VReading(None, 0.3, note="non renvoyé par le modèle", status_hint="NEEDS_REVIEW")
            continue
        st = str(f.get("status", "NEEDS_REVIEW")).upper()
        st = st if st in STATUSES else "NEEDS_REVIEW"
        conf = float(f.get("confidence", 0.5) or 0)
        conf = max(0.0, min(1.0, conf))
        val = f.get("value")
        if spec.kind in ("check", "checkrow"):
            out[fid] = VReading(bool(val) if val is not None else False, conf if st == "KNOWN" else min(conf, 0.6))
            continue
        if st == "NOT_APPLICABLE":
            out[fid] = VReading("—", conf, dash=True)
        elif st == "NOT_PROVIDED":
            out[fid] = VReading(None, conf, note="blank")
        elif st in ("ILLEGIBLE", "UNKNOWN"):
            out[fid] = VReading(None, conf, status_hint=st)
        else:
            # self-reported confidence is capped when the model itself flags review
            out[fid] = VReading(None if val is None else str(val), conf if st == "KNOWN" else min(conf, 0.7))
    return out


class VLMExtractor:
    def __init__(self, network: Network, provider: Optional[str] = None):
        self.net = network
        self.provider = provider or os.environ.get("VLM_PROVIDER") or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY")
                                                                      else ("openai" if os.environ.get("VLM_BASE_URL") else None))
        self.model = os.environ.get("VLM_MODEL") or ("claude-sonnet-4-5" if self.provider == "anthropic" else "qwen2.5-vl")

    @property
    def available(self) -> bool:
        return self.provider is not None

    def read_page(self, img: Image.Image, page_type: str) -> dict[str, VReading]:
        self.net.require("AI processing")
        prompt = (f"Page type: {PAGE_TYPES[page_type][1]}.\nFields (id | printed label | type):\n"
                  f"{page_schema(page_type)}\nReturn the JSON now.")
        b64 = _image_b64(img)
        if self.provider == "anthropic":
            body = {"model": self.model, "max_tokens": 16000, "system": SYSTEM, "temperature": 0,
                    "messages": [{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": prompt}]}]}
            base = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com")
            res = _post(f"{base}/v1/messages", {"content-type": "application/json",
                                                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                                "anthropic-version": "2023-06-01"}, body, timeout=300)
            text = "".join(c.get("text", "") for c in res.get("content", []))
        elif self.provider == "openai":
            body = {"model": self.model, "temperature": 0, "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                                             {"type": "text", "text": prompt}]}]}
            base = os.environ["VLM_BASE_URL"].rstrip("/")
            hdr = {"content-type": "application/json"}
            if os.environ.get("VLM_API_KEY"):
                hdr["authorization"] = f"Bearer {os.environ['VLM_API_KEY']}"
            res = _post(f"{base}/chat/completions", hdr, body, timeout=300)
            text = res["choices"][0]["message"]["content"]
        else:
            raise RuntimeError("no VLM provider configured")
        self.net.require("AI response")
        return parse_response(text, page_type)
