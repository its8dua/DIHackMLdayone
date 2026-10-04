"""WhatsApp-style conversational agent: the midwife's only interface.

Input events  : {"type": "text", "text": ...} | {"type": "button", "id": ...} | {"type": "photo", "data": bytes}
Output        : list of messages {"text", "buttons": [{"id","label"}], "image": url|None}
Buttons are capped/paginated so the same flow maps onto WhatsApp interactive messages
(3 reply buttons, or a list message of up to 10 rows) — see whatsapp.py.

The agent never hides doubt: every uncertain field is shown with what was read, the
confidence, and the photo excerpt, and the midwife decides (confirm / edit / blank / illegible).
"""
from __future__ import annotations

import re
import threading
import time
from typing import Optional

from .extraction.pipeline import capture_check
from .extraction.postprocess import consistency, fv
from .forms.layout import GRID_ROWS, PAGE_ORDER, PAGE_TYPES, SPEC_BY_ID, VISIT_COLS
from .i18n import fmt, t
from .linking import diff, find_candidates, link_record, summary
from .net import Network
from .schema import FIELDS, FieldStatus, normalize, section_label, strip_accents
from .schema import RecordState as S
from .store import LocalStore

ESSENTIAL_ROWS = {"visit_date", "ga", "weight", "bp", "fhr", "albuminuria", "syphilis", "hiv", "hb", "glycemia"}


def msg(text, buttons=None, image=None, kind="bot"):
    return {"from": kind, "text": text, "buttons": buttons or [], "image": image, "ts": time.time()}


def B(i, label):
    return {"id": i, "label": label}


def pct(c):
    return int(round(100 * float(c)))


def label(fid, lang):
    s = SPEC_BY_ID[fid]
    return s.fr if lang == "fr" or not s.en else s.en


def page_label(pt, lang):
    return section_label(pt, lang)


def show(v, lang="fr"):
    if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):      # ISO -> as written on paper
        return f"{v[8:10]}/{v[5:7]}/{v[0:4]}"
    if v is True:
        return t("yes", lang)
    if v is False:
        return t("no", lang)
    return "—" if v is None else str(v)


def is_valid(fid, value) -> bool:
    if value is None or isinstance(value, bool):
        return True
    return bool(normalize(SPEC_BY_ID[fid], value)[1])


class Agent:
    def __init__(self, store: LocalStore, net: Network, worker=None):
        self.store, self.net, self.worker = store, net, worker
        self.sessions: dict[str, dict] = {}
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ sessions
    def session(self, midwife_id: str, name: str = "", role: str = "midwife") -> dict:
        with self.lock:
            s = self.sessions.get(midwife_id)
            if not s:
                s = {"midwife_id": midwife_id, "name": name or midwife_id, "role": role, "lang": "fr",
                     "mode": "menu", "ctx": {}, "outbox": [], "seq": 0}
                self.sessions[midwife_id] = s
            return s

    def push(self, s, *messages):
        for m in messages:
            s["seq"] += 1
            m["seq"] = s["seq"]
            s["outbox"].append(m)
            if len(s["outbox"]) > 400:
                s["outbox"] = s["outbox"][-300:]

    def start(self, s) -> None:
        L = s["lang"]
        self.push(s, msg(t("welcome", L, name=s["name"])))
        open_caps = [r for r in self.store.records([S.CAPTURED]) if r["midwife_id"] == s["midwife_id"]]
        if open_caps:
            rid = open_caps[-1]["id"]
            n = len(self.store.image_rows(rid))
            s["ctx"] = {"rid": rid}
            self.push(s, msg(t("resume", L, rid=rid, n=n), [B("cap:resume", t("b_continue", L)), B("menu", t("b_menu", L))]))
            return
        self.menu(s)

    def menu(self, s, text=None):
        L = s["lang"]
        s["mode"], s["ctx"] = "menu", {}
        self.push(s, msg(text or t("menu", L), [B("menu:new", t("b_new", L)), B("menu:manual", t("b_manual", L)),
                                                 B("menu:queue", t("b_queue", L)), B("menu:find", t("b_find", L)),
                                                 B("menu:lang", t("b_lang", L))]))

    # ------------------------------------------------------------------ entry point
    def handle(self, s, ev: dict) -> None:
        with self.lock:
            L = s["lang"]
            if ev["type"] == "text":
                self.push(s, msg(ev["text"], kind="user"))
            elif ev["type"] == "button":
                self.push(s, msg(ev.get("label", ev["id"]), kind="user"))
            elif ev["type"] == "photo":
                self.push(s, msg("📷", image=ev.get("preview"), kind="user"))
            try:
                self._dispatch(s, ev)
            except Exception as e:      # never crash the conversation
                import traceback
                traceback.print_exc()
                self.push(s, msg(f"⚠️ Erreur interne : {e}. Vos données sont conservées."))
                self.menu(s)

    def _dispatch(self, s, ev):
        L, mode, ctx = s["lang"], s["mode"], s["ctx"]
        bid = ev.get("id", "") if ev["type"] == "button" else ""
        text = ev.get("text", "").strip() if ev["type"] == "text" else ""
        if bid == "menu" or text.lower() in ("menu", "accueil", "home"):
            return self.menu(s)
        if bid.startswith("menu:"):
            a = bid.split(":")[1]
            if a == "new":
                return self.start_capture(s)
            if a == "manual":
                return self.start_manual(s)
            if a == "queue":
                return self.show_queue(s)
            if a == "find":
                s["mode"] = "find"
                return self.push(s, msg(t("ask_code", L)))
            if a == "lang":
                s["lang"] = "en" if L == "fr" else "fr"
                return self.menu(s)
        if bid.startswith("open:"):
            return self.open_record(s, bid.split(":", 1)[1])
        if bid.startswith("view:"):        # opens the full record panel, conversation state unchanged
            return self.push(s, msg("📄 " + bid[5:], []) | {"open_record": bid[5:]})
        if bid.startswith("retry:"):
            rid = bid.split(":", 1)[1]
            self.store.set_state(rid, S.PENDING_AI, s["midwife_id"], "retry demandé")
            self.push(s, msg(t("queued_online" if self.net.online else "queued_offline", L, rid=rid,
                               n=len(self.store.image_rows(rid)))))
            self.kick()
            return
        if bid.startswith("manualrec:"):
            return self.start_manual(s, bid.split(":", 1)[1])
        if bid == "cap:resume":
            s["mode"] = "capture"
            return self.push(s, msg(t("cap_start", L), self._cap_buttons(s)))

        if ev["type"] == "photo":
            if mode == "retake":
                return self.on_retake_photo(s, ev["data"])
            if mode not in ("capture", "cap_quality", "cap_dup"):
                self.start_capture(s, silent=True)
            return self.on_photo(s, ev["data"])

        handler = getattr(self, f"on_{mode}", None)
        if handler:
            return handler(s, bid, text)
        return self.menu(s)

    def kick(self):
        if self.worker and self.net.online:
            threading.Thread(target=self.worker.tick, daemon=True).start()

    # ------------------------------------------------------------------ capture
    def _cap_buttons(self, s):
        L = s["lang"]
        return [B("cap:done", t("b_done", L)), B("cap:cancel", t("b_cancel", L))]

    def start_capture(self, s, silent=False):
        s["mode"], s["ctx"] = "capture", {"rid": None}
        if not silent:
            self.push(s, msg(t("cap_start", s["lang"]), self._cap_buttons(s)))

    def _ensure_record(self, s):
        ctx = s["ctx"]
        if not ctx.get("rid"):
            ctx["rid"] = self.store.create_record(s["midwife_id"], {"fields": {}, "pii_polys": {}, "page_types": {},
                                                                    "origin": "photo"})
        return ctx["rid"]

    def on_photo(self, s, data: bytes, force=False):
        L, ctx = s["lang"], s["ctx"]
        chk = capture_check(data)               # offline, on-device
        pt = chk["page_type"]
        if not force and (not chk["quality"]["ok"] and "page_not_recognised" not in chk["quality"]["issues"]):
            s["mode"] = "cap_quality"
            ctx["pending"] = data
            issues = ", ".join(chk["quality"]["issues"])
            return self.push(s, msg(t("cap_quality", L, issues=issues),
                                    [B("cap:retake", t("b_retake", L)), B("cap:keep", t("b_keep", L))]))
        if not pt:
            s["mode"] = "capture"
            return self.push(s, msg(t("cap_page_unknown", L), self._cap_buttons(s)))
        rid = self._ensure_record(s)
        rec = self.store.get_record(rid)
        existing = {v: k for k, v in rec["payload"].get("page_types", {}).items()}
        if pt in existing and not force:
            s["mode"] = "cap_dup"
            ctx["pending"], ctx["pending_pt"] = data, pt
            return self.push(s, msg(t("cap_dup", L, page=page_label(pt, L)),
                                    [B("cap:replace", t("b_replace", L)), B("cap:keepold", t("b_keep_old", L))]))
        self._store_page(s, rid, data, pt, chk, replace_img=existing.get(pt))
        s["mode"] = "capture"
        n = len(self.store.image_rows(rid))
        self.push(s, msg(t("cap_page_ok", L, n=n, page=page_label(pt, L)), self._cap_buttons(s)))

    def _store_page(self, s, rid, data, pt, chk, replace_img=None):
        rec = self.store.get_record(rid)
        payload = rec["payload"]
        with self.store.tx() as db:
            if replace_img:
                db.execute("UPDATE images SET superseded=1 WHERE id=?", (replace_img,))
                payload["page_types"].pop(replace_img, None)
            img_id = self.store.save_image(rid, PAGE_ORDER.index(pt) + 1, data, s["midwife_id"],
                                           {**chk["quality"], "page_type": pt}, db=db)
            payload["page_types"][img_id] = pt
            payload["pii_polys"][img_id] = chk["pii_polys"]
            self.store.update_payload(rid, payload, db=db)
        return img_id

    def on_cap_quality(self, s, bid, text):
        L, ctx = s["lang"], s["ctx"]
        if bid == "cap:keep":
            s["mode"] = "capture"
            return self.on_photo(s, ctx.pop("pending"), force=True)
        ctx.pop("pending", None)
        s["mode"] = "capture"
        self.push(s, msg("📷", self._cap_buttons(s)))

    def on_cap_dup(self, s, bid, text):
        ctx = s["ctx"]
        data, pt = ctx.pop("pending", None), ctx.pop("pending_pt", None)
        s["mode"] = "capture"
        if bid == "cap:replace" and data:
            return self.on_photo(s, data, force=True)
        self.push(s, msg("👌", self._cap_buttons(s)))

    def on_capture(self, s, bid, text):
        L, ctx = s["lang"], s["ctx"]
        if bid == "cap:cancel":
            rid = ctx.get("rid")
            if rid:
                self.push(s, msg(t("cancelled", L, rid=rid)))
            return self.menu(s)
        if bid == "cap:done":
            rid = ctx.get("rid")
            if not rid:
                return self.menu(s)
            rec = self.store.get_record(rid)
            have = set(rec["payload"]["page_types"].values())
            missing = [page_label(p, L) for p in PAGE_ORDER if p not in have]
            if missing:
                self.push(s, msg(t("cap_missing", L, pages=", ".join(missing))))
            self.store.set_state(rid, S.PENDING_AI, s["midwife_id"], "capture terminée")
            n = len(self.store.image_rows(rid))
            s["mode"], s["ctx"] = "menu", {}
            if self.net.online:
                self.push(s, msg(t("queued_online", L, rid=rid, n=n)))
                self.kick()
            else:
                self.push(s, msg(t("queued_offline", L, rid=rid, n=n),
                                 [B(f"manualrec:{rid}", t("b_manual", L)), B("menu", t("b_menu", L))]))
            return
        self.push(s, msg(t("cap_start", L), self._cap_buttons(s)))

    # ------------------------------------------------------------------ notifications from the worker
    def notify(self, kind: str, data: dict):
        with self.lock:
            rid = data.get("record")
            rec = self.store.get_record(rid) if rid else None
            targets = [x for x in self.sessions.values() if not rec or x["midwife_id"] == rec["midwife_id"]]
            for s in targets:
                L = s["lang"]
                if kind == "processed":
                    c = data["counts"]
                    self.push(s, msg(t("processed", L, rid=rid, sec=data["seconds"], known=c.get("KNOWN", 0),
                                       review=c.get("NEEDS_REVIEW", 0), illegible=c.get("ILLEGIBLE", 0),
                                       blank=c.get("NOT_PROVIDED", 0) + c.get("NOT_APPLICABLE", 0) + c.get("UNKNOWN", 0)),
                                     [B(f"open:{rid}", t("b_review", L)), B("menu", t("b_later", L))]))
                elif kind == "offline_during_processing":
                    self.push(s, msg(t("offline_mid", L, rid=rid)))
                elif kind == "processing_failed":
                    self.push(s, msg(t("proc_failed", L, rid=rid), [B(f"retry:{rid}", t("b_retry", L)),
                                                                     B(f"manualrec:{rid}", t("b_manual", L))]))
                elif kind == "synced":
                    self.push(s, msg(t("synced", L, rid=rid)))
                elif kind == "duplicate_central":
                    self.push(s, msg(t("dup_central", L, rid=rid, others=", ".join(data["others"])),
                                     [B(f"open:{rid}", t("b_view", L))]))
                elif kind == "sync_failed":
                    self.push(s, msg(t("sync_failed", L, rid=rid, err=data.get("error", ""))))
                elif kind == "network":
                    if data["online"]:
                        n = len(self.store.records([S.PENDING_AI]))
                        self.push(s, msg(t("net_back", L, n=n)))
                    else:
                        self.push(s, msg(t("net_lost", L)))

    # ------------------------------------------------------------------ queue / open
    STATE_ICON = {"CAPTURED": "📷", "PENDING_AI": "⏳", "AI_PROCESSED": "🤖", "NEEDS_REVIEW": "🔍", "VALIDATED": "✅",
                  "PATIENT_MATCHED": "🔗", "REGISTERED": "💾", "SYNCED": "☁️", "PROCESSING_FAILED": "❌",
                  "SYNC_FAILED": "⚠️", "DUPLICATE_SUSPECTED": "🤔", "MANUAL_REVIEW_REQUIRED": "⌨️"}

    def show_queue(self, s):
        L = s["lang"]
        recs = [r for r in self.store.records() if r["midwife_id"] == s["midwife_id"]][-12:]
        if not recs:
            return self.menu(s, t("queue_empty", L))
        lines = "\n".join(f"{self.STATE_ICON.get(r['state'], '•')} {r['id']} — {r['state']}"
                          + (f" → {r['patient_id']}" if r["patient_id"] else "") for r in recs)
        btns = []
        for r in recs:
            if r["state"] in ("NEEDS_REVIEW", "VALIDATED", "DUPLICATE_SUSPECTED", "MANUAL_REVIEW_REQUIRED"):
                btns.append(B(f"open:{r['id']}", f"🔍 {r['id']}"))
            elif r["state"] == "PROCESSING_FAILED":
                btns.append(B(f"retry:{r['id']}", f"🔁 {r['id']}"))
        self.push(s, msg(t("queue", L, lines=lines), btns[:6] + [B("menu", t("b_menu", L))]))

    def open_record(self, s, rid):
        rec = self.store.get_record(rid)
        st = rec["state"]
        if st == "NEEDS_REVIEW":
            return self.start_review(s, rid)
        if st == "VALIDATED" or st == "DUPLICATE_SUSPECTED":
            return self.start_linking(s, rid)
        if st == "MANUAL_REVIEW_REQUIRED":
            return self.start_manual(s, rid)
        self.push(s, msg(f"{self.STATE_ICON.get(st, '')} {rid} — {st}", [B("menu", t("b_menu", s['lang']))]))

    # ------------------------------------------------------------------ review
    def _fields(self, rid):
        return self.store.get_record(rid)["payload"]["fields"]

    def set_field(self, rid, fid, value, status, source, note=""):
        rec = self.store.get_record(rid)
        p = rec["payload"]
        old = p["fields"].get(fid, {})
        f = fv(value, status, 1.0, source, old.get("raw"), note, SPEC_BY_ID[fid].page)
        f["previous"] = {"value": old.get("value"), "status": old.get("status"), "confidence": old.get("confidence")}
        p["fields"][fid] = f
        self.store.update_payload(rid, p)

    def build_tasks(self, fields: dict, essential_only=False) -> list[dict]:
        tasks = []
        code = fields.get("fiche_number")
        if code and not str(code.get("source", "")).startswith(("midwife", "manual")):
            tasks.append({"kind": "code", "fid": "fiche_number"})
        rows = {}
        singles = []
        for fid, f in fields.items():
            if fid == "fiche_number" or f["status"] not in ("NEEDS_REVIEW", "ILLEGIBLE"):
                continue
            spec = SPEC_BY_ID[fid]
            if essential_only and spec.importance < 2:
                continue
            if fid.startswith("g_"):
                row = fid[2:].rsplit("_", 1)[0]
                rows.setdefault(row, []).append(fid)
            else:
                singles.append(fid)
        singles.sort(key=lambda f: (-SPEC_BY_ID[f].importance, PAGE_ORDER.index(SPEC_BY_ID[f].page)))
        for fid in singles:
            tasks.append({"kind": "field", "fid": fid})
        order = [r[0] for r in GRID_ROWS]
        for row in sorted(rows, key=lambda r: (-dict((x[0], x[3]) for x in GRID_ROWS)[r], order.index(r))):
            cols = [c for c in (f"g_{row}_{cid}" for cid, *_ in VISIT_COLS) if c in rows[row]]
            tasks.append({"kind": "row", "row": row, "fids": cols})
        return tasks

    def start_review(self, s, rid, essential_only=None):
        L = s["lang"]
        fields = self._fields(rid)
        all_tasks = self.build_tasks(fields)
        if essential_only is None and len(all_tasks) > 25:
            ess = self.build_tasks(fields, essential_only=True)
            s["mode"], s["ctx"] = "review_scope", {"rid": rid}
            n = sum(len(x.get("fids", [1])) for x in all_tasks)
            e = sum(len(x.get("fids", [1])) for x in ess)
            return self.push(s, msg(t("rev_many", L, n=n, e=e), [B("scope:ess", t("b_essential", L)),
                                                                  B("scope:all", t("b_all", L)),
                                                                  B(f"sum:{rid}", t("b_view", L))]))
        tasks = self.build_tasks(fields, essential_only=bool(essential_only))
        s["mode"], s["ctx"] = "review", {"rid": rid, "tasks": tasks, "i": 0, "await": None}
        if tasks:
            n = sum(len(x.get("fids", [1])) for x in tasks)
            self.push(s, msg(t("rev_intro", L, rid=rid, n=n)))
        self.next_task(s)

    def on_review_scope(self, s, bid, text):
        rid = s["ctx"]["rid"]
        if bid.startswith("sum:"):
            return self.show_summary(s, rid)
        self.start_review(s, rid, essential_only=(bid == "scope:ess"))

    def crop_url(self, rid, fid):
        return f"/api/crop/{rid}/{fid}"

    def next_task(self, s):
        L, ctx = s["lang"], s["ctx"]
        rid = ctx["rid"]
        if ctx["i"] >= len(ctx["tasks"]):
            return self.show_summary(s, rid)
        task = ctx["tasks"][ctx["i"]]
        fields = self._fields(rid)
        if task["kind"] == "code":
            f = fields["fiche_number"]
            ctx["await"] = None
            if f["value"]:
                return self.push(s, msg(t("q_code", L, v=f["value"], c=pct(f["confidence"])),
                                        [B("t:ok", t("b_ok", L)), B("t:edit", t("b_edit", L)),
                                         B("t:retake", t("b_retake", L))], image=self.crop_url(rid, "fiche_number")))
            ctx["await"] = "value"
            return self.push(s, msg(t("q_code_missing", L), [B("t:retake", t("b_retake", L))],
                                    image=self.crop_url(rid, "fiche_number")))
        if task["kind"] == "row":
            row = task["row"]
            rl = dict((r[0], r[1]) for r in GRID_ROWS)[row]
            lines = []
            for fid in task["fids"]:
                f = fields[fid]
                col = SPEC_BY_ID[fid].fr.split("—")[-1].strip()
                if f["status"] == "ILLEGIBLE":
                    v = "❓ illisible" if L == "fr" else "❓ illegible"
                elif f["value"] is None:
                    v = f"∅ {'vide ?' if L == 'fr' else 'blank?'} ({pct(f['confidence'])}%)"
                else:
                    v = f"« {show(f['value'], L)} » ({pct(f['confidence'])}%)"
                lines.append(f"• {col} : {v}")
            ctx["await"] = None
            return self.push(s, msg(t("q_row", L, label=rl, page=page_label("pregnancy", L), n=len(task["fids"]),
                                      rows="\n".join(lines)),
                                    [B("t:rowok", t("b_row_ok", L)), B("t:rowedit", t("b_row_edit", L)),
                                     B("t:skip", t("b_skip", L))], image=f"/api/croprow/{rid}/{row}"))
        fid = task["fid"]
        f, spec = fields[fid], SPEC_BY_ID[fid]
        pl = page_label(spec.page, L)
        img = self.crop_url(rid, fid)
        ctx["await"] = None
        if spec.kind in ("check", "checkrow"):
            yn = t("yes", L) if f["value"] else t("no", L)
            return self.push(s, msg(t("q_check", L, label=label(fid, L), page=pl, yn=yn, c=pct(f["confidence"])),
                                    [B("t:tick", t("b_ticked", L)), B("t:untick", t("b_unticked", L)),
                                     B("t:skip", t("b_skip", L))], image=img))
        if f["status"] == "ILLEGIBLE":
            return self.push(s, msg(t("q_illegible", L, label=label(fid, L), page=pl),
                                    [B("t:edit", t("b_edit", L)), B("t:blank", t("b_blank", L)),
                                     B("t:illegible", t("b_illegible", L)), B("t:skip", t("b_skip", L))], image=img))
        if f["value"] is None:
            return self.push(s, msg(t("q_blank", L, label=label(fid, L), page=pl, c=pct(f["confidence"])),
                                    [B("t:blank", t("b_blank", L)), B("t:edit", t("b_edit", L)),
                                     B("t:skip", t("b_skip", L))], image=img))
        note = f"\n_{f['note']}_" if f.get("note") else ""
        btns = [B("t:edit", t("b_edit", L)), B("t:blank", t("b_blank", L)),
                B("t:illegible", t("b_illegible", L)), B("t:skip", t("b_skip", L))]
        if is_valid(fid, f["value"]):          # an unparseable reading cannot be "confirmed"
            btns.insert(0, B("t:ok", t("b_ok", L)))
        return self.push(s, msg(t("q_field", L, label=label(fid, L), page=pl, v=show(f["value"], L),
                                  c=pct(f["confidence"]), note=note), btns, image=img))

    def parse_answer(self, fid, text) -> tuple[bool, object, str]:
        """Midwife's typed answer -> (ok, value, status)."""
        spec = SPEC_BY_ID[fid]
        low = strip_accents(text.strip().lower())
        if low in ("vide", "blank", "empty", "rien", "∅"):
            return True, None, "NOT_PROVIDED"
        if low in ("-", "—", "/", "na", "n/a", "non applicable"):
            return True, None, "NOT_APPLICABLE"
        if low in ("?", "illisible", "illegible"):
            return True, None, "ILLEGIBLE"
        if low in ("inconnu", "inconnue", "unknown", "nsp"):
            return True, None, "UNKNOWN"
        v, ok = normalize(spec, text)
        if spec.kind in ("check", "checkrow") or spec.type == "bool":
            v2, ok2 = normalize(type("X", (), {"type": "bool", "vmin": None, "vmax": None})(), text)
            return ok2, v2, "KNOWN"
        if spec.type in ("int", "float", "ga") and not isinstance(v, (int, float)):
            return False, None, ""
        if spec.type == "date" and not ok:
            return False, None, ""
        if spec.type == "bp" and not ok:
            return False, None, ""
        return True, v, "KNOWN"

    def on_review(self, s, bid, text):
        L, ctx = s["lang"], s["ctx"]
        rid = ctx["rid"]
        task = ctx["tasks"][ctx["i"]] if ctx["i"] < len(ctx["tasks"]) else None
        if task is None:
            return self.show_summary(s, rid)
        who = s["midwife_id"]
        if ctx.get("await") == "value" and text:
            fid = task["fid"]
            ok, v, st = self.parse_answer(fid, text)
            if not ok:
                return self.push(s, msg(t("bad_value", L, v=text, label=label(fid, L), fmt=fmt(SPEC_BY_ID[fid].type, L))))
            self.set_field(rid, fid, v, st, "midwife_edit")
            self.push(s, msg(t("saved_value", L, label=label(fid, L), v=show(v, L) if st == "KNOWN" else st)))
            ctx["i"] += 1
            return self.next_task(s)
        if ctx.get("await") == "row" and text:
            parts = [p.strip() for p in re.split(r"[;\n]", text) if p.strip() != ""]
            fids = task["fids"]
            if len(parts) != len(fids):
                return self.push(s, msg(t("row_count", L, n=len(fids), got=len(parts)) + " " +
                                        t("ask_row", L, cols=", ".join(SPEC_BY_ID[f].fr.split("—")[-1].strip() for f in fids))))
            parsed = [self.parse_answer(f, p) for f, p in zip(fids, parts)]
            bad = [f for f, (ok, *_r) in zip(fids, parsed) if not ok]
            if bad:
                return self.push(s, msg(t("bad_value", L, v=text, label=label(bad[0], L), fmt=fmt(SPEC_BY_ID[bad[0]].type, L))))
            for f, (ok, v, st) in zip(fids, parsed):
                self.set_field(rid, f, v, st, "midwife_edit")
            self.push(s, msg("👍"))
            ctx["i"] += 1
            return self.next_task(s)
        if task["kind"] == "row":
            if bid == "t:rowok":
                f = self._fields(rid)
                bad = []
                for fid in task["fids"]:
                    if f[fid]["status"] == "ILLEGIBLE" or not is_valid(fid, f[fid]["value"]):
                        bad.append(fid)
                        continue
                    v = f[fid]["value"]
                    self.set_field(rid, fid, v, "KNOWN" if v is not None else "NOT_PROVIDED", "midwife_confirmed")
                if bad:      # unreadable / invalid cells must be typed, not confirmed
                    task["fids"] = bad
                    ctx["await"] = "row"
                    cols = ", ".join(SPEC_BY_ID[x].fr.split("—")[-1].strip() for x in bad)
                    return self.push(s, msg(t("ask_row", L, cols=cols)))
                ctx["i"] += 1
                return self.next_task(s)
            if bid == "t:rowedit":
                ctx["await"] = "row"
                return self.push(s, msg(t("ask_row", L, cols=", ".join(SPEC_BY_ID[f].fr.split("—")[-1].strip() for f in task["fids"]))))
            if bid == "t:skip":
                ctx["i"] += 1
                return self.next_task(s)
            return self.next_task(s)
        fid = task["fid"]
        f = self._fields(rid)[fid]
        if bid == "t:ok":
            if not is_valid(fid, f["value"]):
                ctx["await"] = "value"
                return self.push(s, msg(t("ask_value", L, label=label(fid, L), fmt=fmt(SPEC_BY_ID[fid].type, L))))
            self.set_field(rid, fid, f["value"], "KNOWN", "midwife_confirmed")
        elif bid in ("t:tick", "t:untick"):
            self.set_field(rid, fid, bid == "t:tick", "KNOWN", "midwife_confirmed")
        elif bid == "t:edit":
            ctx["await"] = "value"
            return self.push(s, msg(t("ask_value", L, label=label(fid, L), fmt=fmt(SPEC_BY_ID[fid].type, L))))
        elif bid == "t:blank":
            self.set_field(rid, fid, None, "NOT_PROVIDED", "midwife_confirmed")
        elif bid == "t:illegible":
            self.set_field(rid, fid, None, "ILLEGIBLE", "midwife_confirmed", "illisible aussi pour la sage-femme")
        elif bid == "t:retake":
            ctx["retake_pt"] = SPEC_BY_ID[fid].page
            s["mode"] = "retake"
            return self.push(s, msg(t("retake_now", L, page=page_label(ctx["retake_pt"], L))))
        elif bid == "t:skip":
            pass
        else:
            return self.next_task(s)
        ctx["i"] += 1
        self.next_task(s)

    # ------------------------------------------------------------------ summary / change / retake
    def show_summary(self, s, rid):
        L = s["lang"]
        fields = self._fields(rid)
        consistency(fields)
        lines, left = [], 0
        for pt in PAGE_ORDER:
            fs = [f for fid, f in fields.items() if SPEC_BY_ID[fid].page == pt]
            if all(f.get("note") == "page non photographiée" for f in fs):
                lines.append(f"▫️ {page_label(pt, L)} : page non photographiée")
                continue
            c = {}
            for f in fs:
                c[f["status"]] = c.get(f["status"], 0) + 1
            left += c.get("NEEDS_REVIEW", 0)
            lines.append(f"▪️ {page_label(pt, L)} : ✅{c.get('KNOWN', 0)} ⚠️{c.get('NEEDS_REVIEW', 0)} "
                         f"❓{c.get('ILLEGIBLE', 0)} ∅{c.get('NOT_PROVIDED', 0) + c.get('NOT_APPLICABLE', 0)}")
        key = []
        for fid in ("fiche_number", "age", "gravidity", "parity", "edd", "del_date", "nb_weight"):
            f = fields.get(fid)
            if f and f["value"] is not None:
                key.append(f"{label(fid, L)} : {show(f['value'], L)}{'' if f['status'] == 'KNOWN' else ' ⚠️'}")
        warn = t("warn_left", L, n=left) if left else ""
        s["mode"], s["ctx"] = "summary", {"rid": rid}
        self.push(s, msg(t("summary", L, rid=rid, lines="\n".join(lines) + "\n\n" + " · ".join(key), warn=warn),
                         [B("s:validate", t("b_validate", L)), B("s:change", t("b_change", L)),
                          B("s:retake", t("b_retake_page", L)), B(f"view:{rid}", t("b_view", L)),
                          B("menu", t("b_later", L))]))

    def on_summary(self, s, bid, text):
        L, ctx = s["lang"], s["ctx"]
        rid = ctx["rid"]
        if bid == "s:validate":
            rec = self.store.get_record(rid)
            if rec["state"] in ("NEEDS_REVIEW", "MANUAL_REVIEW_REQUIRED"):
                self.store.set_state(rid, S.VALIDATED, s["midwife_id"], "validé par la sage-femme")
            self.push(s, msg(t("validated", L, rid=rid)))
            return self.start_linking(s, rid)
        if bid == "s:change":
            s["mode"] = "change"
            return self.push(s, msg(t("ask_change", L)))
        if bid == "s:retake":
            s["mode"] = "retake_pick"
            return self.push(s, msg(t("which_page", L), [B(f"rp:{p}", page_label(p, L)) for p in PAGE_ORDER]))
        if bid.startswith("view:"):
            return self.push(s, msg("📄", [B("s:validate", t("b_validate", L)), B("s:change", t("b_change", L))],
                                    image=None) | {"open_record": rid})
        self.show_summary(s, rid)

    def find_field(self, q: str) -> Optional[str]:
        q0 = strip_accents(q.lower()).replace(" ", "")
        if q.strip() in SPEC_BY_ID:
            return q.strip()
        best, bd = None, 99
        from .schema import levenshtein
        for f in FIELDS:
            for cand in (f.fr, f.en, f.id):
                c = strip_accents(cand.lower()).replace(" ", "").replace("—", "")
                if c == q0:
                    return f.id
                d = levenshtein(q0, c)
                if d < bd:
                    best, bd = f.id, d
        return best if bd <= max(2, len(q0) // 5) else None

    def on_change(self, s, bid, text):
        L, ctx = s["lang"], s["ctx"]
        rid = ctx["rid"]
        if "=" not in text:
            return self.push(s, msg(t("ask_change", L)))
        q, v = text.split("=", 1)
        fid = self.find_field(q)
        if not fid:
            return self.push(s, msg(t("no_field", L, q=q.strip())))
        ok, val, st = self.parse_answer(fid, v.strip())
        if not ok:
            return self.push(s, msg(t("bad_value", L, v=v.strip(), label=label(fid, L), fmt=fmt(SPEC_BY_ID[fid].type, L))))
        self.set_field(rid, fid, val, st, "midwife_edit")
        self.push(s, msg(t("saved_value", L, label=label(fid, L), v=show(val, L) if st == "KNOWN" else st)))
        self.show_summary(s, rid)

    def on_retake_pick(self, s, bid, text):
        if bid.startswith("rp:"):
            s["ctx"]["retake_pt"] = bid[3:]
            s["mode"] = "retake"
            return self.push(s, msg(t("retake_now", s["lang"], page=page_label(bid[3:], s["lang"]))))
        self.show_summary(s, s["ctx"]["rid"])

    def on_retake(self, s, bid, text):
        self.push(s, msg(t("retake_now", s["lang"], page=page_label(s["ctx"]["retake_pt"], s["lang"]))))

    def on_retake_photo(self, s, data):
        L, ctx = s["lang"], s["ctx"]
        rid, want = ctx["rid"], ctx["retake_pt"]
        chk = capture_check(data)
        if chk["page_type"] != want:
            got = page_label(chk["page_type"], L) if chk["page_type"] else "?"
            return self.push(s, msg(f"⚠️ {got} ≠ {page_label(want, L)}. " + t("retake_now", L, page=page_label(want, L))))
        rec = self.store.get_record(rid)
        old = {v: k for k, v in rec["payload"]["page_types"].items()}.get(want)
        self._store_page(s, rid, data, want, chk, replace_img=old)
        p = self.store.get_record(rid)["payload"]
        p["retake_pages"] = list(set(p.get("retake_pages", [])) | {want})
        self.store.update_payload(rid, p)
        self.store.set_state(rid, S.PENDING_AI, s["midwife_id"], f"page reprise : {want}")
        s["mode"], s["ctx"] = "menu", {}
        self.push(s, msg(t("queued_online" if self.net.online else "queued_offline", L, rid=rid,
                           n=len(self.store.image_rows(rid)))))
        self.kick()

    # ------------------------------------------------------------------ patient linking
    def start_linking(self, s, rid):
        L = s["lang"]
        fields = self._fields(rid)
        cands = find_candidates(self.store, fields)
        s["mode"], s["ctx"] = "link", {"rid": rid, "cands": cands}
        code = (fields.get("fiche_number") or {}).get("value") or "?"
        if cands:
            lines = "\n".join(f"{i + 1}. {c['patient_id']} — {c['summary']} ({', '.join(c['why'])})"
                              for i, c in enumerate(cands))
            btns = [B(f"lk:{i}", t("b_patient", L, i=i + 1, pid=c["patient_id"])) for i, c in enumerate(cands)]
            return self.push(s, msg(t("match_found", L, n=len(cands), lines=lines),
                                    btns + [B("lk:new", t("b_create", L)), B("lk:unsure", t("b_unsure", L))]))
        self.push(s, msg(t("match_none", L, code=code), [B("lk:new", t("b_create", L)), B("lk:unsure", t("b_unsure", L))]))

    def on_link(self, s, bid, text):
        L, ctx = s["lang"], s["ctx"]
        rid = ctx["rid"]
        rec = self.store.get_record(rid)
        if bid == "lk:unsure":
            if rec["state"] == "VALIDATED":
                self.store.set_state(rid, S.DUPLICATE_SUSPECTED, s["midwife_id"], "sage-femme incertaine")
            return self.menu(s, t("unsure_done", L, rid=rid))
        if bid == "lk:new":
            if rec["state"] == "DUPLICATE_SUSPECTED":
                pass
            pid = link_record(self.store, rid, None, s["midwife_id"])
            return self._registered(s, rid, pid, "nouvelle patiente" if L == "fr" else "new patient")
        if bid.startswith("lk:"):
            c = ctx["cands"][int(bid[3:])]
            pid = c["patient_id"]
            pat = self.store.get_patient(pid)["payload"]
            d = diff(pat.get("fields", {}), self._fields(rid))
            if not d["added"] and not d["changed"]:
                pid = link_record(self.store, rid, pid, s["midwife_id"], accept=set())
                return self._registered(s, rid, pid, f"{d['same']} identiques")
            ctx.update({"pid": pid, "diff": d})
            s["mode"] = "redigit"
            old = pat.get("fields", {})
            new = self._fields(rid)
            lines = "\n".join(f"{i + 1}. {label(f, L)} : {show(old[f]['value'], L)} → {show(new[f]['value'], L)}"
                              for i, f in enumerate(d["changed"][:15]))
            return self.push(s, msg(t("redigit", L, a=len(d["added"]), c=len(d["changed"]), s=d["same"], lines=lines),
                                    [B("rd:new", t("b_add_new", L)), B("rd:all", t("b_take_all", L)),
                                     B("rd:choose", t("b_choose", L)), B("rd:keep", t("b_keep_existing", L))]))
        self.start_linking(s, rid)

    def on_redigit(self, s, bid, text):
        L, ctx = s["lang"], s["ctx"]
        rid, pid, d = ctx["rid"], ctx["pid"], ctx["diff"]
        if bid == "rd:choose":
            s["mode"] = "redigit_choose"
            return self.push(s, msg(t("ask_choose", L)))
        acc = {"rd:new": set(d["added"]), "rd:all": set(d["added"]) | set(d["changed"]), "rd:keep": set()}.get(bid)
        if acc is None:
            return self.start_linking(s, rid)
        pid = link_record(self.store, rid, pid, s["midwife_id"], accept=acc)
        self._registered(s, rid, pid, f"{len(acc)} mis à jour")

    def on_redigit_choose(self, s, bid, text):
        ctx = s["ctx"]
        d = ctx["diff"]
        nums = {int(x) - 1 for x in re.findall(r"\d+", text)}
        acc = set(d["added"]) | {f for i, f in enumerate(d["changed"]) if i in nums}
        pid = link_record(self.store, ctx["rid"], ctx["pid"], s["midwife_id"], accept=acc)
        self._registered(s, ctx["rid"], pid, f"{len(acc)} mis à jour")

    def _registered(self, s, rid, pid, detail):
        L = s["lang"]
        sync = t("sync_soon", L) if self.net.online else t("sync_wait", L)
        self.menu(s, t("registered", L, pid=pid, detail=detail, sync=sync))
        self.kick()

    # ------------------------------------------------------------------ find
    def on_find(self, s, bid, text):
        L = s["lang"]
        code, _ = normalize(SPEC_BY_ID["fiche_number"], text)
        ids = self.store.patients_by_code(code) if code else []
        if not ids:
            return self.menu(s, t("no_patient", L))
        for pid in ids:
            p = self.store.get_patient(pid)["payload"]
            lines = []
            for rid in p.get("records", []):
                r = self.store.get_record(rid)
                when = time.strftime("%d/%m/%Y", time.localtime(r["created_at"]))
                lines.append(f"• {when} — {rid} ({self.STATE_ICON.get(r['state'], '')} {r['state']})")
            self.push(s, msg(t("patient", L, pid=pid, summary=summary(p.get("profile", {})), lines="\n".join(lines)),
                             [B(f"view:{p['records'][-1]}", t("b_view", L))] if p.get("records") else []))
        self.menu(s)

    def on_menu(self, s, bid, text):
        if bid.startswith("view:"):
            return self.push(s, msg("📄", [B("menu", t("b_menu", s["lang"]))]) | {"open_record": bid[5:]})
        self.menu(s)

    # ------------------------------------------------------------------ manual entry (AI unavailable)
    def manual_questions(self, page):
        qs = []
        for f in FIELDS:
            if f.page != page:
                continue
            if f.id.startswith("g_"):
                continue
            if f.importance >= 2 or page in ("cover",) and f.id in ("fiche_number", "region", "province", "facility"):
                qs.append({"kind": "field", "fid": f.id})
        if page == "pregnancy":
            for rid_, rl, ty, imp, en in GRID_ROWS:
                if rid_ in ESSENTIAL_ROWS:
                    qs.append({"kind": "row", "row": rid_, "fids": [f"g_{rid_}_{c}" for c, *_ in VISIT_COLS]})
        return qs

    def start_manual(self, s, rid=None):
        L = s["lang"]
        if rid is None:
            rid = self.store.create_record(s["midwife_id"], {"fields": {}, "pii_polys": {}, "page_types": {},
                                                             "origin": "manual"})
        rec = self.store.get_record(rid)
        if rec["state"] in ("CAPTURED", "PENDING_AI", "PROCESSING_FAILED", "NEEDS_REVIEW"):
            self.store.set_state(rid, S.MANUAL_REVIEW_REQUIRED, s["midwife_id"], "saisie manuelle")
        p = rec["payload"]
        if not p.get("fields"):
            p["fields"] = {f.id: fv(None, "NOT_PROVIDED", 1.0, "manual", None, "non saisi (saisie manuelle)", f.page)
                           for f in FIELDS}
            self.store.update_payload(rid, p)
        s["mode"], s["ctx"] = "manual_pick", {"rid": rid}
        self.push(s, msg(t("manual_intro", L)))
        self._manual_pick(s)

    def _manual_pick(self, s):
        L = s["lang"]
        btns = [B(f"mp:{p}", page_label(p, L)) for p in PAGE_ORDER] + [B("mp:finish", t("b_finish_manual", L))]
        self.push(s, msg(t("manual_page", L), btns))

    def on_manual_pick(self, s, bid, text):
        ctx = s["ctx"]
        if bid == "mp:finish":
            rid = ctx["rid"]
            return self.show_summary(s, rid)
        if bid.startswith("mp:"):
            ctx.update({"page": bid[3:], "qs": self.manual_questions(bid[3:]), "i": 0})
            s["mode"] = "manual"
            return self._manual_next(s)
        self._manual_pick(s)

    def _manual_next(self, s):
        L, ctx = s["lang"], s["ctx"]
        if ctx["i"] >= len(ctx["qs"]):
            s["mode"] = "manual_pick"
            return self._manual_pick(s)
        q = ctx["qs"][ctx["i"]]
        n = len(ctx["qs"])
        if q["kind"] == "row":
            rl = dict((r[0], r[1]) for r in GRID_ROWS)[q["row"]]
            cols = ", ".join(c[3] for c in VISIT_COLS)
            return self.push(s, msg(f"({ctx['i'] + 1}/{n}) *{rl}* — " + t("ask_row", L, cols=cols),
                                    [B("m:skip", t("b_skip", L)), B("m:page", "⏭ page")]))
        fid = q["fid"]
        spec = SPEC_BY_ID[fid]
        btns = [B("m:skip", t("b_skip", L)), B("m:page", "⏭ page")]
        if spec.kind in ("check", "checkrow"):
            btns = [B("m:yes", t("b_ticked", L)), B("m:no", t("b_unticked", L))] + btns
        self.push(s, msg(t("manual_q", L, i=ctx["i"] + 1, n=n, label=label(fid, L), fmt=fmt(spec.type, L)), btns))

    def on_manual(self, s, bid, text):
        L, ctx = s["lang"], s["ctx"]
        rid = ctx["rid"]
        q = ctx["qs"][ctx["i"]]
        if bid == "m:page" or strip_accents(text.lower()) in ("passer la page", "skip page"):
            s["mode"] = "manual_pick"
            return self._manual_pick(s)
        if bid == "m:skip":
            ctx["i"] += 1
            return self._manual_next(s)
        if q["kind"] == "row":
            parts = [p.strip() for p in re.split(r"[;\n]", text) if p.strip()]
            if len(parts) != len(q["fids"]):
                return self.push(s, msg(t("row_count", L, n=len(q["fids"]), got=len(parts)) + " " +
                                        t("ask_row", L, cols=", ".join(c[3] for c in VISIT_COLS))))
            parsed = [self.parse_answer(f, p) for f, p in zip(q["fids"], parts)]
            if not all(ok for ok, *_r in parsed):
                return self.push(s, msg(t("bad_value", L, v=text, label=q["row"], fmt="")))
            for f, (ok, v, st) in zip(q["fids"], parsed):
                self.set_field(rid, f, v, st, "manual")
        else:
            fid = q["fid"]
            if bid in ("m:yes", "m:no"):
                self.set_field(rid, fid, bid == "m:yes", "KNOWN", "manual")
            else:
                ok, v, st = self.parse_answer(fid, text)
                if not ok:
                    return self.push(s, msg(t("bad_value", L, v=text, label=label(fid, L), fmt=fmt(SPEC_BY_ID[fid].type, L))))
                self.set_field(rid, fid, v, st, "manual")
        ctx["i"] += 1
        self._manual_next(s)
