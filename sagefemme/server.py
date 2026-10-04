"""Local demo server: WhatsApp-style web prototype + simulated network + central server.

  python -m sagefemme.server --port 8000 --pin 1234 [--data data/device] [--fresh]

Everything runs in one process for the demo, but the layers stay separated:
  device  = LocalStore (encrypted) + Agent + capture checks   (works offline)
  online  = Worker AI processing + CentralServer sync           (gated by Network)
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import mimetypes
import secrets
import shutil
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .agent import Agent
from .analytics import CSV_COLUMNS, aggregates, to_analytic_row
from .central import CentralServer
from .extraction.pipeline import load_image
from .forms.align import align, templates
from .forms.layout import GRID_ROWS, PAGE_ORDER, PAGE_TYPES, SPEC_BY_ID, VISIT_COLS
from .net import Network
from .schema import FIELDS, STATUS_LABELS, section_label
from .store import LocalStore, WrongPin
from .worker import Worker

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


class App:
    def __init__(self, data_dir: Path, pin: str, online=True):
        self.net = Network(online)
        self.store = LocalStore(data_dir / "device", pin)
        self.central = CentralServer(data_dir / "central.db", self.net)
        hist = ROOT / "data/reference/maternal_registry_synthetic.csv"
        if hist.exists() and not self.central.historical_rows():
            self.central.import_historical(hist)
        self.agent = Agent(self.store, self.net)
        self.worker = Worker(self.store, self.net, self.central, notify=self.agent.notify)
        self.agent.worker = self.worker
        self.net.listeners.append(lambda on: self.agent.notify("network", {"online": on}))
        self.tokens: dict[str, dict] = {}
        self._align_cache: dict[str, object] = {}
        self.pin = pin
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                if self.net.online:
                    self.worker.tick()
            except Exception:
                traceback.print_exc()
            time.sleep(1.5)

    # ---------------------------------------------------------------- crops (role-restricted)
    def _aligned(self, image_id: str):
        if image_id not in self._align_cache:
            img = load_image(self.store._image_bytes_internal(image_id))
            self._align_cache[image_id] = (np.array(img), align(np.array(img)))
        return self._align_cache[image_id]

    def crop(self, rid: str, bbox_fn, page: str, user: dict) -> bytes | None:
        rec = self.store.get_record(rid)
        img_id = {v: k for k, v in rec["payload"].get("page_types", {}).items()}.get(page)
        if not img_id or self.store.read_image(img_id, user["id"], user["role"]) is None:
            return None
        rgb, al = self._aligned(img_id)
        if not al.ok:
            return None
        x0, y0, x1, y1 = bbox_fn(templates()[page]["regions"])
        pts = al.project((x0 - 40, y0 - 8, x1 + 25, y1 + 8))
        xs, ys = pts[:, 0], pts[:, 1]
        h, w = rgb.shape[:2]
        c = rgb[max(0, int(ys.min())):min(h, int(ys.max())), max(0, int(xs.min())):min(w, int(xs.max()))]
        im = Image.fromarray(c)
        if im.width > 6 * im.height:          # long grid row: fold in two for a phone screen
            h2 = im.height
            left, right = im.crop((0, 0, im.width // 2 + 20, h2)), im.crop((im.width // 2 - 20, 0, im.width, h2))
            folded = Image.new("RGB", (left.width, 2 * h2 + 6), (255, 255, 255))
            folded.paste(left, (0, 0))
            folded.paste(right, (0, h2 + 6))
            im = folded
        if im.width < 480:
            im = im.resize((480, int(im.height * 480 / max(im.width, 1))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()

    def record_view(self, rid: str, lang="fr") -> dict:
        rec = self.store.get_record(rid)
        p = rec["payload"]
        pages = []
        for pt in PAGE_ORDER:
            fs = [{"id": f.id, "label": f.fr if lang == "fr" else (f.en or f.fr), "group": f.group,
                   **p.get("fields", {}).get(f.id, {})} for f in FIELDS if f.page == pt]
            pages.append({"page": pt, "label": section_label(pt, lang), "fields": fs})
        imgs = [{"id": r["id"], "page_no": r["page_no"], "page_type": p.get("page_types", {}).get(r["id"]),
                 "captured_at": r["captured_at"], "midwife_id": r["midwife_id"], "processing_status": r["processing_status"],
                 "quality": json.loads(r["quality"] or "{}")} for r in self.store.image_rows(rid)]
        return {"id": rid, "state": rec["state"], "patient_id": rec["patient_id"], "pages": pages, "images": imgs,
                "events": self.store.events(rid), "extraction": p.get("extraction"),
                "analytic": to_analytic_row(p.get("fields", {})) if p.get("fields") else None}

    def state(self) -> dict:
        recs = self.store.records()
        counts = {}
        for r in recs:
            counts[r["state"]] = counts.get(r["state"], 0) + 1
        return {"online": self.net.online, "records": recs[-40:], "counts": counts, "central": self.central.counts(),
                "worker_busy": self.worker.busy, "patients": len(self.store.patients()),
                "inject": {"ai": self.worker.fail_next, "sync": self.central.fail_next, "drop": self.net.drop_rate}}

    def dashboard(self) -> dict:
        hist = self.central.historical_rows()
        dig = [r["analytic"] for r in self.central.all_records() if r.get("analytic")]
        temps = []
        for r in self.central.all_records():
            for fid in ("pme_temp", "pml_temp"):
                f = r["fields"].get(fid)
                if f and f["status"] == "KNOWN" and isinstance(f["value"], (int, float)):
                    temps.append(f["value"])
        out = aggregates(hist + dig, temps)
        out["sources"] = {"historical_csv": len(hist), "digitised_synced": len(dig)}
        return out


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False, default=str).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _bytes(self, b, ctype, code=200):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(b)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def _body(self):
            n = int(self.headers.get("content-length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def _user(self, q=None, body=None):
            tok = (body or {}).get("token") or (q or {}).get("token", [None])[0] or self.headers.get("x-token")
            return app.tokens.get(tok)

        def do_GET(self):
            try:
                u = urlparse(self.path)
                q = parse_qs(u.query)
                path = u.path
                if path == "/" or path == "/index.html":
                    return self._bytes((WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
                if path.startswith("/static/"):
                    f = WEB / path[len("/static/"):]
                    if f.is_file() and WEB in f.resolve().parents:
                        return self._bytes(f.read_bytes(), mimetypes.guess_type(str(f))[0] or "application/octet-stream")
                    return self._json({"error": "not found"}, 404)
                if path == "/api/state":
                    return self._json(app.state())
                if path == "/api/dashboard":
                    return self._json(app.dashboard())
                if path == "/api/samples":
                    d = ROOT / "data/testset"
                    out = {}
                    for lvl in ("clean", "mild", "heavy"):
                        out[lvl] = sorted(p.name for p in (d / lvl).glob("*.jpg")) if (d / lvl).exists() else []
                    return self._json(out)
                if path.startswith("/api/sample/"):
                    _, _, _, lvl, name = path.split("/", 4)
                    f = ROOT / "data/testset" / lvl / name
                    if f.is_file() and lvl in ("clean", "mild", "heavy"):
                        im = Image.open(f)
                        im.thumbnail((360, 480))
                        buf = io.BytesIO()
                        im.convert("RGB").save(buf, "JPEG", quality=70)
                        return self._bytes(buf.getvalue(), "image/jpeg")
                    return self._json({"error": "not found"}, 404)
                user = self._user(q)
                if not user:
                    return self._json({"error": "auth"}, 401)
                s = app.agent.session(user["id"])
                if path == "/api/poll":
                    since = int(q.get("since", ["0"])[0])
                    return self._json({"messages": [m for m in s["outbox"] if m["seq"] > since], "mode": s["mode"],
                                       "lang": s["lang"]})
                if path.startswith("/api/record/"):
                    return self._json(app.record_view(path.split("/")[3], s["lang"]))
                if path.startswith("/api/crop/"):
                    _, _, _, rid, fid = path.split("/")
                    spec = SPEC_BY_ID[fid]
                    b = app.crop(rid, lambda regs: (regs[fid].get("bbox") or regs[fid].get("box")), spec.page, user)
                    return self._bytes(b, "image/jpeg") if b else self._json({"error": "forbidden or unavailable"}, 403)
                if path.startswith("/api/croprow/"):
                    _, _, _, rid, row = path.split("/")
                    def rowbox(regs):
                        bs = [regs[f"g_{row}_{c}"]["bbox"] for c, *_ in VISIT_COLS if f"g_{row}_{c}" in regs]
                        return (25, min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))
                    b = app.crop(rid, rowbox, "pregnancy", user)
                    return self._bytes(b, "image/jpeg") if b else self._json({"error": "forbidden"}, 403)
                if path.startswith("/api/image/"):
                    b = app.store.read_image(path.split("/")[3], user["id"], user["role"])
                    return self._bytes(b, "image/jpeg") if b else self._json({"error": "forbidden (role)"}, 403)
                if path == "/api/export.csv":
                    if user["role"] not in ("supervisor", "analyst"):
                        return self._json({"error": "forbidden (role)"}, 403)
                    buf = io.StringIO()
                    w = csv.DictWriter(buf, fieldnames=["record_id"] + CSV_COLUMNS)
                    w.writeheader()
                    for r in app.central.all_records():
                        if r.get("analytic"):
                            w.writerow({"record_id": r["id"], **r["analytic"]})
                    return self._bytes(buf.getvalue().encode(), "text/csv; charset=utf-8")
                return self._json({"error": "not found"}, 404)
            except Exception as e:
                traceback.print_exc()
                return self._json({"error": str(e)}, 500)

        def do_POST(self):
            try:
                path = urlparse(self.path).path
                body = self._body()
                if path == "/api/login":
                    if body.get("pin") != app.pin:
                        return self._json({"error": "PIN incorrect"}, 403)
                    tok = secrets.token_urlsafe(16)
                    user = {"id": body.get("midwife_id") or "SF-001", "role": body.get("role") or "midwife",
                            "name": body.get("name") or "Sage-femme"}
                    app.tokens[tok] = user
                    s = app.agent.session(user["id"], user["name"], user["role"])
                    if body.get("lang") in ("fr", "en"):
                        s["lang"] = body["lang"]
                    if not s["outbox"]:
                        app.agent.start(s)
                    return self._json({"token": tok, "user": user})
                if path == "/api/network":
                    app.net.set_online(bool(body.get("online")))
                    if "drop_rate" in body:
                        app.net.drop_rate = float(body["drop_rate"])
                    return self._json(app.state())
                if path == "/api/inject":
                    app.worker.fail_next = int(body.get("ai", app.worker.fail_next))
                    app.central.fail_next = int(body.get("sync", app.central.fail_next))
                    return self._json(app.state())
                user = self._user(body=body)
                if not user:
                    return self._json({"error": "auth"}, 401)
                s = app.agent.session(user["id"])
                if path == "/api/send":
                    ev = {"type": body.get("type", "text")}
                    if ev["type"] == "button":
                        ev["id"], ev["label"] = body["id"], body.get("label", body["id"])
                    else:
                        ev["text"] = body.get("text", "")
                    threading.Thread(target=app.agent.handle, args=(s, ev), daemon=True).start()
                    return self._json({"ok": True})
                if path == "/api/photo":
                    data = base64.b64decode(body["data"].split(",")[-1])
                    prev = body.get("preview")
                    threading.Thread(target=app.agent.handle, args=(s, {"type": "photo", "data": data, "preview": prev}),
                                     daemon=True).start()
                    return self._json({"ok": True})
                if path == "/api/sample":
                    f = ROOT / "data/testset" / body["level"] / body["name"]
                    data = f.read_bytes()
                    prev = f"/api/sample/{body['level']}/{body['name']}"
                    threading.Thread(target=app.agent.handle, args=(s, {"type": "photo", "data": data, "preview": prev}),
                                     daemon=True).start()
                    return self._json({"ok": True})
                return self._json({"error": "not found"}, 404)
            except Exception as e:
                traceback.print_exc()
                return self._json({"error": str(e)}, 500)

    return H


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--pin", default="1234")
    ap.add_argument("--data", default=str(ROOT / "data/runtime"))
    ap.add_argument("--fresh", action="store_true", help="wipe the demo device + central DB first")
    ap.add_argument("--offline", action="store_true", help="start with the network cut")
    a = ap.parse_args(argv)
    d = Path(a.data)
    if a.fresh and d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    try:
        app = App(d, a.pin, online=not a.offline)
    except WrongPin:
        raise SystemExit("PIN incorrect for this device store (use --fresh to reset the demo)")
    print(f"SageFemme agent prototype on http://{a.host}:{a.port}  (PIN {a.pin})")
    ThreadingHTTPServer((a.host, a.port), make_handler(app)).serve_forever()


if __name__ == "__main__":
    main()
