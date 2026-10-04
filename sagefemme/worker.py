"""Offline queue: AI processing and synchronisation, modelled as a state machine.

Guarantees
* A captured record is committed (encrypted, WAL, fsync) before anything else happens.
* Processing never moves a record forward unless the result is committed in the same
  transaction; if the connection drops mid-way the record simply stays PENDING_AI.
* Sync is idempotent (server upserts by record id); a lost acknowledgement just means the
  same record is pushed again later.
"""
from __future__ import annotations

import io
import threading
import traceback
from typing import Callable, Optional

from PIL import Image

from .analytics import to_analytic_row
from .central import CentralServer
from .extraction.pipeline import extract_record, load_image
from .extraction.vlm import redact
from .net import Network, OfflineError
from .schema import RecordState as S
from .store import LocalStore

MAX_ATTEMPTS = 3


class Worker:
    def __init__(self, store: LocalStore, net: Network, central: CentralServer,
                 notify: Optional[Callable[[str, dict], None]] = None, use_vlm: Optional[bool] = None):
        self.store, self.net, self.central = store, net, central
        self.notify = notify or (lambda kind, data: None)
        self.use_vlm = use_vlm
        self._lock = threading.Lock()
        self.busy = None
        self.fail_next = 0          # inject AI-service failures (demo / tests)

    # ------------------------------------------------------------------ AI processing
    def process_pending(self) -> int:
        done = 0
        if not self.net.online:
            return 0
        with self._lock:
            for r in self.store.records([S.PENDING_AI]):
                rid = r["id"]
                rec = self.store.get_record(rid)
                imgs = [self.store._image_bytes_internal(im["id"]) for im in self.store.image_rows(rid)]
                self.busy = rid
                try:
                    if self.fail_next > 0:
                        self.fail_next -= 1
                        raise RuntimeError("AI service error 500 (injected)")
                    out = extract_record(imgs, self.net, use_vlm=self.use_vlm)
                except OfflineError:
                    self.busy = None
                    self.notify("offline_during_processing", {"record": rid})
                    break                    # stays PENDING_AI, nothing lost
                except Exception as e:
                    self.busy = None
                    traceback.print_exc()
                    with self.store.tx() as db:
                        att = r["attempts"] + 1
                        if att >= MAX_ATTEMPTS:
                            self.store.set_state(rid, S.PROCESSING_FAILED, "worker", str(e)[:200], db=db,
                                                 error=str(e)[:500], bump_attempts=True)
                            self.notify("processing_failed", {"record": rid, "error": str(e)[:200]})
                        else:
                            self.store.set_state(rid, S.PENDING_AI, "worker", "retry", db=db,
                                                 error=str(e)[:500], bump_attempts=True)
                    continue
                self.busy = None
                payload = rec["payload"]
                old = payload.get("fields") or {}
                retaken = set(payload.pop("retake_pages", []) or [])
                merged = out["fields"]
                for fid, f in old.items():   # keep what the midwife already confirmed elsewhere
                    if str(f.get("source", "")).startswith(("midwife", "manual")) and f.get("page") not in retaken:
                        merged[fid] = f
                payload["fields"] = merged
                payload["extraction"] = {k: v for k, v in out.items() if k != "fields"}
                # remember page types per image (for crops / retakes)
                rows = self.store.image_rows(rid)
                for im, pm in zip(rows, out["pages"]):
                    payload.setdefault("page_types", {})[im["id"]] = pm.get("page_type")
                with self.store.tx() as db:
                    self.store.update_payload(rid, payload, db=db)
                    self.store.set_state(rid, S.AI_PROCESSED, "worker", ",".join(out["backends"]), db=db)
                    self.store.set_state(rid, S.NEEDS_REVIEW, "worker", "", db=db)
                done += 1
                c = {}
                for f in out["fields"].values():
                    c[f["status"]] = c.get(f["status"], 0) + 1
                self.notify("processed", {"record": rid, "counts": c, "seconds": out["seconds"],
                                          "midwife": rec["midwife_id"]})
        return done

    # ------------------------------------------------------------------ sync
    def sync(self) -> int:
        n = 0
        if not self.net.online:
            return 0
        import time as _t
        for r in self.store.records([S.REGISTERED, S.SYNC_FAILED]):
            rid = r["id"]
            if r["state"] == "SYNC_FAILED" and _t.time() - r["updated_at"] < min(300, 5 * 2 ** r["attempts"]):
                continue                      # exponential back-off between retries
            rec = self.store.get_record(rid)
            pat = self.store.get_patient(rec["patient_id"])
            payload = rec["payload"]
            images = []
            for im in self.store.image_rows(rid):
                raw = self.store._image_bytes_internal(im["id"])
                polys = (payload.get("pii_polys") or {}).get(im["id"], [])
                red = redact(load_image(raw), polys)          # identifiers never leave the device
                buf = io.BytesIO()
                red.save(buf, "JPEG", quality=80)
                images.append({"id": im["id"], "blob": buf.getvalue(),
                               "meta": {"page_no": im["page_no"], "captured_at": im["captured_at"],
                                        "midwife_id": im["midwife_id"], "device_id": im["device_id"],
                                        "sha256_original": im["sha256"]}})
            try:
                res = self.central.push(self.store.device_id,
                                        {"id": pat["id"], "code_idx": pat["code_idx"],
                                         "summary": {"profile": {k: v for k, v in pat["payload"]["profile"].items()}}},
                                        {"id": rid, "fields": payload["fields"],
                                         "analytic": to_analytic_row(payload["fields"])}, images)
            except OfflineError:
                self.notify("offline_during_sync", {"record": rid})
                break
            except Exception as e:
                self.store.set_state(rid, S.SYNC_FAILED, "worker", str(e)[:200], error=str(e)[:500],
                                     bump_attempts=True)
                if r["attempts"] == 0:
                    self.notify("sync_failed", {"record": rid, "error": str(e)[:200]})
                continue
            with self.store.tx() as db:
                self.store.set_state(rid, S.SYNCED, "worker", "", db=db)
                db.execute("UPDATE patients SET synced=1 WHERE id=?", (pat["id"],))
                if res["status"] == "duplicate":
                    self.store.set_state(rid, S.DUPLICATE_SUSPECTED, "central",
                                         "même N° de fiche sur un autre appareil: " + ",".join(res["others"]), db=db)
            n += 1
            self.notify("duplicate_central" if res["status"] == "duplicate" else "synced",
                        {"record": rid, "patient": pat["id"], "others": res.get("others", [])})
        return n

    def tick(self):
        try:
            self.process_pending()
            self.sync()
        except OfflineError:
            pass
