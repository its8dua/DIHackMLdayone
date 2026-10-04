"""Unit tests: schema/status model, lifecycle, encryption, offline queue, linking, privacy.

  python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sagefemme.central import CentralServer  # noqa: E402
from sagefemme.extraction.postprocess import consistency, fv, scrub_pii  # noqa: E402
from sagefemme.forms.layout import SPEC_BY_ID  # noqa: E402
from sagefemme.linking import find_candidates, link_record  # noqa: E402
from sagefemme.net import Network, OfflineError  # noqa: E402
from sagefemme.schema import FIELDS, IllegalTransition, RecordState, check_transition, normalize  # noqa: E402
from sagefemme.store import LocalStore, WrongPin  # noqa: E402
from sagefemme.whatsapp import parse_webhook, to_whatsapp, verify_signature  # noqa: E402
from sagefemme.worker import Worker  # noqa: E402


def fake_fields(code="2026-111-001", age=27, edd="2026-05-01"):
    out = {f.id: fv(None, "NOT_PROVIDED", 1.0, "ai", None, "", f.page) for f in FIELDS}
    out["fiche_number"] = fv(code, "KNOWN", 0.99, "midwife_confirmed", code)
    out["age"] = fv(age, "KNOWN", 0.95, "ai", str(age))
    out["edd"] = fv(edd, "KNOWN", 0.95, "ai", edd)
    out["province"] = fv("Azilal", "KNOWN", 0.95, "ai", "Azilal")
    return out


class TestSchema(unittest.TestCase):
    def test_normalize_types(self):
        S = SPEC_BY_ID
        self.assertEqual(normalize(S["nb_weight"], "3587 g"), (3587, True))
        self.assertEqual(normalize(S["g_ga_m9"], "38 SA"), (38, True))
        self.assertEqual(normalize(S["g_bp_m9"], "12/8"), ("120/80", True))      # cmHg
        self.assertEqual(normalize(S["g_bp_m9"], "60/120")[1], False)          # diastolic > systolic
        self.assertEqual(normalize(S["lmp"], "26/04/2025"), ("2025-04-26", True))
        self.assertEqual(normalize(S["lmp"], "45/04/2025")[1], False)
        self.assertEqual(normalize(S["g_hiv_t1v1"], "Neg")[0], "négatif")
        self.assertEqual(normalize(S["g_iron_t1v1"], "Oui"), (True, True))
        self.assertEqual(normalize(S["age"], "7")[1], False)                   # out of range

    def test_every_field_has_a_page_and_unique_id(self):
        ids = [f.id for f in FIELDS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(len(ids), 500)

    def test_lifecycle(self):
        check_transition("CAPTURED", "PENDING_AI")
        check_transition("PENDING_AI", "AI_PROCESSED")
        check_transition("REGISTERED", "SYNCED")
        with self.assertRaises(IllegalTransition):
            check_transition("CAPTURED", "SYNCED")
        with self.assertRaises(IllegalTransition):
            check_transition("PENDING_AI", "REGISTERED")     # cannot skip human validation

    def test_consistency_flags_contradictions(self):
        f = fake_fields()
        f["lmp"] = fv("2025-01-01", "KNOWN", 0.95, "ai")        # 280 d before EDD would be 2025-07-25
        consistency(f)
        self.assertEqual(f["lmp"]["status"], "NEEDS_REVIEW")
        self.assertEqual(f["edd"]["status"], "NEEDS_REVIEW")
        g = fake_fields()
        g["lmp"] = fv("2025-07-25", "NEEDS_REVIEW", 0.6, "ai")
        consistency(g)
        self.assertEqual(g["lmp"]["status"], "KNOWN")           # agreement raises confidence

    def test_pii_scrub(self):
        self.assertNotIn("0600761348", scrub_pii("appeler 06 00 76 13 48 svp").replace(" ", ""))
        self.assertIn("[masqué]", scrub_pii("CIN CB609814"))


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.store = LocalStore(self.dir / "dev", "1234")
        self.net = Network(True)
        self.central = CentralServer(self.dir / "central.db", self.net)


class TestStore(StoreCase):
    def test_encrypted_at_rest(self):
        rid = self.store.create_record("SF-1", {"fields": fake_fields(code="ZZTOP-SECRET-42")})
        self.store.save_image(rid, 1, b"\xff\xd8 fake jpeg NOM-PATIENTE", "SF-1", {})
        raw = (self.dir / "dev/device.db").read_bytes()
        for p in (self.dir / "dev/blobs").iterdir():
            raw += p.read_bytes()
        self.assertNotIn(b"ZZTOP-SECRET-42", raw)
        self.assertNotIn(b"NOM-PATIENTE", raw)
        self.assertEqual(self.store.get_record(rid)["payload"]["fields"]["fiche_number"]["value"], "ZZTOP-SECRET-42")

    def test_wrong_pin(self):
        with self.assertRaises(WrongPin):
            LocalStore(self.dir / "dev", "0000")

    def test_role_based_image_access_is_logged(self):
        rid = self.store.create_record("SF-1", {"fields": {}})
        img = self.store.save_image(rid, 1, b"img", "SF-1", {})
        self.assertEqual(self.store.read_image(img, "SF-1", "midwife"), b"img")
        self.assertIsNone(self.store.read_image(img, "EPI-1", "analyst"))
        log = list(self.store.db.execute("SELECT role, granted FROM access_log"))
        self.assertEqual([tuple(r) for r in log], [("midwife", 1), ("analyst", 0)])

    def test_ids_are_random_not_personal(self):
        pid = self.store.create_patient({"code": "2026-1", "profile": {"age": 30}})
        self.assertRegex(pid, r"^PAT-[0-9A-Z]{4}-[0-9A-Z]{4}$")


class TestOffline(StoreCase):
    """Cut connectivity mid-process: nothing may be lost or corrupted."""

    def _pending(self, n=3):
        ids = []
        for i in range(n):
            rid = self.store.create_record("SF-1", {"fields": {}, "page_types": {}, "pii_polys": {}})
            self.store.save_image(rid, 1, b"x", "SF-1", {})
            self.store.set_state(rid, "PENDING_AI", "SF-1")
            ids.append(rid)
        return ids

    def test_drop_during_processing_keeps_records_pending(self):
        ids = self._pending(3)
        calls = {"n": 0}
        import sagefemme.worker as W

        def fake_extract(imgs, net, use_vlm=None):
            calls["n"] += 1
            if calls["n"] == 2:
                self.net.set_online(False)           # connection drops in the middle of the batch
            net.require("AI")
            return {"fields": fake_fields(), "pages": [{"page_type": "cover", "backend": ["test"]}],
                    "consistency": [], "seconds": 0.1, "backends": ["test"]}
        orig = W.extract_record
        W.extract_record = fake_extract
        try:
            w = Worker(self.store, self.net, self.central)
            w.process_pending()
        finally:
            W.extract_record = orig
        states = [self.store.get_record(r)["state"] for r in ids]
        self.assertEqual(states, ["NEEDS_REVIEW", "PENDING_AI", "PENDING_AI"])
        self.assertTrue(all(len(self.store.image_rows(r)) == 1 for r in ids))   # images intact

    def test_offline_does_nothing_and_back_online_processes(self):
        ids = self._pending(2)
        self.net.set_online(False)
        import sagefemme.worker as W
        orig = W.extract_record
        W.extract_record = lambda imgs, net, use_vlm=None: (net.require("AI"), {
            "fields": fake_fields(), "pages": [{}], "consistency": [], "seconds": 0, "backends": ["t"]})[1]
        try:
            w = Worker(self.store, self.net, self.central)
            self.assertEqual(w.process_pending(), 0)
            self.assertEqual({self.store.get_record(r)["state"] for r in ids}, {"PENDING_AI"})
            self.net.set_online(True)
            self.assertEqual(w.process_pending(), 2)
        finally:
            W.extract_record = orig

    def test_processing_failures_end_in_failure_state_after_retries(self):
        (rid,) = self._pending(1)
        w = Worker(self.store, self.net, self.central)
        w.fail_next = 5
        for _ in range(3):
            w.process_pending()
        self.assertEqual(self.store.get_record(rid)["state"], "PROCESSING_FAILED")
        self.assertEqual(self.store.get_record(rid)["attempts"], 3)

    def _registered(self, code="2026-111-001"):
        rid = self.store.create_record("SF-1", {"fields": fake_fields(code), "page_types": {}, "pii_polys": {}})
        for st in ("PENDING_AI", "AI_PROCESSED", "NEEDS_REVIEW", "VALIDATED"):
            self.store.set_state(rid, st, "t")
        pid = link_record(self.store, rid, None, "SF-1")
        return rid, pid

    def test_sync_is_idempotent_when_ack_is_lost(self):
        rid, pid = self._registered()
        w = Worker(self.store, self.net, self.central)
        orig = self.central.push

        def push_then_drop(*a, **k):
            r = orig(*a, **k)                  # server committed...
            raise OfflineError("ack lost")     # ...but the phone never got the answer
        self.central.push = push_then_drop
        w.sync()
        self.assertEqual(self.store.get_record(rid)["state"], "REGISTERED")
        self.central.push = orig
        w.sync()
        self.assertEqual(self.store.get_record(rid)["state"], "SYNCED")
        self.assertEqual(self.central.counts()["records"], 1)   # no duplicate on the server

    def test_sync_failure_then_retry(self):
        rid, pid = self._registered()
        self.central.fail_next = 1
        w = Worker(self.store, self.net, self.central)
        w.sync()
        self.assertEqual(self.store.get_record(rid)["state"], "SYNC_FAILED")
        self.store.db.execute("UPDATE records SET updated_at=updated_at-999 WHERE id=?", (rid,))
        w.sync()
        self.assertEqual(self.store.get_record(rid)["state"], "SYNCED")


class TestLinking(StoreCase):
    def _validated(self, fields):
        rid = self.store.create_record("SF-1", {"fields": fields, "page_types": {}, "pii_polys": {}})
        for st in ("PENDING_AI", "AI_PROCESSED", "NEEDS_REVIEW", "VALIDATED"):
            self.store.set_state(rid, st, "t")
        return rid

    def test_exact_and_fuzzy_code_candidates_never_autocreate(self):
        r1 = self._validated(fake_fields("2026-823-001"))
        p1 = link_record(self.store, r1, None, "SF-1")
        # same registry re-photographed, OCR slip in the code
        cands = find_candidates(self.store, fake_fields("2026-828-001"))
        self.assertEqual(cands[0]["patient_id"], p1)
        self.assertGreaterEqual(cands[0]["score"], 0.35)
        # unrelated woman
        self.assertEqual(find_candidates(self.store, fake_fields("2031-555-999", age=41, edd="2027-02-01")), [])
        self.assertEqual(len(self.store.patients()), 1)

    def test_redigitisation_merge_respects_choice(self):
        r1 = self._validated(fake_fields("2026-823-001", age=27))
        p1 = link_record(self.store, r1, None, "SF-1")
        newf = fake_fields("2026-823-001", age=28)
        newf["nb_weight"] = fv(3200, "KNOWN", 0.95, "ai")
        r2 = self._validated(newf)
        link_record(self.store, r2, p1, "SF-1", accept={"nb_weight"})   # add new info, keep age
        p = self.store.get_patient(p1)["payload"]
        self.assertEqual(p["fields"]["age"]["value"], 27)
        self.assertEqual(p["fields"]["nb_weight"]["value"], 3200)
        self.assertEqual(p["records"], [r1, r2])

    def test_central_flags_cross_device_duplicates(self):
        r1 = self._validated(fake_fields("2026-777-001"))
        link_record(self.store, r1, None, "SF-1")
        other = LocalStore(self.dir / "dev2", "9999", device_id="DEV-02")
        r2 = other.create_record("SF-2", {"fields": fake_fields("2026-777-001"), "page_types": {}, "pii_polys": {}})
        for st in ("PENDING_AI", "AI_PROCESSED", "NEEDS_REVIEW", "VALIDATED"):
            other.set_state(r2, st, "t")
        link_record(other, r2, None, "SF-2")
        Worker(self.store, self.net, self.central).sync()
        Worker(other, self.net, self.central).sync()
        self.assertEqual(other.get_record(r2)["state"], "DUPLICATE_SUSPECTED")


class TestWhatsApp(unittest.TestCase):
    def test_button_mapping(self):
        m = {"text": "Q?", "buttons": [{"id": "a", "label": "✔ Correct"}, {"id": "b", "label": "✏️ Corriger"}]}
        p = to_whatsapp("212600000000", m)[0]
        self.assertEqual(p["interactive"]["type"], "button")
        m["buttons"] = [{"id": str(i), "label": f"Option {i}"} for i in range(6)]
        self.assertEqual(to_whatsapp("x", m)[0]["interactive"]["type"], "list")

    def test_webhook_and_signature(self):
        body = {"entry": [{"changes": [{"value": {"messages": [
            {"from": "2126", "type": "interactive", "interactive": {"button_reply": {"id": "t:ok", "title": "OK"}}},
            {"from": "2126", "type": "image", "image": {"id": "MEDIA1"}}]}}]}]}
        ev = parse_webhook(body)
        self.assertEqual(ev[0][1]["id"], "t:ok")
        self.assertEqual(ev[1][1]["media_id"], "MEDIA1")
        import hashlib, hmac
        raw = json.dumps(body).encode()
        sig = "sha256=" + hmac.new(b"s3cret", raw, hashlib.sha256).hexdigest()
        self.assertTrue(verify_signature(raw, sig, "s3cret"))
        self.assertFalse(verify_signature(raw + b" ", sig, "s3cret"))


if __name__ == "__main__":
    unittest.main()
