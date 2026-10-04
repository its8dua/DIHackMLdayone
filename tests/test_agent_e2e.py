"""End-to-end conversation on real test-set photos (≈1 min, local OCR backend).

Offline capture -> connectivity returns -> AI processing -> review of an uncertain field
-> validation -> patient linking -> sync; then the same registry is photographed again
and the agent proposes the existing patient instead of creating a duplicate.
"""
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sagefemme.agent import Agent  # noqa: E402
from sagefemme.central import CentralServer  # noqa: E402
from sagefemme.net import Network  # noqa: E402
from sagefemme.store import LocalStore  # noqa: E402
from sagefemme.worker import Worker  # noqa: E402

PAGES = ["cover", "identification", "delivery"]
TESTSET = ROOT / "data/testset/mild"


@unittest.skipUnless(TESTSET.exists(), "run tools/build_dataset.py first")
class TestConversation(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self.net = Network(False)
        self.store = LocalStore(d / "dev", "1234")
        self.central = CentralServer(d / "c.db", self.net)
        self.agent = Agent(self.store, self.net)
        self.worker = Worker(self.store, self.net, self.central, notify=self.agent.notify, use_vlm=False)
        self.s = self.agent.session("SF-001", "Amina")
        self.agent.start(self.s)

    def bot(self):
        return [m for m in self.s["outbox"] if m["from"] == "bot"][-1]

    def press(self, bid):
        self.agent.handle(self.s, {"type": "button", "id": bid})
        return self.bot()

    def say(self, text):
        self.agent.handle(self.s, {"type": "text", "text": text})
        return self.bot()

    def capture(self, patient="P01"):
        self.press("menu:new")
        for pt in PAGES:
            self.agent.handle(self.s, {"type": "photo", "data": (TESTSET / f"{patient}_{pt}.jpg").read_bytes()})
            self.assertIn("✅", self.bot()["text"])
        return self.press("cap:done")

    def review_all(self, code):
        rid = self.store.records()[-1]["id"]
        m = self.press(f"open:{rid}")
        if self.s["mode"] == "review_scope":
            m = self.press("scope:all")
        for _ in range(200):
            if self.s["mode"] != "review":
                break
            ids = [b["id"] for b in m["buttons"]]
            task = self.s["ctx"]["tasks"][self.s["ctx"]["i"]]
            if task["kind"] == "code":
                self.assertIn("N° de la fiche", m["text"])
                self.assertIsNotNone(m["image"])                 # photo excerpt shown with the question
                self.press("t:edit")
                m = self.say(code)
            elif "t:tick" in ids:
                m = self.press("t:tick")
            elif "t:ok" in ids:
                m = self.press("t:ok")
            elif "t:rowok" in ids:
                m = self.press("t:rowok")
            else:
                m = self.press("t:blank")
        self.assertEqual(self.s["mode"], "summary")
        return rid

    def test_full_flow(self):
        m = self.capture()
        self.assertIn("En attente de traitement IA", m["text"])          # offline capture
        rid = self.store.records()[-1]["id"]
        self.assertEqual(self.store.get_record(rid)["state"], "PENDING_AI")
        self.worker.process_pending()                                      # still offline: no-op
        self.assertEqual(self.store.get_record(rid)["state"], "PENDING_AI")

        self.net.set_online(True)                                          # connectivity returns
        self.assertEqual(self.worker.process_pending(), 1)
        self.assertEqual(self.store.get_record(rid)["state"], "NEEDS_REVIEW")
        fields = self.store.get_record(rid)["payload"]["fields"]
        self.assertEqual(fields["g_bp_m9"]["note"], "page non photographiée")
        self.assertNotIn("Tazi", str(fields))                              # name never extracted

        self.review_all("2026-823-001")
        m = self.press("s:validate")
        self.assertIn("Aucune patiente", m["text"])                       # no plausible match yet
        m = self.press("lk:new")
        self.assertIn("PAT-", m["text"])
        self.worker.sync()
        self.assertEqual(self.store.get_record(rid)["state"], "SYNCED")

        # same registry photographed again two months later
        self.capture()
        self.worker.process_pending()
        self.review_all("2026-823-001")
        m = self.press("s:validate")
        self.assertIn("possible", m["text"])                              # existing patient proposed
        self.assertTrue(any(b["id"] == "lk:0" for b in m["buttons"]))
        self.assertTrue(any(b["id"] == "lk:unsure" for b in m["buttons"]))
        m = self.press("lk:0")
        if self.s["mode"] == "redigit":
            m = self.press("rd:new")
        self.assertEqual(len(self.store.patients()), 1)                    # no duplicate patient
        self.assertEqual(len(self.store.patients()[0]["payload"]["records"]), 2)


if __name__ == "__main__":
    unittest.main()
