"""Simulated central health-system database (the online layer).

Receives synced patients/records idempotently and flags cross-device duplicates: if another
device already registered a different patient ID with the same paper code, the record comes
back as DUPLICATE_SUSPECTED for the midwife to resolve (the system never merges by itself).

Only de-identified structured data and the encrypted original image reach this server.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .net import Network


class SyncRejected(Exception):
    pass


class CentralServer:
    def __init__(self, path: Path | str, network: Network, fail_next: int = 0):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS patients(id TEXT PRIMARY KEY, code_idx TEXT, device_id TEXT, payload TEXT, ts REAL);
        CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY, patient_id TEXT, payload TEXT, ts REAL);
        CREATE TABLE IF NOT EXISTS images(id TEXT PRIMARY KEY, record_id TEXT, blob BLOB, meta TEXT);
        """)
        self.net = network
        self.fail_next = fail_next  # inject server errors (tests / demo)
        self._lock = threading.Lock()

    def _maybe_fail(self):
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("central server error 503 (injected)")

    def lookup_code(self, code_idx: str, exclude_device: str) -> list[dict]:
        """Central duplicate search (online only)."""
        self.net.require("central lookup")
        rows = self.db.execute("SELECT id, payload FROM patients WHERE code_idx=? AND device_id<>?",
                               (code_idx, exclude_device)).fetchall()
        return [{"id": r[0], **json.loads(r[1])} for r in rows]

    def push(self, device_id: str, patient: dict, record: dict, images: list[dict]) -> dict:
        """Idempotent upsert. Returns {'status': 'ok'|'duplicate', 'others': [...]}."""
        self.net.require("sync")
        with self._lock:
            self._maybe_fail()
            others = [r[0] for r in self.db.execute(
                "SELECT id FROM patients WHERE code_idx=? AND id<>?", (patient["code_idx"], patient["id"]))]
            self.db.execute("BEGIN")
            self.db.execute("INSERT OR REPLACE INTO patients VALUES(?,?,?,?,?)",
                            (patient["id"], patient["code_idx"], device_id, json.dumps(patient["summary"]), time.time()))
            self.db.execute("INSERT OR REPLACE INTO records VALUES(?,?,?,?)",
                            (record["id"], patient["id"],
                             json.dumps({"fields": record["fields"], "analytic": record.get("analytic")}), time.time()))
            for im in images:
                self.db.execute("INSERT OR IGNORE INTO images VALUES(?,?,?,?)",
                                (im["id"], record["id"], im["blob"], json.dumps(im["meta"])))
            self.db.execute("COMMIT")
            # connection may drop right after the server committed but before the ack arrives:
            self.net.require("sync ack")
            return {"status": "duplicate" if others else "ok", "others": others}

    def counts(self) -> dict:
        return {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("patients", "records", "images")}

    def all_records(self) -> list[dict]:
        out = []
        for r in self.db.execute("SELECT id, patient_id, payload FROM records"):
            d = json.loads(r[2])
            out.append({"id": r[0], "patient_id": r[1], "fields": d.get("fields", {}), "analytic": d.get("analytic")})
        return out

    def import_historical(self, csv_path) -> int:
        """Load the organisers' tabular registry (already de-identified) for the dashboard."""
        import csv
        self.db.execute("CREATE TABLE IF NOT EXISTS historical(id TEXT PRIMARY KEY, row TEXT)")
        n = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = row.pop("id")
                self.db.execute("INSERT OR REPLACE INTO historical VALUES(?,?)", (f"H{rid}", json.dumps(row)))
                n += 1
        return n

    def historical_rows(self) -> list[dict]:
        try:
            return [json.loads(r[0]) for r in self.db.execute("SELECT row FROM historical")]
        except sqlite3.OperationalError:
            return []
