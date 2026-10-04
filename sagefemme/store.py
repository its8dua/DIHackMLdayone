"""Encrypted on-device storage (works fully offline).

* Key derivation: scrypt(PIN, per-device salt) -> 32-byte key -> Fernet (AES-128-CBC + HMAC-SHA256).
* Every record payload, patient payload and image blob is encrypted at rest.
  Only non-sensitive metadata needed for queueing (ids, state, timestamps) is in clear.
* The patient code written on the paper is indexed through HMAC(key, code) so it can be
  looked up without being stored in clear.
* All writes are single SQLite transactions (WAL) -> a crash or a dropped connection in the
  middle of an operation can never lose a captured record.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .schema import RecordState, check_transition

CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def random_id(prefix: str, n: int = 8) -> str:
    """Opaque random identifier. Never derived from personal information."""
    s = "".join(secrets.choice(CROCKFORD) for _ in range(n))
    return f"{prefix}-{s[:4]}-{s[4:]}"


def now() -> float:
    return time.time()


class WrongPin(Exception):
    pass


class Vault:
    def __init__(self, data_dir: Path, pin: str):
        data_dir.mkdir(parents=True, exist_ok=True)
        salt_f = data_dir / "device.salt"
        if not salt_f.exists():
            salt_f.write_bytes(os.urandom(16))
        salt = salt_f.read_bytes()
        key = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1).derive(pin.encode())
        # The code index must be comparable across devices (central duplicate search), so it uses
        # a deployment-wide secret provisioned with the app, not the device key.
        self._mac_key = hashlib.sha256(os.environ.get("SAGEFEMME_INDEX_KEY", "demo-index-key").encode()).digest()
        self.f = Fernet(base64.urlsafe_b64encode(key))
        canary = data_dir / "device.canary"
        if canary.exists():
            try:
                self.f.decrypt(canary.read_bytes())
            except InvalidToken:
                raise WrongPin("PIN incorrect")
        else:
            canary.write_bytes(self.f.encrypt(b"ok"))

    def enc(self, obj: Any) -> bytes:
        return self.f.encrypt(json.dumps(obj, ensure_ascii=False).encode())

    def dec(self, blob: bytes) -> Any:
        return json.loads(self.f.decrypt(blob))

    def enc_bytes(self, b: bytes) -> bytes:
        return self.f.encrypt(b)

    def dec_bytes(self, b: bytes) -> bytes:
        return self.f.decrypt(b)

    def index(self, s: str) -> str:
        return hmac.new(self._mac_key, s.encode(), "sha256").hexdigest()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS records(
  id TEXT PRIMARY KEY, state TEXT NOT NULL, midwife_id TEXT NOT NULL,
  patient_id TEXT, created_at REAL, updated_at REAL, attempts INTEGER DEFAULT 0,
  last_error TEXT, enc BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS images(
  id TEXT PRIMARY KEY, record_id TEXT NOT NULL, page_no INTEGER, captured_at REAL,
  midwife_id TEXT, device_id TEXT, sha256 TEXT, quality TEXT, processing_status TEXT, path TEXT,
  superseded INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS patients(
  id TEXT PRIMARY KEY, code_idx TEXT, created_at REAL, updated_at REAL, synced INTEGER DEFAULT 0,
  enc BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS patients_code ON patients(code_idx);
CREATE TABLE IF NOT EXISTS events(
  ts REAL, record_id TEXT, from_state TEXT, to_state TEXT, actor TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS access_log(ts REAL, user_id TEXT, role TEXT, image_id TEXT, granted INTEGER);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""


class LocalStore:
    def __init__(self, data_dir: Path | str, pin: str, device_id: str = "DEV-01"):
        self.dir = Path(data_dir)
        self.vault = Vault(self.dir, pin)
        self.blob_dir = self.dir / "blobs"
        self.blob_dir.mkdir(exist_ok=True)
        self.device_id = device_id
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.dir / "device.db", check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA_SQL)

    # ------------------------------------------------------------------ tx helper
    def tx(self):
        store = self

        class _Tx:
            def __enter__(self_):
                store._lock.acquire()
                store.db.execute("BEGIN IMMEDIATE")
                return store.db

            def __exit__(self_, et, ev, tb):
                try:
                    store.db.execute("COMMIT" if et is None else "ROLLBACK")
                finally:
                    store._lock.release()
                return False

        return _Tx()

    # ------------------------------------------------------------------ images
    def save_image(self, record_id: str, page_no: int, data: bytes, midwife_id: str,
                   quality: dict, db=None) -> str:
        img_id = random_id("IMG")
        path = self.blob_dir / f"{img_id}.bin"
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(self.vault.enc_bytes(data))
        os.replace(tmp, path)  # atomic
        (db or self.db).execute(
            "INSERT INTO images VALUES(?,?,?,?,?,?,?,?,?,?,0)",
            (img_id, record_id, page_no, now(), midwife_id, self.device_id,
             hashlib.sha256(data).hexdigest(), json.dumps(quality), "CAPTURED", str(path)))
        return img_id

    def image_rows(self, record_id: str, include_superseded=False) -> list[sqlite3.Row]:
        q = "SELECT * FROM images WHERE record_id=?" + ("" if include_superseded else " AND superseded=0")
        return list(self.db.execute(q + " ORDER BY page_no, captured_at", (record_id,)))

    def read_image(self, image_id: str, user_id: str, role: str) -> Optional[bytes]:
        """Role-based access to original paper images. Every attempt is logged."""
        granted = role in ("midwife", "supervisor")
        self.db.execute("INSERT INTO access_log VALUES(?,?,?,?,?)", (now(), user_id, role, image_id, int(granted)))
        if not granted:
            return None
        row = self.db.execute("SELECT path FROM images WHERE id=?", (image_id,)).fetchone()
        if not row:
            return None
        return self.vault.dec_bytes(Path(row["path"]).read_bytes())

    def _image_bytes_internal(self, image_id: str) -> bytes:
        row = self.db.execute("SELECT path FROM images WHERE id=?", (image_id,)).fetchone()
        return self.vault.dec_bytes(Path(row["path"]).read_bytes())

    # ------------------------------------------------------------------ records
    def create_record(self, midwife_id: str, payload: dict) -> str:
        rid = random_id("REC")
        with self.tx() as db:
            db.execute("INSERT INTO records(id,state,midwife_id,created_at,updated_at,enc) VALUES(?,?,?,?,?,?)",
                       (rid, RecordState.CAPTURED.value, midwife_id, now(), now(), self.vault.enc(payload)))
            db.execute("INSERT INTO events VALUES(?,?,?,?,?,?)", (now(), rid, None, "CAPTURED", midwife_id, ""))
        return rid

    def get_record(self, rid: str) -> Optional[dict]:
        row = self.db.execute("SELECT * FROM records WHERE id=?", (rid,)).fetchone()
        if not row:
            return None
        rec = dict(row)
        rec["payload"] = self.vault.dec(rec.pop("enc"))
        return rec

    def update_payload(self, rid: str, payload: dict, db=None):
        (db or self.db).execute("UPDATE records SET enc=?, updated_at=? WHERE id=?",
                                (self.vault.enc(payload), now(), rid))

    def set_state(self, rid: str, new: RecordState | str, actor: str, detail: str = "", db=None,
                  error: Optional[str] = None, bump_attempts: bool = False):
        if db is None:
            with self.tx() as conn:
                return self.set_state(rid, new, actor, detail, conn, error, bump_attempts)
        new = RecordState(new)
        conn = db
        cur =conn.execute("SELECT state FROM records WHERE id=?", (rid,)).fetchone()["state"]
        check_transition(cur, new)
        conn.execute("UPDATE records SET state=?, updated_at=?, last_error=?, attempts=attempts+? WHERE id=?",
                     (new.value, now(), error, 1 if bump_attempts else 0, rid))
        conn.execute("INSERT INTO events VALUES(?,?,?,?,?,?)", (now(), rid, cur, new.value, actor, detail))
        conn.execute("UPDATE images SET processing_status=? WHERE record_id=? AND superseded=0", (new.value, rid))

    def records(self, states: Iterable[str] | None = None) -> list[dict]:
        if states:
            states = [RecordState(s).value for s in states]
            q = f"SELECT id,state,midwife_id,patient_id,created_at,updated_at,attempts,last_error FROM records " \
                f"WHERE state IN ({','.join('?' * len(states))}) ORDER BY created_at"
            rows = self.db.execute(q, states)
        else:
            rows = self.db.execute("SELECT id,state,midwife_id,patient_id,created_at,updated_at,attempts,"
                                   "last_error FROM records ORDER BY created_at")
        return [dict(r) for r in rows]

    def events(self, rid: str) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM events WHERE record_id=? ORDER BY ts", (rid,))]

    # ------------------------------------------------------------------ patients
    def create_patient(self, payload: dict, db=None) -> str:
        conn = db or self.db
        while True:
            pid = random_id("PAT")
            if not conn.execute("SELECT 1 FROM patients WHERE id=?", (pid,)).fetchone():
                break
        code = payload.get("code") or ""
        conn.execute("INSERT INTO patients VALUES(?,?,?,?,0,?)",
                     (pid, self.vault.index(code) if code else None, now(), now(), self.vault.enc(payload)))
        return pid

    def get_patient(self, pid: str) -> Optional[dict]:
        row = self.db.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
        if not row:
            return None
        p = dict(row)
        p["payload"] = self.vault.dec(p.pop("enc"))
        return p

    def update_patient(self, pid: str, payload: dict, db=None):
        code = payload.get("code") or ""
        (db or self.db).execute("UPDATE patients SET enc=?, code_idx=?, updated_at=?, synced=0 WHERE id=?",
                                (self.vault.enc(payload), self.vault.index(code) if code else None, now(), pid))

    def patients(self) -> list[dict]:
        out = []
        for row in self.db.execute("SELECT * FROM patients ORDER BY created_at"):
            p = dict(row)
            p["payload"] = self.vault.dec(p.pop("enc"))
            out.append(p)
        return out

    def patients_by_code(self, code: str) -> list[str]:
        return [r["id"] for r in self.db.execute("SELECT id FROM patients WHERE code_idx=?",
                                                  (self.vault.index(code),))]

    def meta_get(self, k, default=None):
        r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return json.loads(r["v"]) if r else default

    def meta_set(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, json.dumps(v)))
