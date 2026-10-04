"""Simulated connectivity. Every online-layer call (AI, sync) goes through Network.require().

The connection can be cut at any moment (UI toggle, tests, or a random drop rate) — including
in the middle of a batch — to prove that no record is lost.
"""
from __future__ import annotations

import random
import threading


class OfflineError(Exception):
    pass


class Network:
    def __init__(self, online: bool = True, drop_rate: float = 0.0, seed: int | None = None):
        self._online = online
        self.drop_rate = drop_rate
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.listeners = []

    @property
    def online(self) -> bool:
        return self._online

    def set_online(self, v: bool):
        with self._lock:
            changed = v != self._online
            self._online = v
        if changed:
            for cb in list(self.listeners):
                cb(v)

    def require(self, what: str = "network"):
        if not self._online:
            raise OfflineError(f"offline: {what}")
        if self.drop_rate and self._rng.random() < self.drop_rate:
            raise OfflineError(f"connection dropped during {what}")
