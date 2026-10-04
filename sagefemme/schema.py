"""Registry field schema, field-status model and record lifecycle.

The schema is derived from the sections of the paper maternal registry
("Cuadernito Rosa"-style booklet), NOT from generic OCR. Every field has:
  - a stable id (used everywhere: extraction, storage, evaluation)
  - the page it normally lives on
  - bilingual labels + French aliases as printed on paper (used by the local OCR parser)
  - a type and a validator (used to downgrade confidence / flag NEEDS_REVIEW)
  - an importance weight (used to order follow-up questions)

Direct identifiers (name, husband's name, national ID, phone, address) are listed in
PII_LABELS only so that the pipeline can recognise and DROP them. They have no field id
and can never be stored.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable, Optional


# --------------------------------------------------------------------------- statuses
class FieldStatus(str, Enum):
    KNOWN = "KNOWN"                    # value read (or entered) and trusted
    UNKNOWN = "UNKNOWN"                # paper explicitly says unknown ("inconnu", "?", "NSP")
    NOT_PROVIDED = "NOT_PROVIDED"      # left blank on paper / page not captured
    ILLEGIBLE = "ILLEGIBLE"            # something is written but cannot be read
    NOT_APPLICABLE = "NOT_APPLICABLE"  # logically N/A (e.g. C-section indication after vaginal birth) or "/" on paper
    NEEDS_REVIEW = "NEEDS_REVIEW"      # value read but confidence too low / failed validation


STATUS_LABELS = {
    "fr": {
        "KNOWN": "connu", "UNKNOWN": "inconnu", "NOT_PROVIDED": "non fourni",
        "ILLEGIBLE": "illisible", "NOT_APPLICABLE": "non applicable", "NEEDS_REVIEW": "à réviser",
    },
    "en": {
        "KNOWN": "known", "UNKNOWN": "unknown", "NOT_PROVIDED": "not provided",
        "ILLEGIBLE": "illegible", "NOT_APPLICABLE": "not applicable", "NEEDS_REVIEW": "needs review",
    },
}

# Where a field value came from (kept for audit and for evaluation)
SOURCES = ("ai", "ai_ensemble", "midwife_confirmed", "midwife_edit", "manual", "rule")


# --------------------------------------------------------------------------- lifecycle
class RecordState(str, Enum):
    CAPTURED = "CAPTURED"
    PENDING_AI = "PENDING_AI"
    AI_PROCESSED = "AI_PROCESSED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    VALIDATED = "VALIDATED"
    PATIENT_MATCHED = "PATIENT_MATCHED"
    REGISTERED = "REGISTERED"
    SYNCED = "SYNCED"
    # failure / exception states
    PROCESSING_FAILED = "PROCESSING_FAILED"
    SYNC_FAILED = "SYNC_FAILED"
    DUPLICATE_SUSPECTED = "DUPLICATE_SUSPECTED"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


S = RecordState
TRANSITIONS: dict[RecordState, set[RecordState]] = {
    S.CAPTURED: {S.PENDING_AI, S.MANUAL_REVIEW_REQUIRED},
    S.PENDING_AI: {S.AI_PROCESSED, S.PROCESSING_FAILED, S.MANUAL_REVIEW_REQUIRED, S.PENDING_AI},
    S.AI_PROCESSED: {S.NEEDS_REVIEW},
    S.NEEDS_REVIEW: {S.VALIDATED, S.PENDING_AI, S.MANUAL_REVIEW_REQUIRED},
    S.PROCESSING_FAILED: {S.PENDING_AI, S.MANUAL_REVIEW_REQUIRED},
    S.MANUAL_REVIEW_REQUIRED: {S.VALIDATED, S.PENDING_AI},
    S.VALIDATED: {S.PATIENT_MATCHED, S.DUPLICATE_SUSPECTED, S.NEEDS_REVIEW},
    S.DUPLICATE_SUSPECTED: {S.PATIENT_MATCHED, S.NEEDS_REVIEW},
    S.PATIENT_MATCHED: {S.REGISTERED},
    S.REGISTERED: {S.SYNCED, S.SYNC_FAILED},
    S.SYNC_FAILED: {S.SYNCED, S.SYNC_FAILED, S.REGISTERED},
    # re-digitisation reopens a synced record; central DB may reveal a cross-device duplicate
    S.SYNCED: {S.NEEDS_REVIEW, S.DUPLICATE_SUSPECTED},
}


class IllegalTransition(Exception):
    pass


def check_transition(cur: RecordState | str, new: RecordState | str) -> None:
    cur, new = RecordState(cur), RecordState(new)
    if new not in TRANSITIONS[cur]:
        raise IllegalTransition(f"{cur.value} -> {new.value} not allowed")


# --------------------------------------------------------------------------- field types
from .forms.layout import (DATA_SPECS, PAGE_ORDER, PAGE_TYPES, PII_SPECS, SPEC_BY_ID,  # noqa: E402
                           Spec)

FIELDS = DATA_SPECS
FIELD_BY_ID = {f.id: f for f in FIELDS}
FIELD_IDS = [f.id for f in FIELDS]

YES = {"oui", "o", "yes", "y", "x", "+", "نعم", "true", "vrai", "1"}
NO = {"non", "n", "no", "لا", "false", "faux", "0"}
LAB = {"neg": "négatif", "negatif": "négatif", "negative": "négatif", "-": "négatif", "n": "négatif",
       "pos": "positif", "positif": "positif", "positive": "positif", "+": "positif",
       "immune": "immune", "immun": "immune", "nonimmune": "non immune", "nonimmun": "non immune",
       "nf": "non fait", "nonfait": "non fait", "notdone": "non fait"}
UNKNOWN_TOKENS = {"inconnu", "inconnue", "?", "nsp", "ne sait pas", "unknown", "غير معروف"}
NA_TOKENS = {"/", "—", "-", "–", "na", "n/a", "nap", "sans objet", "s/o", "non applicable"}


def strip_accents(s: str) -> str:
    s = s.replace("œ", "oe").replace("Œ", "OE")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _norm(s: str) -> str:
    return strip_accents(s.strip().lower())


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


levenshtein = _lev


def _parse_date(v: str) -> Optional[str]:
    v = v.strip().replace(" ", "").replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            d = datetime.strptime(v, fmt).date()
            if 1990 <= d.year <= 2035:
                return d.isoformat()
        except ValueError:
            pass
    if re.fullmatch(r"(19|20)\d\d", v):          # year only ("2023")
        return v
    return None


def _num(s: str):
    s = s.replace(",", ".").replace(" ", "")
    s = re.sub(r"(?<=\d)[Oo]|[Oo](?=\d)", "0", s)
    s = re.sub(r"(?<=\d)[lI]|[lI](?=\d)", "1", s)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def normalize(spec: Spec, raw: Any) -> tuple[Any, bool]:
    """Typed, canonical value + whether it passes the field validator."""
    if raw is None:
        return None, False
    if isinstance(raw, bool):
        return raw, True
    s = str(raw).strip()
    if not s:
        return None, False
    t = spec.type
    if t in ("int", "float", "ga"):
        v = _num(s)
        if v is None:
            return s, False
        if t != "float":
            if abs(v - round(v)) > 1e-6 and t == "int":
                return s, False
            v = int(round(v))
        ok = True
        if t == "ga":
            ok = 4 <= v <= 45
        if spec.vmin is not None and v < spec.vmin:
            ok = False
        if spec.vmax is not None and v > spec.vmax:
            ok = False
        return v, ok
    if t == "date":
        d = _parse_date(s)
        return (d, True) if d else (s, False)
    if t == "bool":
        n = re.sub(r"\s", "", _norm(s)).rstrip(".")
        if n in YES:
            return True, True
        if n in NO:
            return False, True
        best = min(("oui", "non"), key=lambda c: _lev(n, c))
        if _lev(n, best) <= 1 and len(n) >= 2:
            return best == "oui", True
        return s, False
    if t == "lab":
        n = re.sub(r"[^a-z+\-]", "", _norm(s))
        if n in LAB:
            return LAB[n], True
        best = min(LAB, key=lambda c: _lev(n, c))
        if len(best) >= 3 and _lev(n, best) <= 1:
            return LAB[best], True
        return s, False
    if t == "bp":
        m = re.search(r"(\d{2,3})\s*[/\\|-]\s*(\d{1,3})", s.replace(" ", ""))
        if not m:
            return s, False
        sy, di = int(m.group(1)), int(m.group(2))
        if sy < 30:                         # cmHg ("12/8")
            sy, di = sy * 10, di * 10
        return f"{sy}/{di}", (60 <= sy <= 250 and 30 <= di <= 160 and sy > di)
    if t == "code":
        v = re.sub(r"\s+", "", s).upper()
        return v, bool(re.fullmatch(r"[A-Z0-9][A-Z0-9\-/]{3,20}", v))
    return " ".join(s.split()), True


def section_label(page_type: str, lang: str = "fr") -> str:
    _, fr, en = PAGE_TYPES[page_type]
    return fr if lang == "fr" else en


def field_label(fid: str, lang: str = "fr") -> str:
    f = SPEC_BY_ID[fid]
    return f.fr if lang == "fr" or not f.en else f.en


def na_rules(values: dict[str, Any]) -> dict[str, str]:
    """Fields that are logically not applicable given other values."""
    out = {}
    if values.get("mode_cs_planned") is False and values.get("mode_cs_emergency") is False:
        out["cs_indication"] = "pas de césarienne"
    if values.get("del_complications") is False:
        out["dc_other_text"] = "pas de complication"
    for pre in ("pme_", "pml_"):
        if values.get(pre + "cesarean") is False:
            out[pre + "scar"] = "pas de césarienne"
    return out
