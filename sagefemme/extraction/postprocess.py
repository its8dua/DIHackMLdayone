"""Turn raw readings into the record object: value + explicit status + calibrated confidence.

Status model (every field, never a bare "N/A"):
  KNOWN           value read with enough confidence (or confirmed by the midwife)
  NEEDS_REVIEW    something was read but confidence is too low / a validator or consistency rule failed
  ILLEGIBLE       ink is present but could not be read
  NOT_PROVIDED    blank on paper, or the page was not photographed
  NOT_APPLICABLE  a dash / "/" on paper, or logically N/A given other answers
  UNKNOWN         the paper explicitly says unknown ("inconnu", "?")
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional

from ..forms.layout import PAGE_TYPES, SPEC_BY_ID
from ..schema import (FIELDS, NA_TOKENS, UNKNOWN_TOKENS, FieldStatus, na_rules, normalize)
from .lexicon import snap

KNOWN_THRESHOLD = {1: 0.80, 2: 0.85, 3: 0.92}

# direct identifiers that must never be stored, even if they leak into a free-text field
PII_PATTERNS = [
    re.compile(r"\b0[5-7](?:[\s.-]?\d{2}){4}\b"),            # Moroccan phone numbers
    re.compile(r"\+212[\s\d.-]{8,}"),
    re.compile(r"\b[A-Z]{1,2}\d{5,7}\b"),                    # CIN
]


def scrub_pii(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    for p in PII_PATTERNS:
        text = p.sub("[masqué]", text)
    return text


def fv(value=None, status: FieldStatus | str = FieldStatus.NOT_PROVIDED, confidence: float = 1.0,
       source: str = "ai", raw: Any = None, note: str = "", page: Optional[str] = None) -> dict:
    if hasattr(value, "item"):
        value = value.item()
    if hasattr(raw, "item"):
        raw = raw.item()
    return {"value": value, "status": FieldStatus(status).value, "confidence": round(float(confidence), 3),
            "source": source, "raw": scrub_pii(raw) if isinstance(raw, str) else raw, "note": note, "page": page}


def from_reading(fid: str, r, page: str, source="ai") -> dict:
    """r: object with raw, conf, dash, note (local Reading or VLM reading)."""
    spec = SPEC_BY_ID[fid]
    raw, conf = r.raw, float(r.conf)
    if spec.kind in ("check", "checkrow"):
        st = FieldStatus.KNOWN if conf >= 0.7 else FieldStatus.NEEDS_REVIEW
        return fv(bool(raw), st, conf, source, raw, "", page)
    if getattr(r, "status_hint", None) in ("ILLEGIBLE", "UNKNOWN", "NOT_APPLICABLE", "NOT_PROVIDED"):
        return fv(None, r.status_hint, conf, source, raw, r.note, page)
    if raw is None:
        if r.note == "blank":
            st = FieldStatus.NOT_PROVIDED if conf >= 0.7 else FieldStatus.NEEDS_REVIEW
            return fv(None, st, conf, source, None, "vide sur le papier", page)
        return fv(None, FieldStatus.ILLEGIBLE, 0.3, source, None, "encre présente mais illisible", page)
    if r.dash:
        return fv(None, FieldStatus.NOT_APPLICABLE, conf, source, "—", "tiret sur le papier", page)
    s = str(raw).strip()
    low = s.lower().strip(" .")
    if low in UNKNOWN_TOKENS:
        return fv(None, FieldStatus.UNKNOWN, max(conf, 0.7), source, s, "inconnu sur le papier", page)
    if low in NA_TOKENS:
        return fv(None, FieldStatus.NOT_APPLICABLE, max(conf, 0.7), source, s, "", page)
    value, valid = normalize(spec, s)
    note = ""
    if spec.type == "text":
        snapped, sim = snap(fid, value)
        if sim > 0:
            if sim < 1:
                note = f"lu « {value} », rapproché de « {snapped} »"
            value = snapped
            conf = max(conf, 0.55 + 0.4 * sim) if sim >= 0.8 else conf * (0.6 + 0.4 * sim)
        else:
            conf = min(conf, 0.75)          # free text never auto-trusted without a lexicon hit
        value = scrub_pii(value)
    if not valid:
        conf = min(conf, 0.35)
        note = note or "valeur hors format / hors plage"
    thr = KNOWN_THRESHOLD.get(spec.importance, 0.8)
    st = FieldStatus.KNOWN if conf >= thr and valid else FieldStatus.NEEDS_REVIEW
    return fv(value, st, conf, source, s, note, page)


def missing_page(fid: str, page: str) -> dict:
    return fv(None, FieldStatus.NOT_PROVIDED, 1.0, "rule", None, "page non photographiée", page)


# ---------------------------------------------------------------- consistency rules
def _d(v) -> Optional[date]:
    try:
        return date.fromisoformat(v) if isinstance(v, str) and len(v) == 10 else None
    except ValueError:
        return None


def _flag(fields, fid, why):
    f = fields.get(fid)
    if f and f["status"] == "KNOWN" and f["source"] in ("ai", "ai_ensemble"):
        f["status"] = "NEEDS_REVIEW"
        f["confidence"] = min(f["confidence"], 0.5)
        f["note"] = (f["note"] + "; " if f["note"] else "") + why


def _boost(fields, fid, why):
    f = fields.get(fid)
    if f and f["status"] == "NEEDS_REVIEW" and f["value"] is not None and f["source"] in ("ai", "ai_ensemble") \
            and "hors" not in f["note"]:
        f["confidence"] = max(f["confidence"], 0.93)
        f["status"] = "KNOWN"
        f["note"] = (f["note"] + "; " if f["note"] else "") + why


def consistency(fields: dict) -> list[str]:
    """Cross-field checks. Agreement raises confidence, contradiction forces review."""
    msgs = []
    v = {k: f["value"] for k, f in fields.items()}
    lmp, edd, term = _d(v.get("lmp")), _d(v.get("edd")), _d(v.get("term_date"))
    if lmp and edd:
        if abs((edd - lmp).days - 280) <= 3:
            _boost(fields, "lmp", "cohérent avec la DPA")
            _boost(fields, "edd", "cohérent avec la DDR")
        else:
            _flag(fields, "lmp", "DDR et DPA incohérentes (≠ 280 j)")
            _flag(fields, "edd", "DDR et DPA incohérentes (≠ 280 j)")
            msgs.append("DDR/DPA incohérentes")
    if edd and term:
        if abs((term - edd).days - 7) <= 1:
            _boost(fields, "term_date", "DPA + 7 j")
        else:
            _flag(fields, "term_date", "date de dépassement ≠ DPA + 7 j")
    if lmp:
        for cid in ("t1v1", "t1v2", "t1v3", "t2v1", "t2v2", "t2v3", "m7", "m8", "m9"):
            vd, ga = _d(v.get(f"g_visit_date_{cid}")), v.get(f"g_ga_{cid}")
            if vd and isinstance(ga, int):
                weeks = (vd - lmp).days / 7
                if abs(weeks - ga) <= 1.5:
                    _boost(fields, f"g_visit_date_{cid}", "cohérent avec DDR + âge gestationnel")
                    _boost(fields, f"g_ga_{cid}", "cohérent avec la date de visite")
                else:
                    _flag(fields, f"g_ga_{cid}", f"âge gestationnel incohérent avec la date ({weeks:.0f} SA attendues)")
                    _flag(fields, f"g_visit_date_{cid}", "date incohérente avec l'âge gestationnel")
    g, p, lc = v.get("gravidity"), v.get("parity"), v.get("living_children")
    if isinstance(g, int) and isinstance(p, int) and p > g:
        _flag(fields, "parity", "parité > gestité")
        _flag(fields, "gravidity", "parité > gestité")
        msgs.append("parité > gestité")
    if isinstance(p, int) and isinstance(lc, int) and lc > p + 1:
        _flag(fields, "living_children", "enfants vivants > parité")
    # mutually exclusive tick groups
    for group in (("nb_alive", "nb_stillborn", "nb_death24"),
                  ("bg_a", "bg_b", "bg_o", "bg_ab"), ("rh_neg", "rh_pos"),
                  ("nbe_bf_exclusive", "nbe_bf_artificial", "nbe_bf_mixed"),
                  ("nbl_bf_exclusive", "nbl_bf_artificial", "nbl_bf_mixed")):
        ticked = [x for x in group if v.get(x) is True]
        if len(ticked) > 1:
            for x in ticked:
                _flag(fields, x, "plusieurs cases cochées dans un groupe exclusif")
    # logical not-applicable
    for fid, why in na_rules(v).items():
        f = fields.get(fid)
        if f and f["status"] in ("NOT_PROVIDED",):
            f["status"] = "NOT_APPLICABLE"
            f["note"] = why
    return msgs


def page_of(fid: str) -> str:
    return SPEC_BY_ID[fid].page


def empty_record_fields(pages_present: set[str]) -> dict:
    return {f.id: missing_page(f.id, f.page) for f in FIELDS if f.page not in pages_present}
