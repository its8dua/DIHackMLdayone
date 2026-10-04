"""Patient linking (visit -> longitudinal profile) and re-digitisation merge.

* The linking key is the code the midwife writes on the registry (here: "N° de la fiche").
* Internal patient IDs are random (PAT-XXXX-XXXX), never derived from personal data.
* Candidates = exact code, near-identical code (OCR slips), plus agreement of non-identifying
  attributes (age, province, expected delivery date, LMP). The system NEVER auto-creates a
  patient when a plausible match exists, and never merges by itself: the midwife decides.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from .schema import FIELD_BY_ID, levenshtein
from .store import LocalStore

PROFILE_KEYS = ("fiche_number", "age", "province", "region", "facility", "lmp", "edd", "del_date", "gravidity", "parity")
PLAUSIBLE = 0.35


def profile_from_fields(fields: dict) -> dict:
    out = {}
    for k in PROFILE_KEYS:
        f = fields.get(k)
        if f and f["value"] is not None and f["status"] == "KNOWN":
            out[k] = f["value"]
    return out


def _days(a, b) -> Optional[int]:
    try:
        return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)
    except Exception:
        return None


def score(new: dict, prof: dict) -> tuple[float, list[str]]:
    s, why = 0.0, []
    c1, c2 = (new.get("fiche_number") or "").upper(), (prof.get("fiche_number") or "").upper()
    if c1 and c2:
        d = levenshtein(c1.replace("-", ""), c2.replace("-", ""))
        if d == 0:
            s += 0.7
            why.append("même N° de fiche")
        elif d <= 2:
            s += 0.45 - 0.1 * d
            why.append(f"N° de fiche très proche ({c2})")
    a1, a2 = new.get("age"), prof.get("age")
    if isinstance(a1, int) and isinstance(a2, int):
        if abs(a1 - a2) <= 1:
            s += 0.1
            why.append("âge compatible")
        elif abs(a1 - a2) > 3:
            s -= 0.2
    if new.get("province") and prof.get("province") and new["province"].lower() == prof["province"].lower():
        s += 0.08
        why.append("même province")
    for k, lab, w in (("edd", "même DPA", 0.25), ("lmp", "même DDR", 0.2)):
        d = _days(new.get(k), prof.get(k)) if new.get(k) and prof.get(k) else None
        if d is not None:
            if d <= 10:
                s += w
                why.append(lab)
            elif d > 60:
                s -= 0.1
    return round(max(0.0, min(1.0, s)), 3), why


def find_candidates(store: LocalStore, fields: dict, limit: int = 3) -> list[dict]:
    new = profile_from_fields(fields)
    # the code may still be NEEDS_REVIEW-free here (it is confirmed first in the review flow)
    f = fields.get("fiche_number")
    if f and f["value"]:
        new["fiche_number"] = f["value"]
    out = []
    for p in store.patients():
        prof = p["payload"].get("profile", {})
        sc, why = score(new, prof)
        if sc >= PLAUSIBLE:
            out.append({"patient_id": p["id"], "score": sc, "why": why, "summary": summary(prof),
                        "n_records": len(p["payload"].get("records", []))})
    out.sort(key=lambda c: -c["score"])
    return out[:limit]


def summary(prof: dict) -> str:
    parts = []
    if prof.get("fiche_number"):
        parts.append(f"N° {prof['fiche_number']}")
    if prof.get("age"):
        parts.append(f"{prof['age']} ans")
    if prof.get("province"):
        parts.append(prof["province"])
    if prof.get("edd"):
        parts.append(f"DPA {prof['edd']}")
    return " · ".join(parts) or "profil sans détails"


def diff(existing: dict, new: dict) -> dict:
    """Compare the patient's current merged fields with a re-digitised registry."""
    added, changed, same = [], [], 0
    for fid, nf in new.items():
        if fid not in FIELD_BY_ID:
            continue
        of = existing.get(fid)
        nv = nf.get("value")
        ov = of.get("value") if of else None
        if nf["status"] not in ("KNOWN",) or nv is None:
            continue
        if ov is None:
            added.append(fid)
        elif ov != nv:
            changed.append(fid)
        else:
            same += 1
    return {"added": added, "changed": changed, "same": same}


def merge(existing: dict, new: dict, accept: set[str]) -> dict:
    out = {k: dict(v) for k, v in existing.items()}
    for fid in accept:
        out[fid] = dict(new[fid])
    for fid, nf in new.items():            # fill fields that never existed
        out.setdefault(fid, dict(nf))
    return out


def link_record(store: LocalStore, rid: str, patient_id: Optional[str], actor: str, accept: Optional[set] = None) -> str:
    """Attach a validated record to a patient (existing or new). Atomic."""
    rec = store.get_record(rid)
    payload = rec["payload"]
    fields = payload["fields"]
    with store.tx() as db:
        if patient_id is None:
            prof = profile_from_fields(fields)
            patient_id = store.create_patient({"code": prof.get("fiche_number", ""), "profile": prof,
                                               "fields": fields, "records": [rid], "created_by": actor}, db=db)
            detail = "nouvelle patiente"
        else:
            p = store.get_patient(patient_id)["payload"]
            accept = set(accept or [])
            p["fields"] = merge(p.get("fields", {}), fields, accept)
            p["profile"] = {**p.get("profile", {}), **profile_from_fields({k: p["fields"][k] for k in p["fields"]})}
            p["records"] = p.get("records", []) + [rid]
            p["code"] = p["profile"].get("fiche_number", p.get("code", ""))
            store.update_patient(patient_id, p, db=db)
            detail = f"liée à {patient_id} ({len(accept)} champs mis à jour)"
        db.execute("UPDATE records SET patient_id=? WHERE id=?", (patient_id, rid))
        store.set_state(rid, "PATIENT_MATCHED", actor, detail, db=db)
        store.set_state(rid, "REGISTERED", actor, "", db=db)
    return patient_id
