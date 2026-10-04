"""Epidemiological layer: export each record to the organisers' analytic schema
(maternal_registry_synthetic.csv columns) and compute anonymised aggregates.

Derived variables are computed only from KNOWN / midwife-validated fields; anything else is
left missing (never imputed). Aggregates apply small-cell suppression (counts < 5 hidden).
"""
from __future__ import annotations

import math
import re
import statistics
from typing import Optional

CSV_COLUMNS = [
    "age (years)", "education level (0=none/primary,1=secondary,2=higher)", "consanguinity", "desired pregnancy",
    "hypertension history", "diabetes mellitus", "gravidity (number)", "parity (number)", "abortions (number)",
    "living children (number)", "previous cesarean", "bmi pregestational (kg/m2)", "mean systolic bp (mmhg)",
    "mean diastolic bp (mmhg)", "hemoglobin (g/dl)", "first fasting glucose (mg/dl)", "proteinuria",
    "hiv test result", "syphilis test result", "hepatitis c test result", "gestational age at enrollment (weeks)",
    "gestational dm", "gestational age at birth (weeks)", "preterm birth", "type of delivery (0=vaginal,1=cesarean)",
    "newborn sex (0=female,1=male)", "child birth weight (g)", "head circumference (cm)", "breastfeeding initiated",
    "referral to higher care"]
VISITS = ("t1v1", "t1v2", "t1v3", "t2v1", "t2v2", "t2v3", "m7", "m8", "m9")
SMALL_CELL = 5


def _v(fields, fid):
    """Only trusted values (KNOWN) that pass the field validator feed the statistics."""
    f = fields.get(fid)
    if f and f["status"] == "KNOWN" and f["value"] is not None:
        from .schema import FIELD_BY_ID, normalize
        v = f["value"]
        if isinstance(v, bool):
            return v
        nv, ok = normalize(FIELD_BY_ID[fid], v)
        return nv if ok else None
    return None


def _first(fields, row):
    for c in VISITS:
        v = _v(fields, f"g_{row}_{c}")
        if v is not None:
            return v
    return None


def _lab(v):
    return None if v in (None, "non fait") else (1 if v == "positif" else 0)


def to_analytic_row(fields: dict) -> dict:
    r: dict[str, Optional[float]] = {c: None for c in CSV_COLUMNS}
    r["age (years)"] = _v(fields, "age")
    edu = (_v(fields, "education") or "").lower()
    if edu:
        r["education level (0=none/primary,1=secondary,2=higher)"] = (
            2 if any(k in edu for k in ("sup", "univ")) else 1 if any(k in edu for k in ("coll", "lyc", "second")) else 0)
    for col, fid in (("consanguinity", "consanguinity"), ("desired pregnancy", "desired_pregnancy")):
        v = _v(fields, fid)
        r[col] = None if v is None else int(bool(v))
    med = (_v(fields, "hx_medical") or "").lower()
    r["hypertension history"] = 1 if ("hta" in med or "hypert" in med) else (0 if med else None)
    r["diabetes mellitus"] = 1 if "diab" in med else (0 if med else None)
    r["gravidity (number)"] = _v(fields, "gravidity")
    r["parity (number)"] = _v(fields, "parity")
    r["abortions (number)"] = _v(fields, "ob_abortion_n")
    if r["abortions (number)"] is None and _v(fields, "gravidity") is not None:
        st = fields.get("ob_abortion_n", {}).get("status")
        r["abortions (number)"] = 0 if st == "NOT_PROVIDED" else None
    r["living children (number)"] = _v(fields, "living_children")
    modes = [(_v(fields, f"pd{k}_mode") or "").lower() for k in range(1, 6)]
    if any(modes):
        r["previous cesarean"] = int(any("c" in m and "sar" in m for m in modes))
    w, h = _first(fields, "weight"), _v(fields, "height_cm")
    if w and h:
        r["bmi pregestational (kg/m2)"] = round(w / (h / 100) ** 2, 2)
    bps = [_v(fields, f"g_bp_{c}") for c in VISITS]
    bps = [tuple(map(int, b.split("/"))) for b in bps if isinstance(b, str) and re.fullmatch(r"\d{2,3}/\d{2,3}", b)]
    if bps:
        r["mean systolic bp (mmhg)"] = round(statistics.mean(b[0] for b in bps), 1)
        r["mean diastolic bp (mmhg)"] = round(statistics.mean(b[1] for b in bps), 1)
    r["hemoglobin (g/dl)"] = _first(fields, "hb")
    g = _first(fields, "glycemia")
    r["first fasting glucose (mg/dl)"] = round(g * 100, 1) if g else None
    alb = [_v(fields, f"g_albuminuria_{c}") for c in VISITS]
    alb = [a for a in alb if a in ("positif", "négatif")]
    r["proteinuria"] = int(any(a == "positif" for a in alb)) if alb else None
    r["hiv test result"] = _lab(_first(fields, "hiv"))
    r["syphilis test result"] = _lab(_first(fields, "syphilis"))
    r["hepatitis c test result"] = None          # not on the paper form (only Ag HBs)
    r["gestational age at enrollment (weeks)"] = _first(fields, "ga")
    gab = _v(fields, "nb_ga")
    r["gestational age at birth (weeks)"] = gab
    r["preterm birth"] = None if gab is None else int(gab < 37)
    cs = [_v(fields, "mode_cs_planned"), _v(fields, "mode_cs_emergency")]
    vag = [_v(fields, "mode_vaginal"), _v(fields, "mode_instrumental")]
    if any(x is True for x in cs + vag):
        r["type of delivery (0=vaginal,1=cesarean)"] = int(any(x is True for x in cs))
    sex = (_v(fields, "nb_sex") or "").upper()
    r["newborn sex (0=female,1=male)"] = 1 if sex.startswith("M") else (0 if sex.startswith("F") else None)
    r["child birth weight (g)"] = _v(fields, "nb_weight")
    r["head circumference (cm)"] = _v(fields, "nb_hc")
    bf = [_v(fields, f"nbe_bf_{k}") for k in ("exclusive", "mixed", "artificial")]
    if any(x is not None for x in bf):
        r["breastfeeding initiated"] = int(bool(bf[0] or bf[1]))
    tr = [_v(fields, "nbe_transfer"), _v(fields, "nbl_transfer"), _v(fields, "pme_fp_referred")]
    if any(x is not None for x in tr):
        r["referral to higher care"] = int(any(x is True for x in tr))
    return r


def _clean(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _hist(vals, edges):
    counts = [0] * (len(edges) - 1)
    for v in vals:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v == edges[-1]):
                counts[i] += 1
                break
    return [{"bin": f"{edges[i]}–{edges[i + 1]}", "n": c if c >= SMALL_CELL or c == 0 else None}
            for i, c in enumerate(counts)]


def aggregates(rows: list[dict], temps: Optional[list[float]] = None) -> dict:
    """rows: analytic rows (historical CSV + digitised records). Counts < 5 are suppressed."""
    def col(c):
        return [x for x in (_clean(r.get(c)) for r in rows) if x is not None]
    sbp, dbp = col("mean systolic bp (mmhg)"), col("mean diastolic bp (mmhg)")
    out = {"n_records": len(rows),
           "sbp_hist": _hist(sbp, [70, 90, 100, 110, 120, 130, 140, 160, 200]),
           "dbp_hist": _hist(dbp, [30, 50, 60, 70, 80, 90, 110, 130]),
           "sbp_mean": round(statistics.mean(sbp), 1) if sbp else None,
           "dbp_mean": round(statistics.mean(dbp), 1) if dbp else None,
           "bp_missing": len(rows) - len(sbp)}
    tests = {}
    for name, c in (("VIH", "hiv test result"), ("Syphilis", "syphilis test result"),
                    ("Hépatite C", "hepatitis c test result")):
        v = col(c)
        pos = int(sum(v))
        tests[name] = {"tested": len(v), "positive": pos if pos >= SMALL_CELL or pos == 0 else None,
                       "positive_suppressed": 0 < pos < SMALL_CELL,
                       "rate": round(pos / len(v), 4) if v and (pos >= SMALL_CELL or pos == 0) else None,
                       "missing": len(rows) - len(v)}
    out["tests"] = tests
    t = [x for x in (temps or []) if x]
    out["temp_hist"] = _hist(t, [35, 36, 36.5, 37, 37.5, 38, 39, 42])
    out["temp_n"] = len(t)
    return out
