"""Evaluate extraction + uncertainty handling on the test set.

  python -m tools.evaluate --levels clean,mild,heavy [--vlm] [--patients P01,P02] [--jobs 4]

Metrics (per degradation level)
  value_accuracy      fields whose ground truth is a written value: exact match after normalisation
  status_accuracy     predicted status family vs ground truth (blank / dash / value), NEEDS_REVIEW
                      and ILLEGIBLE count as a correct "value is there" detection
  checkbox_accuracy   ticked vs not ticked
  auto_accept_*       among fields the agent marks KNOWN (no human check): coverage and precision
  silent_error_rate   KNOWN but wrong — the dangerous case the agent must minimise
  review_catch_rate   among wrong readings, share that were flagged for review (not KNOWN)
  ECE + reliability   calibration of confidence scores
  pii_leaks           direct identifiers found anywhere in the extracted record (must be 0)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from sagefemme.extraction.lexicon import _k  # noqa: E402
from sagefemme.forms.layout import PAGE_ORDER, SPEC_BY_ID  # noqa: E402
from sagefemme.schema import normalize  # noqa: E402


def match(spec, pred, gt) -> bool:
    if gt is None or (isinstance(gt, float) and math.isnan(gt)):
        return pred is None
    if spec.kind in ("check", "checkrow") or spec.type == "bool":
        if pred is None or isinstance(pred, str):
            return False
        return bool(pred) == bool(float(gt))
    if pred is None:
        return False
    if isinstance(gt, str) and "�" in gt:                 # glyph not rendered on paper
        import re
        pat = "".join("." if c == "�" else re.escape(c) for c in _k(gt).replace("�", "?"))
        pat = pat.replace(r"\?", ".?")
        return re.fullmatch(pat, _k(str(pred))) is not None
    if spec.type in ("int", "float", "ga"):
        try:
            return abs(float(pred) - float(gt)) < 1e-6
        except (TypeError, ValueError):
            return False
    if spec.type in ("text", "code"):
        return _k(str(pred)) == _k(str(gt))
    return str(pred).lower() == str(gt).lower()


def _run_patient(args):
    level, pid, use_vlm = args
    from sagefemme.extraction.pipeline import extract_record
    from sagefemme.net import Network
    imgs = []
    for pt in PAGE_ORDER:
        f = ROOT / "data/testset" / level / f"{pid}_{pt}.jpg"
        if f.exists():
            imgs.append(f.read_bytes())
    t = time.time()
    rec = extract_record(imgs, Network(True), use_vlm=use_vlm)
    return level, pid, rec, time.time() - t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="clean,mild,heavy")
    ap.add_argument("--patients", default="")
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--reuse", action="store_true", help="re-score cached extraction results")
    a = ap.parse_args(argv)
    gt_v = pd.read_excel(ROOT / "data/testset/ground_truth.xlsx", "values").set_index("patient")
    gt_s = pd.read_excel(ROOT / "data/testset/ground_truth.xlsx", "status").set_index("patient")
    pids = a.patients.split(",") if a.patients else list(gt_v.index)
    jobs = [(lvl, pid, a.vlm) for lvl in a.levels.split(",") for pid in pids]
    cache = Path(a.out) / f"eval_records_{'vlm' if a.vlm else 'local'}.pkl"
    if a.reuse and cache.exists():
        import pickle
        results = [r for r in pickle.loads(cache.read_bytes()) if r[0] in a.levels.split(",") and r[1] in pids]
    else:
        with ProcessPoolExecutor(a.jobs) as ex:
            results = list(ex.map(_run_patient, jobs))
        import pickle
        Path(a.out).mkdir(exist_ok=True)
        cache.write_bytes(pickle.dumps(results))

    report = {}
    errors = []
    for level in a.levels.split(","):
        C = Counter()
        rel = defaultdict(lambda: [0, 0])
        by_type = defaultdict(lambda: [0, 0])
        secs = []
        for lv, pid, rec, sec in results:
            if lv != level:
                continue
            secs.append(sec)
            for fid, f in rec["fields"].items():
                spec = SPEC_BY_ID[fid]
                gs = gt_s.loc[pid, fid]
                gv = gt_v.loc[pid, fid]
                gv = None if (isinstance(gv, float) and math.isnan(gv)) else gv
                pst, pv, conf = f["status"], f["value"], f["confidence"]
                if spec.kind in ("check", "checkrow"):
                    ok = match(spec, pv, gv)
                    C["check_n"] += 1
                    C["check_ok"] += ok
                else:
                    fam_gt = {"KNOWN": "value", "NOT_PROVIDED": "blank", "NOT_APPLICABLE": "dash"}.get(gs, "value")
                    fam_p = {"KNOWN": "value", "NEEDS_REVIEW": "value", "ILLEGIBLE": "value", "UNKNOWN": "value",
                             "NOT_PROVIDED": "blank", "NOT_APPLICABLE": "dash"}[pst]
                    if pst == "NOT_APPLICABLE" and fam_gt == "blank":
                        fam_p = "blank"   # logical N/A on a blank line is correct
                    C["status_n"] += 1
                    C["status_ok"] += fam_gt == fam_p
                    if fam_gt == "value":
                        ok = match(spec, pv, gv)
                        C["value_n"] += 1
                        C["value_ok"] += ok
                        by_type[spec.type][0] += 1
                        by_type[spec.type][1] += ok
                    else:
                        ok = fam_p == fam_gt
                if pst == "KNOWN":
                    C["auto_n"] += 1
                    C["auto_ok"] += ok
                if not ok:
                    C["wrong"] += 1
                    C["wrong_flagged"] += pst != "KNOWN"
                    if len(errors) < 4000:
                        errors.append({"level": level, "patient": pid, "field": fid, "gt": str(gv), "gt_status": gs,
                                       "pred": str(pv), "status": pst, "conf": conf, "raw": str(f.get("raw")),
                                       "note": f.get("note", "")})
                C["n"] += 1
                b = min(9, int(conf * 10))
                rel[b][0] += 1
                rel[b][1] += ok
        n = max(C["n"], 1)
        ece = sum(abs(r[1] / r[0] - (b + .5) / 10) * r[0] for b, r in rel.items() if r[0]) / n
        report[level] = {
            "fields_evaluated": C["n"],
            "value_accuracy": round(C["value_ok"] / max(C["value_n"], 1), 4),
            "value_accuracy_by_type": {t: round(v[1] / v[0], 3) for t, v in sorted(by_type.items())},
            "status_accuracy": round(C["status_ok"] / max(C["status_n"], 1), 4),
            "checkbox_accuracy": round(C["check_ok"] / max(C["check_n"], 1), 4),
            "overall_field_accuracy": round(1 - C["wrong"] / n, 4),
            "auto_accept_coverage": round(C["auto_n"] / n, 4),
            "auto_accept_precision": round(C["auto_ok"] / max(C["auto_n"], 1), 4),
            "silent_error_rate": round((C["auto_n"] - C["auto_ok"]) / n, 4),
            "review_catch_rate": round(C["wrong_flagged"] / max(C["wrong"], 1), 4),
            "ece": round(ece, 4),
            "reliability": {f"{b / 10:.1f}-{(b + 1) / 10:.1f}": {"n": r[0], "acc": round(r[1] / r[0], 3)}
                            for b, r in sorted(rel.items())},
            "seconds_per_registry": round(sum(secs) / max(len(secs), 1), 1),
        }
    # PII leak check: identifiers from the specimen text layer must never appear in a record
    from sagefemme.forms.pdfdoc import load_pdf
    from sagefemme.forms.geometry import resolve
    from sagefemme.forms.layout import PII_SPECS
    pii_strings = set()
    for p in load_pdf(ROOT / "specimen.pdf"):
        for s in PII_SPECS:
            if s.page == p.page_type and s.id != "pii_address":   # address tokens = village/province (legit data)
                r = resolve(s, p.layout)
                if r:
                    from sagefemme.forms.pdfdoc import _hand_runs, _inside
                    txt = "".join(h["text"] for h in _hand_runs(p) if _inside(h["first"], r["bbox"]))
                    txt = txt or " ".join(rr.text for rr in p.layout.runs if rr.text.startswith("MÈRE"))
                    for tok in txt.replace("MÈRE —", "").split():
                        if len(tok) >= 4:
                            pii_strings.add(_k(tok))
    # tokens that legitimately appear in non-identifying fields on paper (e.g. facility "DR Bni Ahmed",
    # staff "Sage-femme Salma") are not patient identifiers -> excluded from the leak test
    gt_raw = pd.read_excel(ROOT / "data/testset/ground_truth.xlsx", "raw")
    import re as _re0
    legit = {_k(w) for col in gt_raw.columns if col != "patient" for v in gt_raw[col].dropna().astype(str)
             for w in _re0.split(r"[\s,;/]+", v)}
    pii_strings -= legit
    leaks, leaked = 0, []
    import re as _re
    for _, pid, rec, _ in results:
        for fid, f in rec["fields"].items():
            for val in (f.get("value"), f.get("raw")):
                if not isinstance(val, str):
                    continue
                words = {_k(w) for w in _re.split(r"[\s,;/]+", val) if len(_k(w)) >= 5}
                hit = words & pii_strings
                if hit:
                    leaks += 1
                    leaked.append((pid, fid, sorted(hit)))
    report["pii_leak_details"] = leaked[:20]
    report["pii_leaks"] = leaks
    report["backend"] = "local_ocr + vlm" if a.vlm else "local_ocr"
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    pd.DataFrame(errors).to_csv(out / "eval_errors.csv", index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
