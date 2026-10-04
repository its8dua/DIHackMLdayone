"""Build the evaluation set + page templates from the organisers' specimen PDF.

  python -m tools.build_dataset --pdf specimen.pdf --out data/testset

Produces
  data/testset/<level>/P<nn>_<page_type>.jpg   level in clean | mild | heavy (field-like phone photos)
  data/testset/ground_truth.xlsx               sheets: values, raw, status (one row per patient)
  sagefemme/forms/templates.json               reference geometry for every page type

Ground truth comes from the PDF text layer (handwriting fonts) and the blue tick strokes, so
it is exact. The source PDF / images are never modified.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sagefemme.forms.layout import DATA_SPECS, SPEC_BY_ID  # noqa: E402
from sagefemme.forms.pdfdoc import ground_truth, load_pdf, template_for  # noqa: E402
from sagefemme.schema import normalize  # noqa: E402


def render(pdf_path: Path, dpi=200) -> list[Image.Image]:
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf_path))
    return [doc[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(len(doc))]


def degrade(img: Image.Image, rng: random.Random, level: str) -> Image.Image:
    """Simulate a phone photo of a paper page: background margin, perspective, rotation,
    uneven lighting / shadow, low light, defocus blur, sensor noise, JPEG compression."""
    import cv2
    a = np.array(img.convert("RGB")).astype(np.float32)
    h, w = a.shape[:2]
    k = {"clean": 0, "mild": 1, "heavy": 2}[level]
    if k == 0:
        return img
    pad = int(0.04 * w * k)
    bg = np.array([rng.uniform(40, 90)] * 3, np.float32) * np.array([1.0, 0.95, 0.9])
    canvas = np.ones((h + 2 * pad, w + 2 * pad, 3), np.float32) * bg
    canvas[pad:pad + h, pad:pad + w] = a
    H, W = canvas.shape[:2]
    j = 0.025 * W * k
    src = np.float32([[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]])
    dst = src + np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)])
    M = cv2.getPerspectiveTransform(src, dst)
    canvas = cv2.warpPerspective(canvas, M, (W, H), borderValue=tuple(float(x) for x in bg))
    R = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-2.0, 2.0) * k, 1.0)
    canvas = cv2.warpAffine(canvas, R, (W, H), borderValue=tuple(float(x) for x in bg))
    # lighting: linear gradient + soft shadow blob
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rng.uniform(0, 2 * math.pi)
    g = (np.cos(ang) * xx / W + np.sin(ang) * yy / H)
    g = 1 - (0.18 * k) * (g - g.min()) / (np.ptp(g) + 1e-6)
    cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(.25, .5) * W
    shadow = 1 - (0.22 * k) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    canvas *= (g * shadow)[..., None]
    canvas = 255 * (canvas / 255) ** (1 + 0.2 * k)               # low light
    blur = rng.uniform(0.6, 1.0) if k == 1 else rng.uniform(1.1, 1.6)
    canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
    canvas += np.random.default_rng(rng.randint(0, 2 ** 31)).normal(0, 3 * k, canvas.shape)
    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=85 if k == 1 else rng.randint(45, 65))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(ROOT / "specimen.pdf"))
    ap.add_argument("--out", default=str(ROOT / "data/testset"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--template-patient", type=int, default=1)
    ap.add_argument("--gt-only", action="store_true", help="only rewrite ground_truth.xlsx")
    a = ap.parse_args(argv)
    out = Path(a.out)
    pages = load_pdf(a.pdf)
    imgs = [None] * len(pages) if a.gt_only else render(Path(a.pdf))
    rng = random.Random(a.seed)

    # templates (reference geometry)
    templates = {}
    for p in pages:
        if p.patient_no == a.template_patient and p.page_type:
            templates[p.page_type] = template_for(p)
    tpath = ROOT / "sagefemme/forms/templates.json"
    tpath.write_text(json.dumps(templates, ensure_ascii=False))
    from sagefemme.forms.align import TPL_DPI
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(a.pdf))
    for i, p in enumerate(pages):
        if p.patient_no == a.template_patient and p.page_type:
            doc[i].render(scale=TPL_DPI / 72).to_pil().convert("L").save(tpath.with_name(f"tpl_{p.page_type}.png"))
    print(f"templates: {sorted(templates)} -> {tpath}")

    rows_raw, rows_val, rows_st = {}, {}, {}
    for p, img in zip(pages, imgs):
        pid = f"P{p.patient_no:02d}"
        gt = ground_truth(p)
        for fid, g in gt.items():
            rows_raw.setdefault(pid, {"patient": pid})[fid] = g["raw"]
            rows_st.setdefault(pid, {"patient": pid})[fid] = g["status"]
            v = None
            if g["status"] == "KNOWN":
                v, _ = normalize(SPEC_BY_ID[fid], g["raw"])
                if isinstance(v, str) and "\ufffd" in str(g["raw"]):
                    v = g["raw"]          # keep the wildcard (glyph not rendered) as-is
            rows_val.setdefault(pid, {"patient": pid})[fid] = v
        for level in (() if a.gt_only else ("clean", "mild", "heavy")):
            d = out / level
            d.mkdir(parents=True, exist_ok=True)
            degrade(img, rng, level).save(d / f"{pid}_{p.page_type}.jpg", quality=92)
    import pandas as pd
    cols = ["patient"] + [s.id for s in DATA_SPECS]
    with pd.ExcelWriter(out / "ground_truth.xlsx") as xw:
        for name, rows in (("values", rows_val), ("raw", rows_raw), ("status", rows_st)):
            df = pd.DataFrame(list(rows.values())).reindex(columns=cols)
            df.to_excel(xw, sheet_name=name, index=False)
    print(f"{len(pages)} pages x 3 levels -> {out}")


if __name__ == "__main__":
    main()
