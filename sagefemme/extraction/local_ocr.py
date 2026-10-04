"""Offline, on-device extraction backend (no network, no third party).

Template-driven, not generic OCR:
  photo -> page type + homography (forms/align.py)
        -> every field region of the template is rectified out of the photo
        -> handwriting is isolated from the printed form by ink colour (blue pen vs black print
           on pink paper) with a darkness+line-removal fallback for black pens
        -> empty regions = NOT_PROVIDED, a lone dash = NOT_APPLICABLE, ticks measured by ink
           density inside the box, remaining regions read by Tesseract (batched per page)
Returns raw readings + per-field confidence; statuses are decided in postprocess.py.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

from ..forms.align import Alignment, align, templates
from ..forms.layout import SPEC_BY_ID

PX_PER_PT = 4.0          # rectified crop resolution (~288 dpi)
NUMERIC = {"int", "float", "date", "bp", "ga"}


@dataclass
class Reading:
    raw: object = None           # str | bool | None
    conf: float = 0.0            # 0..1
    ink: float = 0.0             # fraction of ink pixels in the region
    dash: bool = False
    note: str = ""


@dataclass
class PageReading:
    page_type: Optional[str]
    align_inliers: int
    align_err: float
    fields: dict = field(default_factory=dict)       # fid -> Reading
    pii_polys: list = field(default_factory=list)    # image polygons to redact
    quality: dict = field(default_factory=dict)
    seconds: float = 0.0
    backend: str = "local_ocr"


def _rectify(img: np.ndarray, al: Alignment, bbox, ppt=PX_PER_PT) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    w, h = max(4, int((x1 - x0) * ppt)), max(4, int((y1 - y0) * ppt))
    src = al.project(bbox).astype(np.float32)
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _flatten(rgb: np.ndarray) -> np.ndarray:
    """Remove illumination per channel (divide by blurred background)."""
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[..., c].astype(np.float32)
        bg = cv2.dilate(ch, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        bg = cv2.GaussianBlur(bg, (0, 0), 6)
        out[..., c] = np.clip(255 * ch / (bg + 1), 0, 255)
    return out


def ink_masks(crop: np.ndarray, relative: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(blue_ink_mask, dark_mask) as uint8 0/255. relative=True compares to the crop's own
    paper colour (robust to shadows); checkbox interiors use absolute thresholds."""
    f = _flatten(crop)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    gray = f.mean(axis=2)
    if not relative:
        blue = ((b - r) > 18) & (b < 235) & ((r + g) / 2 < 185)
        return (blue * 255).astype(np.uint8), ((gray < 120) * 255).astype(np.uint8)
    bg = float(np.median(gray))
    blue = ((b - r) > 10) & (gray < bg - 22)
    dark = gray < min(135, bg - 60)
    return (blue * 255).astype(np.uint8), (dark * 255).astype(np.uint8)


def _remove_lines(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * 0.45)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * 0.7))))
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hk) | cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk)
    m = cv2.subtract(mask, cv2.dilate(lines, np.ones((3, 3), np.uint8)))
    return _drop_specks(m)


def _drop_specks(m: np.ndarray, min_area=22) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    return keep


def _drop_border_fragments(m: np.ndarray) -> np.ndarray:
    """Remove thin line fragments touching the crop border (table rules after a slight misalignment)."""
    h, w = m.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = m.copy()
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        touches = x <= 1 or y <= 1 or x + ww >= w - 1 or y + hh >= h - 1
        thin = min(ww, hh) <= 6 and max(ww, hh) >= 3 * min(ww, hh)
        if touches and (thin or a < 60):
            out[lab == i] = 0
    return out


def handwriting_mask(crop: np.ndarray, page_mode: str | None = None) -> tuple[np.ndarray, str]:
    """Blue pen: colour separation alone (printed lines/labels are black, never blue).
    Black pen: darkness, minus long ruled lines, dotted leaders and border fragments.
    page_mode: the pen colour detected for the whole page (a blue-pen page never falls back
    to darkness, so table rules can't be mistaken for writing in empty cells)."""
    blue, dark = ink_masks(crop)
    blue = _drop_specks(blue)
    if page_mode == "blue" or (page_mode is None and (blue > 0).mean() > 0.004):
        return _drop_border_fragments(blue), "blue"
    return _drop_border_fragments(_remove_lines(dark)), "dark"


def _ink_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def _is_dash(mask: np.ndarray) -> bool:
    bb = _ink_bbox(mask)
    if bb is None:
        return False
    x0, y0, x1, y1 = bb
    w, h = x1 - x0 + 1, y1 - y0 + 1
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 14]
    return len(big) == 1 and w / max(h, 1) > 2.5 and h < 0.3 * mask.shape[0] and w < 0.6 * mask.shape[1]


def check_ink(img: np.ndarray, al: Alignment, box) -> tuple[bool, float, float]:
    """Return (ticked, confidence, ink_fraction) for a checkbox."""
    x0, y0, x1, y1 = box
    pad = 0.0
    crop = _rectify(img, al, (x0 - pad, y0 - pad, x1 + pad, y1 + pad), ppt=6)
    h, w = crop.shape[:2]
    i0, j0 = int(h * .24), int(w * .24)
    inner = crop[i0:h - i0, j0:w - j0]
    blue, dark = ink_masks(inner, relative=False)
    frac = max((blue > 0).mean(), (dark > 0).mean())
    thr = 0.07
    ticked = bool(frac > thr)
    # confidence grows with the distance to the decision threshold
    d = (frac - thr) / thr if ticked else (thr - frac) / thr
    conf = float(np.clip(0.5 + 0.5 * d, 0.5, 0.99))
    return ticked, conf, float(frac)


def ocr_image(crop: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """Grayscale image for the recogniser: illumination flattened, printed ruling removed,
    everything far from detected handwriting whitened. Black text on white."""
    f = _flatten(crop)
    if mode == "blue":
        # emphasise blue ink, suppress black print: use the red channel (blue ink is dark in red)
        g = f[..., 0]
    else:
        g = f.mean(axis=2)
    dark = ((f.mean(axis=2) < 135) * 255).astype(np.uint8)
    h, w = dark.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * 0.45)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * 0.7))))
    lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, hk) | cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk)
    near = cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0
    out = np.where(near, g, 255).astype(np.float32)
    out[cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0] = 255
    lo = np.percentile(out[near], 3) if near.any() else 0
    out = np.clip((out - lo) * 255.0 / max(1.0, 235 - lo), 0, 255)
    return out.astype(np.uint8)


def _ocr_batch(crops: list[tuple[str, np.ndarray]], numeric: bool) -> dict:
    """OCR many single-line crops in one Tesseract call: stack them vertically, then map
    recognised lines back to crops by vertical position."""
    if not crops:
        return {}
    H_LINE = 64
    pad = 26
    rows, spans, y = [], [], pad
    width = 0
    for fid, m in crops:
        img = m[1] if isinstance(m, tuple) else 255 - m
        m = m[0] if isinstance(m, tuple) else m
        bb = _ink_bbox(m)
        x0, y0, x1, y1 = bb
        piece = img[max(0, y0 - 5):y1 + 6, max(0, x0 - 5):x1 + 6]
        s = H_LINE / max(piece.shape[0], 1)
        s = min(s, 2.5)
        piece = cv2.resize(piece, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        rows.append(piece)
        spans.append((fid, y, y + piece.shape[0]))
        y += piece.shape[0] + pad
        width = max(width, piece.shape[1])
    canvas = np.full((y + pad, width + 2 * pad), 255, np.uint8)
    for (fid, a, b), piece in zip(spans, rows):
        canvas[a:b, pad:pad + piece.shape[1]] = piece
    cfg = "--psm 6 -c preserve_interword_spaces=1"
    if numeric:
        cfg += " -c tessedit_char_whitelist=0123456789/.,-:SAgkcmdLjoursC°"
    d = pytesseract.image_to_data(canvas, lang="eng", config=cfg, output_type=pytesseract.Output.DICT)
    out = {fid: [] for fid, _, _ in spans}
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t:
            continue
        cy = d["top"][i] + d["height"][i] / 2
        for fid, a, b in spans:
            if a - pad / 2 <= cy <= b + pad / 2:
                out[fid].append((d["left"][i], t, float(d["conf"][i])))
                break
    res = {}
    for fid, ws in out.items():
        ws.sort()
        if ws:
            res[fid] = (" ".join(w for _, w, _ in ws), float(np.mean([c for *_, c in ws])) / 100)
        else:
            res[fid] = ("", 0.0)
    return res


def image_quality(img: np.ndarray) -> dict:
    """On-device quality gate before a capture is accepted."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    s = 1000 / max(g.shape)
    small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
    bright = float(small.mean())
    contrast = float(small.std())
    issues = []
    if sharp < 60:
        issues.append("blur")
    if bright < 60:
        issues.append("dark")
    if bright > 235:
        issues.append("overexposed")
    if contrast < 25:
        issues.append("low_contrast")
    if max(g.shape) < 900:
        issues.append("low_resolution")
    return {"sharpness": round(sharp, 1), "brightness": round(bright, 1), "contrast": round(contrast, 1),
            "issues": issues, "ok": not issues}


def read_page(image: Image.Image | np.ndarray, page_type: Optional[str] = None) -> PageReading:
    t0 = time.time()
    rgb = np.array(image.convert("RGB")) if isinstance(image, Image.Image) else image
    al = align(rgb, page_type)
    pr = PageReading(al.page_type, al.inliers, al.reproj_err, quality=image_quality(rgb))
    if not al.ok:
        pr.seconds = time.time() - t0
        return pr
    tpl = templates()[al.page_type]
    num_batch, txt_batch = [], []
    crops = {}
    for fid, reg in tpl["regions"].items():
        spec = SPEC_BY_ID[fid]
        if spec.kind == "pii":
            pr.pii_polys.append(al.project(reg["bbox"]).tolist())
            continue
        if "box" in reg:
            ticked, conf, frac = check_ink(rgb, al, reg["box"])
            pr.fields[fid] = Reading(ticked, conf, frac)
            continue
        crops[fid] = _rectify(rgb, al, reg["bbox"])
    # pen colour of the page: blue if a good share of the value regions contain blue ink
    blue_regions = sum(1 for c in crops.values() if (_drop_specks(ink_masks(c)[0]) > 0).mean() > 0.004)
    page_mode = "blue" if blue_regions >= max(3, 0.15 * len(crops)) else "dark"
    pr.quality["pen"] = page_mode
    for fid, crop in crops.items():
        spec = SPEC_BY_ID[fid]
        mask, mode = handwriting_mask(crop, page_mode)
        frac = float((mask > 0).mean())
        if frac < 0.003 or _ink_bbox(mask) is None:
            # confidence that the field is really blank
            pr.fields[fid] = Reading(None, float(np.clip(1 - frac / 0.003, 0.6, 0.98)), frac, note="blank")
            continue
        if _is_dash(mask):
            pr.fields[fid] = Reading("—", 0.85, frac, dash=True)
            continue
        pr.fields[fid] = Reading(None, 0.0, frac, note=mode)
        (num_batch if spec.type in NUMERIC else txt_batch).append((fid, (mask, ocr_image(crop, mask, mode))))
    for batch, numeric in ((num_batch, True), (txt_batch, False)):
        for i in range(0, len(batch), 40):
            res = _ocr_batch(batch[i:i + 40], numeric)
            for fid, (txt, conf) in res.items():
                r = pr.fields[fid]
                r.raw = txt or None
                r.conf = conf
                if not txt:
                    r.note = "ink_unreadable"
    pr.seconds = time.time() - t0
    return pr
