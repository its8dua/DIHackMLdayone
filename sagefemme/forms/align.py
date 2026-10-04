"""Photo -> page type + homography onto the page template.

1. OCR the printed labels (Tesseract handles the Helvetica print well even on degraded photos).
2. Classify the page from its title words.
3. Match OCR words to template words (unique words only) and fit a RANSAC homography
   template(pt) -> image(px). This absorbs perspective, rotation, scale and the page offset.
Every template region (field box, checkbox) can then be projected onto the photo.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .layout import PAGE_TYPES

TEMPLATES_PATH = Path(__file__).with_name("templates.json")


@lru_cache(maxsize=1)
def templates() -> dict:
    return json.loads(TEMPLATES_PATH.read_text())


def wnorm(s: str) -> str:
    s = s.replace("œ", "oe").replace("Œ", "OE")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]", "", s)


def _template_words(tpl: dict) -> list[tuple[str, float, float]]:
    """Split template runs into words with approximate centre positions."""
    out = []
    for r in tpl["runs"]:
        text = r["text"]
        x0, t, x1, b = r["bbox"]
        n = max(1, len(text))
        cw = (x1 - x0) / n
        pos = 0
        for w in text.split(" "):
            if w:
                cx = x0 + cw * (pos + len(w) / 2)
                out.append((wnorm(w), cx, (t + b) / 2))
            pos += len(w) + 1
    return out


@lru_cache(maxsize=16)
def _unique_template_words(page_type: str) -> dict:
    words = _template_words(templates()[page_type])
    cnt = {}
    for w, *_ in words:
        cnt[w] = cnt.get(w, 0) + 1
    return {w: (x, y) for w, x, y in words if len(w) >= 3 and cnt[w] == 1 and not w.isdigit()}


@dataclass
class Alignment:
    page_type: Optional[str]
    H: Optional[np.ndarray]          # 3x3 template(pt) -> image(px)
    n_matches: int
    inliers: int
    reproj_err: float
    words: list                      # OCR words (text, conf, bbox px)

    @property
    def ok(self) -> bool:
        return self.H is not None and self.inliers >= 8

    def project(self, bbox) -> np.ndarray:
        x0, y0, x1, y1 = bbox
        pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)


def ocr_words(gray: np.ndarray) -> list[tuple[str, float, tuple]]:
    """Word-level OCR of the full page (printed labels mainly)."""
    h, w = gray.shape
    scale = 2200 / max(h, w) if max(h, w) < 2200 else 1.0
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else gray
    # flatten illumination: divide by a large-kernel background estimate
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
    flat = cv2.divide(g, bg, scale=255)
    d = pytesseract.image_to_data(flat, lang="eng", config="--psm 11", output_type=pytesseract.Output.DICT)
    out = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t:
            continue
        x, y, ww, hh = (d[k][i] / scale for k in ("left", "top", "width", "height"))
        out.append((t, float(d["conf"][i]), (x, y, x + ww, y + hh)))
    return out


def classify(words) -> Optional[str]:
    top = " ".join(wnorm(t) for t, c, b in words)
    best, score = None, 0
    for pt, (kws, *_r) in PAGE_TYPES.items():
        s = sum(1 for k in kws if k in top)
        if s > score:
            best, score = pt, s
    if best and score == len(PAGE_TYPES[best][0]):
        return best
    return None


def classify_by_matches(words) -> Optional[str]:
    """Fallback: the template sharing the most unique words wins."""
    ws = {wnorm(t) for t, c, b in words}
    scores = {pt: len(ws & set(_unique_template_words(pt))) for pt in templates()}
    pt = max(scores, key=scores.get)
    return pt if scores[pt] >= 8 else None


TPL_DPI = 110
TPL_SCALE = TPL_DPI / 72.0


@lru_cache(maxsize=16)
def _tpl_features(page_type: str):
    path = TEMPLATES_PATH.with_name(f"tpl_{page_type}.png")
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=4000)
    kp, des = sift.detectAndCompute(g, None)
    return kp, des, g.shape


def _prep(gray: np.ndarray, target_w: int) -> tuple[np.ndarray, float]:
    s = target_w / gray.shape[1]
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    g = cv2.divide(g, bg, scale=255)
    return g, s


def _match(page_type, kp, des):
    tkp, tdes, _ = _tpl_features(page_type)
    if des is None or tdes is None:
        return None, 0, 0, 99.0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(tdes, des, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 12:
        return None, len(good), 0, 99.0
    src = np.float32([tkp[g.queryIdx].pt for g in good])
    dst = np.float32([kp[g.trainIdx].pt for g in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0, maxIters=4000)
    if H is None:
        return None, len(good), 0, 99.0
    inl = mask.ravel().astype(bool)
    proj = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.mean(np.linalg.norm(proj - dst[inl], axis=1)))
    return H, len(good), int(inl.sum()), err


def title_text(gray: np.ndarray, H_pt: np.ndarray) -> str:
    """OCR only the projected title band (fast) to separate early/late post-partum pages."""
    pts = cv2.perspectiveTransform(np.float32([[30, 25], [400, 25], [400, 62], [30, 62]]).reshape(-1, 1, 2),
                                   H_pt).reshape(-1, 2)
    dst = np.float32([[0, 0], [1480, 0], [1480, 148], [0, 148]])
    M = cv2.getPerspectiveTransform(np.float32(pts), dst)
    crop = cv2.warpPerspective(gray, M, (1480, 148))
    return pytesseract.image_to_string(crop, lang="eng", config="--psm 6")


def align(img: Image.Image | np.ndarray, page_type: Optional[str] = None) -> Alignment:
    """SIFT + RANSAC against each page template (robust to blur/perspective/lighting),
    then a quick OCR of the title band to disambiguate structurally identical pages."""
    a = np.array(img.convert("L") if isinstance(img, Image.Image) else img)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    tw = int(round(templates()["cover"]["size"][0] * TPL_SCALE))
    g, s = _prep(a, int(tw * 1.15))
    kp, des = cv2.SIFT_create(nfeatures=6000).detectAndCompute(g, None)
    cands = [page_type] if page_type else list(templates())
    results = {}
    for pt in cands:
        results[pt] = _match(pt, kp, des)
    best = max(results, key=lambda k: results[k][2])
    H, ng, ni, err = results[best]
    if H is None or ni < 15:
        return Alignment(best if page_type else None, None, ng, ni, err, [])
    # template pt -> template px -> resized image px -> original image px
    S_t = np.diag([TPL_SCALE, TPL_SCALE, 1.0])
    S_i = np.diag([1 / s, 1 / s, 1.0])
    H_pt = S_i @ H @ S_t
    words = []
    if page_type is None and best.startswith("pp_"):
        t = wnorm(title_text(a, H_pt))
        kind = best.split("_")[1]
        if "tardif" in t or "tard" in t:
            best = f"pp_{kind}_late"
        elif "precoce" in t or "preco" in t:
            best = f"pp_{kind}_early"
        words = [(t, 100.0, (0, 0, 0, 0))]
    return Alignment(best, H_pt / H_pt[2, 2], ng, ni, err, words)
