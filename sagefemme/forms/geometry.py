"""Resolve field specs to regions on a page whose printed layout is known.

Works on a `PageLayout` = printed text runs (text + bbox, PDF points) + checkbox squares.
Used on the specimen PDF (ground truth + template) — the photo pipeline instead projects the
template regions through a homography (see align.py).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from .layout import PAGE_TYPES, Spec

BBox = tuple  # (x0, top, x1, bottom)


def norm(s: str) -> str:
    s = s.replace("œ", "oe").replace("Œ", "OE")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9+\-<:]", "", s)


@dataclass
class Run:
    text: str
    bbox: BBox

    @property
    def n(self):
        return norm(self.text)


@dataclass
class PageLayout:
    runs: list[Run]
    boxes: list[BBox] = field(default_factory=list)
    width: float = 595.3
    height: float = 841.9


def _sorted(cands: list[Run]) -> list[Run]:
    """Reading order robust to jitter: group items whose tops are within 4pt."""
    cands = sorted(cands, key=lambda r: r.bbox[1])
    lines, out = [], []
    for r in cands:
        if lines and abs(lines[-1][0].bbox[1] - r.bbox[1]) < 4:
            lines[-1].append(r)
        else:
            lines.append([r])
    for ln in lines:
        out += sorted(ln, key=lambda r: r.bbox[0])
    return out


def find_anchor(layout: PageLayout, anchor: str, nth: int = 0, exact: bool = False) -> Optional[Run]:
    a = norm(anchor)
    if not a:
        return None
    c = [r for r in layout.runs if (r.n == a if exact else r.n.startswith(a))]
    c = _sorted(c)
    return c[nth] if len(c) > nth else None


def classify_page(layout: PageLayout) -> Optional[str]:
    head = " ".join(r.text for r in layout.runs if r.bbox[1] < 70)
    h = norm(head)
    best, score = None, 0
    for pt, (kws, *_rest) in PAGE_TYPES.items():
        s = sum(1 for k in kws if k in h)
        if s > score:
            best, score = pt, s
    return best if score == len(PAGE_TYPES[best][0]) else None


def _cy(b):
    return (b[1] + b[3]) / 2


def _cx(b):
    return (b[0] + b[2]) / 2


def _clip_right(layout: PageLayout, r: Run, xe: float, xs: float) -> float:
    """Stop a value region before the next printed label on the same line."""
    cy = _cy(r.bbox)
    for o in layout.runs:
        if o is r or abs(_cy(o.bbox) - cy) > 4:
            continue
        if xs + 4 < o.bbox[0] < xe and set(o.text) - set(". "):
            xe = o.bbox[0] - 3
    return xe


def resolve(spec: Spec, layout: PageLayout) -> Optional[dict]:
    """Return {'bbox': region} for value fields, {'box': square} for checkboxes."""
    k = spec.kind
    if k in ("inline", "pii"):
        r = find_anchor(layout, spec.anchor, spec.nth, spec.exact)
        if not r:
            return None
        x0, t, x1, b = r.bbox
        if k == "pii" and spec.w == 0:       # whole printed run is the identifier
            return {"bbox": (x0 - 2, t - 3, x1 + 2, b + 3)}
        xs = x1 + spec.dx0
        xe = _clip_right(layout, r, xs + spec.w, xs)
        return {"bbox": (xs, t - 7, xe, b + 5)}
    if k == "below":
        r = find_anchor(layout, spec.anchor, spec.nth, spec.exact)
        if not r:
            return None
        x0, t, x1, b = r.bbox
        xe = _clip_right(layout, r, x0 + spec.w, x1)
        return {"bbox": (x0 - 6, b + 1, xe, b + 1 + spec.h)}
    if k == "cell":
        row = find_anchor(layout, spec.row, spec.row_nth, spec.exact)
        col = find_anchor(layout, spec.col, spec.col_nth, True)
        if not row or not col:
            return None
        # column headers are left-aligned in their cell; values are written from the left edge
        cx0, cy = col.bbox[0] - 5, _cy(row.bbox)
        return {"bbox": (cx0, cy - spec.h, cx0 + 2 * spec.col_hw, cy + spec.h)}
    if k == "check":
        r = find_anchor(layout, spec.anchor, spec.nth, spec.exact)
        if not r:
            return None
        cy = _cy(r.bbox)
        best, bd = None, 1e9
        for bx in layout.boxes:
            if abs(_cy(bx) - cy) > 7:
                continue
            if spec.side == "left":
                d = r.bbox[0] - bx[2]
            else:
                d = bx[0] - r.bbox[2]
            if -3 <= d < 130 and d < bd:
                best, bd = bx, d
        return {"box": best} if best else None
    if k == "checkrow":
        r = find_anchor(layout, spec.anchor, spec.nth, spec.exact)
        if not r:
            return None
        cy = _cy(r.bbox)
        row = sorted([bx for bx in layout.boxes if abs(_cy(bx) - cy) < 7 and bx[0] > r.bbox[2] - 2],
                     key=lambda b: b[0])
        return {"box": row[spec.idx]} if len(row) > spec.idx else None
    return None
