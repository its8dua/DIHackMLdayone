"""Read the specimen PDF: printed layout, handwritten values and ticked boxes.

The organisers' specimen PDF is vector: printed labels are Helvetica, handwriting is drawn
with handwriting fonts (Caveat, Gaegu, ...) and ticks are blue vector strokes. This lets us
recover exact ground truth for every field, and build page templates for the photo pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import pdfplumber

from .geometry import PageLayout, Run, classify_page, resolve
from .layout import DATA_SPECS, PII_SPECS, SPECS, Spec


def _is_print(fontname: str) -> bool:
    return "Helvetica" in fontname


@dataclass
class PdfPage:
    layout: PageLayout
    hand: list            # handwritten chars (dict with text, x0, x1, top, bottom) in stream order
    marks: list           # blue tick strokes bboxes
    page_type: str | None
    patient_no: int | None


def _runs(chars) -> list[Run]:
    out, cur = [], None
    for c in chars:
        if not _is_print(c["fontname"]):
            continue
        if cur and abs(c["x0"] - cur["x1"]) < 1.6 and abs(c["top"] - cur["top"]) < 2.5:
            cur["t"] += c["text"]
            cur["x1"] = c["x1"]
            cur["bottom"] = max(cur["bottom"], c["bottom"])
        else:
            cur = {"t": c["text"], "x0": c["x0"], "x1": c["x1"], "top": c["top"], "bottom": c["bottom"]}
            out.append(cur)
    return [Run(r["t"].strip(), (r["x0"], r["top"], r["x1"], r["bottom"])) for r in out if r["t"].strip()]


def _color(o):
    c = o.get("stroking_color")
    return tuple(c) if isinstance(c, (list, tuple)) else ()


def load_pdf(path) -> list[PdfPage]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            runs = _runs(p.chars)
            boxes, marks = [], []
            for o in p.curves + p.lines + p.rects:
                w, h = o["x1"] - o["x0"], o["bottom"] - o["top"]
                col = _color(o)
                bbox = (o["x0"], o["top"], o["x1"], o["bottom"])
                is_box_col = len(col) == 3 and abs(col[0] - 0.12) < 0.03 and abs(col[1] - 0.08) < 0.03
                if o["object_type"] == "curve" and is_box_col and 5.5 <= w <= 11.5 and 5.5 <= h <= 11.5 \
                        and abs(w - h) < 1.5:
                    boxes.append(bbox)            # printed empty checkbox
                elif w < 20 and h < 20 and not is_box_col:
                    marks.append(bbox)            # tick / cross / hatch (blue or black pen)
            hand = [dict(text=c["text"], x0=c["x0"], x1=c["x1"], top=c["top"], bottom=c["bottom"])
                    for c in p.chars if not _is_print(c["fontname"])]
            lay = PageLayout(runs, boxes, float(p.width), float(p.height))
            pt = classify_page(lay)
            pn = None
            for r in runs:
                if r.text.startswith("Patiente fictive n°"):
                    try:
                        pn = int(r.text.split("°")[1].split("/")[0])
                    except Exception:
                        pass
            pages.append(PdfPage(lay, hand, marks, pt, pn))
    return pages


def _inside(c, b):
    cx, cy = (c["x0"] + c["x1"]) / 2, (c["top"] + c["bottom"]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def _overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _hand_runs(page: PdfPage) -> list[dict]:
    """Group handwritten chars into the strings that were written (stream order + adjacency).
    A value is assigned to the region where its first character sits, so long values that
    overflow their cell are not split between columns."""
    if getattr(page, "_hruns", None) is not None:
        return page._hruns
    out, cur = [], None
    for c in page.hand:
        if cur and -2 < c["x0"] - cur["x1"] < 6 and abs(c["top"] - cur["top"]) < 5:
            cur["text"] += c["text"]
            cur["x1"] = c["x1"]
        else:
            cur = {"text": c["text"], "x1": c["x1"], "top": c["top"], "first": c}
            out.append(cur)
    page._hruns = out
    return out


def ground_truth(page: PdfPage) -> dict:
    """{field_id: {'raw': str|None|bool, 'status': ...}} for every data field of this page type."""
    out = {}
    for s in DATA_SPECS:
        if s.page != page.page_type:
            continue
        reg = resolve(s, page.layout)
        if reg is None:
            out[s.id] = {"raw": None, "status": "UNRESOLVED"}
            continue
        if "box" in reg:
            box = reg["box"]
            area = (box[2] - box[0]) * (box[3] - box[1])
            ticked = sum(_overlap(m, box) for m in page.marks) > 0.15 * area
            out[s.id] = {"raw": ticked, "status": "KNOWN"}
            continue
        txt = "".join(r["text"] for r in _hand_runs(page) if _inside(r["first"], reg["bbox"])).strip()
        txt = " ".join(txt.split())
        # Some handwriting fonts of the specimen lack glyphs (é, è, ç, —): the text layer holds
        # U+0000 and nothing is drawn. Ground truth = what is visible: a value made only of
        # missing glyphs is blank on paper; partial gaps become the wildcard U+FFFD.
        if txt.replace("\x00", "").strip() == "":
            out[s.id] = {"raw": None, "status": "NOT_PROVIDED", "note": "glyphs_not_rendered" if txt else ""}
            continue
        txt = txt.replace("\x00", "\ufffd")
        if not txt:
            out[s.id] = {"raw": None, "status": "NOT_PROVIDED"}
        elif txt in ("—", "-", "–", "/"):
            out[s.id] = {"raw": txt, "status": "NOT_APPLICABLE"}
        else:
            out[s.id] = {"raw": txt, "status": "KNOWN"}
    return out


def template_for(page: PdfPage) -> dict:
    regs = {}
    for s in SPECS:
        if s.page == page.page_type:
            r = resolve(s, page.layout)
            if r:
                regs[s.id] = r
    return {
        "page_type": page.page_type,
        "size": [page.layout.width, page.layout.height],
        "runs": [{"text": r.text, "bbox": list(r.bbox)} for r in page.layout.runs
                 if "fictive" not in r.text.lower() and "SPÉCIMEN" not in r.text],
        "boxes": [list(b) for b in page.layout.boxes],
        "regions": regs,
    }
