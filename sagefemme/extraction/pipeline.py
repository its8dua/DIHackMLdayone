"""Record-level extraction: several page photos -> one structured registry record.

capture_check()  — on-device, offline, instant: image quality gate, page type recognition,
                   location of direct identifiers (for redaction). No OCR of values.
extract_record() — online layer ("AI processing"): local template OCR and, when configured, the
                   vision-LLM on the *redacted* image; both readings are fused per field
                   (agreement raises confidence, disagreement forces midwife review).
"""
from __future__ import annotations

import io
import time
from typing import Optional

import numpy as np
from PIL import Image

from ..forms.align import align
from ..forms.layout import PAGE_TYPES, SPEC_BY_ID
from ..net import Network
from ..schema import FIELDS, normalize
from . import local_ocr
from .postprocess import consistency, empty_record_fields, from_reading, fv
from .vlm import VLMExtractor, redact


def load_image(data: bytes | Image.Image) -> Image.Image:
    if isinstance(data, Image.Image):
        return data.convert("RGB")
    im = Image.open(io.BytesIO(data))
    try:
        from PIL import ImageOps
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im.convert("RGB")


def capture_check(data: bytes | Image.Image) -> dict:
    img = load_image(data)
    rgb = np.array(img)
    q = local_ocr.image_quality(rgb)
    al = align(rgb)
    if not al.ok:
        q["issues"].append("page_not_recognised")
        q["ok"] = False
    pii = []
    if al.ok:
        from ..forms.align import templates
        for fid, reg in templates()[al.page_type]["regions"].items():
            if SPEC_BY_ID[fid].kind == "pii":
                pii.append(al.project(reg["bbox"]).tolist())
    return {"page_type": al.page_type if al.ok else None, "quality": q, "align_inliers": al.inliers,
            "pii_polys": pii}


def _agree(spec, a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    va, _ = normalize(spec, a)
    vb, _ = normalize(spec, b)
    if isinstance(va, str) and isinstance(vb, str):
        from .lexicon import _k
        return _k(va) == _k(vb)
    return va == vb


def fuse(fid: str, local: Optional[dict], remote: Optional[dict]) -> dict:
    if remote is None:
        return local
    if local is None:
        return remote
    spec = SPEC_BY_ID[fid]
    same_status = local["status"] == remote["status"] or {local["status"], remote["status"]} <= {"KNOWN", "NEEDS_REVIEW"}
    if same_status and _agree(spec, local["value"], remote["value"]):
        out = dict(remote)
        out["confidence"] = round(max(remote["confidence"], 1 - (1 - remote["confidence"]) * (1 - local["confidence"]), 0.0), 3)
        out["source"] = "ai_ensemble"
        thr = {1: .8, 2: .85, 3: .92}[spec.importance]
        if out["status"] == "NEEDS_REVIEW" and out["confidence"] >= thr and out["value"] is not None \
                and "hors" not in (out["note"] or ""):
            out["status"] = "KNOWN"
            out["note"] = "deux lectures concordantes"
        return out
    out = dict(remote)
    if local["confidence"] >= 0.8 and local["status"] == "KNOWN":
        out["status"] = "NEEDS_REVIEW"
        out["confidence"] = min(out["confidence"], 0.55)
        out["note"] = f"lectures divergentes : « {remote['value']} » / « {local['value']} »"
        out["candidates"] = [remote["value"], local["value"]]
    return out


def extract_record(pages: list[bytes | Image.Image], network: Network, use_vlm: Optional[bool] = None,
                   use_local: bool = True, progress=None) -> dict:
    t0 = time.time()
    vlm = VLMExtractor(network)
    use_vlm = vlm.available if use_vlm is None else (use_vlm and vlm.available)
    fields, meta = {}, []
    present = set()
    for i, data in enumerate(pages):
        network.require("AI processing")
        img = load_image(data)
        pr = local_ocr.read_page(img)
        m = {"index": i, "page_type": pr.page_type, "quality": pr.quality, "align_inliers": pr.align_inliers,
             "backend": [], "seconds": 0.0}
        if not pr.page_type:
            m["error"] = "page non reconnue"
            meta.append(m)
            continue
        present.add(pr.page_type)
        local = {}
        if use_local:
            m["backend"].append("local_ocr")
            for fid, r in pr.fields.items():
                local[fid] = from_reading(fid, r, pr.page_type, source="ai")
        remote = {}
        if use_vlm:
            m["backend"].append(f"vlm:{vlm.provider}")
            red = redact(img, pr.pii_polys)          # identifiers never leave the device
            for fid, r in vlm.read_page(red, pr.page_type).items():
                remote[fid] = from_reading(fid, r, pr.page_type, source="ai")
        for fid in set(local) | set(remote):
            fields[fid] = fuse(fid, local.get(fid), remote.get(fid) if remote else None)
        m["seconds"] = round(pr.seconds, 2)
        meta.append(m)
        if progress:
            progress(i + 1, len(pages))
    fields.update(empty_record_fields(present))
    for f in FIELDS:                               # field missing from template resolution
        fields.setdefault(f.id, fv(None, "NOT_PROVIDED", 1.0, "rule", None, "non localisé", f.page))
    msgs = consistency(fields)
    return {"fields": fields, "pages": meta, "consistency": msgs, "seconds": round(time.time() - t0, 1),
            "backends": sorted({b for m in meta for b in m["backend"]})}
