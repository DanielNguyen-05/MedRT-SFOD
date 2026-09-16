#!/usr/bin/env python3
"""
Boundary-specific evaluation metrics for MedRT-SFSeg.

This module is EVALUATION-ONLY (never used during source-free adaptation) and
is meant to sit alongside `analyze_seg_results.py`, which already reports
ASD/HD95 using the project's HEAL-style unit-spacing convention. Here we add
the two boundary metrics requested for the boundary-routing analysis:

  - Boundary IoU     (Cheng et al., CVPR 2021 "Boundary IoU")
  - BF-score         (Boundary F1-score / "F-measure for boundary
                       localization", Csurka et al. style: precision/recall of
                       boundary pixels within a pixel tolerance)

Both are computed on binary masks in ORIGINAL image resolution (same
convention as `analyze_seg_results.py`'s `surface_metrics`). Distances are in
pixels; no physical spacing is assumed (see the project-wide ASD/HD95 note:
Kvasir-SEG / CVC-ClinicDB ship no calibrated spacing, so "mm" in this repo is
numerically pixels under a unit-spacing convention).

Empty-mask convention (kept consistent with `surface_metrics` in
analyze_seg_results.py):
  - both empty            -> boundary_iou = 1.0, bf_score = 1.0 (trivial agreement)
  - exactly one empty     -> boundary_iou = 0.0, bf_score = 0.0
  - both non-empty        -> standard computation
This differs slightly from the ASD/HD95 "NaN + exclude" convention because
Boundary IoU and BF-score are bounded [0,1] metrics that are conventionally
reported as 0 for a total miss rather than excluded; callers that want the
ASD-style NaN/exclude behavior instead can use `boundary_status` in the
returned dict to re-derive that at the aggregation stage.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _mask_boundary_band(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Boundary-IoU-style boundary band: mask minus its (bordered) erosion.

    Matches the official Boundary IoU reference implementation
    (bowenc0221/boundary-iou-api): pad by 1px, erode with a 3x3 kernel for
    `dilation` iterations (dilation = round(dilation_ratio * image_diagonal),
    floored at 1), then crop back. The result is a `dilation`-pixel-wide band
    hugging the mask boundary from the inside.
    """
    x = (mask > 0).astype(np.uint8)
    h, w = x.shape
    img_diag = math.sqrt(h * h + w * w)
    dilation = max(1, int(round(dilation_ratio * img_diag)))

    if not x.any():
        return np.zeros_like(x, dtype=bool)

    padded = cv2.copyMakeBorder(x, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(padded, kernel, iterations=dilation)
    eroded = eroded[1 : h + 1, 1 : w + 1]
    return (x - eroded).astype(bool)


def boundary_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    dilation_ratio: float = 0.02,
) -> dict[str, Any]:
    """Boundary IoU (Cheng et al., CVPR 2021).

    IoU restricted to a `dilation_ratio`-wide band around each mask's own
    boundary (~2% of the image diagonal by default, matching the paper).
    """
    p = pred.astype(bool)
    g = gt.astype(bool)
    p_any, g_any = bool(p.any()), bool(g.any())

    if not p_any and not g_any:
        return {"boundary_iou": 1.0, "boundary_status": "both_empty"}
    if not p_any or not g_any:
        return {
            "boundary_iou": 0.0,
            "boundary_status": "pred_empty" if not p_any else "gt_empty",
        }

    p_band = _mask_boundary_band(p, dilation_ratio)
    g_band = _mask_boundary_band(g, dilation_ratio)

    inter = int(np.logical_and(p_band, g_band).sum())
    union = int(np.logical_or(p_band, g_band).sum())

    return {
        "boundary_iou": float(inter / union) if union > 0 else 0.0,
        "boundary_status": "ok",
        "pred_boundary_pixels": int(p_band.sum()),
        "gt_boundary_pixels": int(g_band.sum()),
    }


def _boundary_trace(mask: np.ndarray) -> np.ndarray:
    """One-pixel inner boundary trace (mask minus its default 4-conn erosion).

    Kept identical to `analyze_seg_results.py::binary_surface` so BF-score and
    ASD/HD95 are computed from the same boundary definition.
    """
    x = (mask > 0).astype(bool)
    if not x.any():
        return np.zeros_like(x, dtype=bool)
    eroded = binary_erosion(x)
    return np.logical_and(x, ~eroded)


def boundary_f_score(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance_px: float = 2.0,
) -> dict[str, Any]:
    """Boundary F-score (BF-score): precision/recall of boundary pixels
    within `tolerance_px` of the other mask's boundary trace.

    precision = fraction of predicted-boundary pixels within tolerance of the
                nearest GT-boundary pixel
    recall    = fraction of GT-boundary pixels within tolerance of the
                nearest predicted-boundary pixel
    """
    p = pred.astype(bool)
    g = gt.astype(bool)
    p_any, g_any = bool(p.any()), bool(g.any())

    if not p_any and not g_any:
        return {
            "bf_score": 1.0,
            "bf_precision": 1.0,
            "bf_recall": 1.0,
            "boundary_status": "both_empty",
        }
    if not p_any or not g_any:
        return {
            "bf_score": 0.0,
            "bf_precision": 0.0,
            "bf_recall": 0.0,
            "boundary_status": "pred_empty" if not p_any else "gt_empty",
        }

    p_trace = _boundary_trace(p)
    g_trace = _boundary_trace(g)

    if not p_trace.any() or not g_trace.any():
        return {
            "bf_score": 0.0,
            "bf_precision": 0.0,
            "bf_recall": 0.0,
            "boundary_status": "trace_empty",
        }

    dt_to_gt_trace = distance_transform_edt(~g_trace)
    dt_to_pred_trace = distance_transform_edt(~p_trace)

    precision = float((dt_to_gt_trace[p_trace] <= tolerance_px).mean())
    recall = float((dt_to_pred_trace[g_trace] <= tolerance_px).mean())
    f = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "bf_score": f,
        "bf_precision": precision,
        "bf_recall": recall,
        "boundary_status": "ok",
    }


def false_positive_region_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    min_component_area: int = 4,
) -> dict[str, Any]:
    """False-positive-specific metrics used for the hallucination-suppression
    comparison (Mask-DHF vs Mask-DHF+DURR on weak/no-evidence images).

    fp_pixel_rate       : FP pixels / total image pixels (always well-defined).
    fp_area_ratio        : FP pixels / GT positive pixels (only meaningful when
                            GT has foreground; NaN when gt is fully empty, since
                            the ratio has no natural reference size there -
                            use fp_pixel_rate / fp_cc_count on GT-empty images
                            instead).
    fp_cc_count           : number of 8-connected FP components with area >=
                             `min_component_area` (specks below that are
                             treated as noise, not "hallucinated regions").
    fp_cc_count_raw       : same, without the min-area filter.
    """
    p = pred.astype(bool)
    g = gt.astype(bool)
    h, w = p.shape

    fp_mask = np.logical_and(p, ~g)
    fp_pixels = int(fp_mask.sum())
    gt_pixels = int(g.sum())

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fp_mask.astype(np.uint8), connectivity=8
    )
    # stats[0] is the background component; skip it.
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.zeros((0,), dtype=np.int32)
    fp_cc_count_raw = int(areas.shape[0])
    fp_cc_count = int((areas >= min_component_area).sum())

    return {
        "fp_pixels": fp_pixels,
        "fp_pixel_rate": float(fp_pixels / (h * w)),
        "fp_area_ratio": float(fp_pixels / gt_pixels) if gt_pixels > 0 else float("nan"),
        "fp_cc_count": fp_cc_count,
        "fp_cc_count_raw": fp_cc_count_raw,
        "gt_pixels": gt_pixels,
        "gt_empty": int(gt_pixels == 0),
    }
