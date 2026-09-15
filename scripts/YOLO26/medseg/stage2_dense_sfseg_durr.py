"""
MedRT-SFSeg Stage-2 with DURR + SegMARD-v2.

Main experiment:
  Mask-DHF-style hard pseudo geometry
  + DURR (Dual-head Uncertainty Reliability Routing)
  + SegMARD-v2
  + epoch-level Mean-Teacher EMA

DURR-v1:
  - signed cross-head boundary routing (expand / shrink)
  - reliable O2M rescue persistence
  - safe hallucination suppression on Teacher-empty images

The module is TRAINING-ONLY. Deployment remains the adapted YOLO26-S-Seg
O2O inference graph.

This script NEVER reads target GT during adaptation.
It automatically saves BOTH final Student and final EMA Teacher checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

# Local repo must precede site-packages.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import distance_transform_edt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ultralytics import YOLO

from smoke_stage2_mt_seg import (
    TargetMTDataset,
    build_pseudo_batch,
    collate,
    list_images,
    resolve_device,
    seed_everything,
    setup_teacher_student,
    update_teacher_ema,
)
from segmard_seg import add_segmard_args, compute_segmard_loss
from durr_seg import (
    compute_directional_routing_loss,
    compute_rescue_loss,
    compute_safe_hallucination_loss,
    durr_ramp,
    generate_durr_pseudo_masks,
)


class SegmentInputFeatureHook:
    """Capture P3/P4/P5 features entering Segment26."""

    def __init__(self, model: nn.Module):
        self.latest = None
        self.head = model.model[-1]
        if not (
            hasattr(self.head, "one2one")
            and hasattr(self.head, "one2many")
        ):
            raise RuntimeError("Final head is not dual-head")
        self.handle = self.head.register_forward_pre_hook(self._hook)

    @staticmethod
    def _valid_feature_list(x):
        return (
            isinstance(x, (list, tuple))
            and len(x) >= 3
            and all(
                isinstance(t, torch.Tensor) and t.ndim == 4
                for t in x[:3]
            )
        )

    def _hook(self, _module, inputs):
        if (
            len(inputs) == 1
            and self._valid_feature_list(inputs[0])
        ):
            self.latest = list(inputs[0][:3])
        elif self._valid_feature_list(inputs):
            self.latest = list(inputs[:3])
        else:
            self.latest = None

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def average_confidence(labels) -> float:
    vals = [
        x[:, 4]
        for x in labels
        if x.numel()
    ]
    if not vals:
        return 0.0
    return float(torch.cat(vals).mean().item())


def mard_weight(
    args,
    global_step: int,
    steps_per_epoch: int,
    avg_conf: float,
) -> float:
    warmup_steps = max(
        1,
        int(args.mard_warmup_epochs * steps_per_epoch),
    )
    ramp = min(
        1.0,
        float(global_step) / float(warmup_steps),
    )
    gate = (
        avg_conf - args.mard_gate_threshold
    ) / max(1.0 - args.mard_gate_threshold, 1e-6)
    gate = float(np.clip(gate, 0.0, 1.0))
    return min(
        args.mard_lambda0 * ramp * gate,
        args.mard_lambda_max,
    )


def save_model(wrapper, model, path: Path) -> None:
    wrapper.model = model
    wrapper.save(str(path))



def _analysis_letterbox(
    image_bgr: np.ndarray,
    imgsz: int,
) -> tuple[torch.Tensor, dict]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = float(imgsz) / float(max(h, w))
    nh = int(round(h * scale))
    nw = int(round(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    tensor = (
        torch.from_numpy(np.ascontiguousarray(canvas))
        .permute(2, 0, 1)
        .float()
        / 255.0
    )
    return tensor, {
        "orig_h": h,
        "orig_w": w,
        "nh": nh,
        "nw": nw,
        "top": top,
        "left": left,
        "imgsz": imgsz,
    }


def _analysis_restore(
    x: torch.Tensor,
    meta: dict,
    *,
    binary: bool,
    clip_range: tuple[float, float] | None = (0.0, 1.0),
) -> np.ndarray:
    """Restore proto/letterbox-space maps to original image resolution.

    `clip_range=None` is required for signed DURR delta maps because negative
    values carry the shrink direction and must never be clipped away.
    """
    x = x.detach().float().cpu()
    y = x[None, None]
    imgsz = int(meta["imgsz"])
    if tuple(y.shape[-2:]) != (imgsz, imgsz):
        if binary:
            y = F.interpolate(y, size=(imgsz, imgsz), mode="nearest")
        else:
            y = F.interpolate(
                y,
                size=(imgsz, imgsz),
                mode="bilinear",
                align_corners=False,
            )

    top = int(meta["top"])
    left = int(meta["left"])
    nh = int(meta["nh"])
    nw = int(meta["nw"])
    crop = y[0, 0, top:top + nh, left:left + nw].numpy()

    interp = cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR
    out = cv2.resize(
        crop,
        (int(meta["orig_w"]), int(meta["orig_h"])),
        interpolation=interp,
    )
    if binary:
        return (out > 0.5).astype(np.uint8)
    out = out.astype(np.float32)
    if clip_range is not None:
        out = np.clip(out, clip_range[0], clip_range[1])
    return out


def _analysis_student_union(
    result,
    h: int,
    w: int,
) -> np.ndarray:
    if (
        result.masks is None
        or result.masks.data is None
        or result.masks.data.numel() == 0
    ):
        return np.zeros((h, w), dtype=np.uint8)
    m = result.masks.data.detach().float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    union = m.amax(dim=0, keepdim=True).unsqueeze(0)
    if tuple(union.shape[-2:]) != (h, w):
        union = F.interpolate(
            union,
            size=(h, w),
            mode="nearest",
        )
    return (
        (union[0, 0] > 0.5)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _analysis_binary_surface(mask: np.ndarray) -> np.ndarray:
    """One-pixel inner surface (3x3 erosion), matching our legacy evaluator."""
    x = (mask > 0).astype(np.uint8)
    if not x.any():
        return np.zeros_like(x, dtype=bool)
    eroded = cv2.erode(x, np.ones((3, 3), np.uint8), iterations=1)
    return np.logical_and(x.astype(bool), ~eroded.astype(bool))


def _analysis_surface_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    spacing_mm: tuple[float, float],
) -> dict[str, float | str | int]:
    """Compute Taha/Hanbury-style symmetric surface distances.

    HEAL (BMVC 2025) reports ASD in mm and cites Taha & Hanbury (2015).
    We therefore use the symmetric pooled surface-distance definition:

      ASD = [sum_{p in S(P)} d(p,S(G)) + sum_{g in S(G)} d(g,S(P))]
            / [|S(P)| + |S(G)|].

    Distances in `asd_heal_mm` / `hd95_mm` use `spacing_mm` through SciPy's
    Euclidean distance transform. CVC-ClinicDB and Kvasir-SEG image files do
    not provide calibrated physical pixel spacing; the default CLI spacing is
    (1.0, 1.0), which is a unit-spacing HEAL-compatible convention and is
    numerically equal to pixels. If true physical spacing is available, pass it
    explicitly with --analysis-spacing-mm-y/x.

    Empty-mask behavior is our explicit finite analysis convention (HEAL does
    not specify it in the paper): if exactly one mask is empty, use the physical
    image diagonal and record the status.
    """
    p = pred.astype(bool)
    g = gt.astype(bool)
    sy, sx = float(spacing_mm[0]), float(spacing_mm[1])
    h, w = p.shape
    diag_px = float(math.hypot(h, w))
    diag_mm = float(math.hypot(h * sy, w * sx))

    if not p.any() and not g.any():
        return {
            "asd_px": 0.0,
            "hd95_px": 0.0,
            "asd_heal_mm": 0.0,
            "hd95_mm": 0.0,
            "boundary_status": "both_empty",
            "pred_surface_pixels": 0,
            "gt_surface_pixels": 0,
        }
    if not p.any() or not g.any():
        return {
            "asd_px": diag_px,
            "hd95_px": diag_px,
            "asd_heal_mm": diag_mm,
            "hd95_mm": diag_mm,
            "boundary_status": "pred_empty" if not p.any() else "gt_empty",
            "pred_surface_pixels": int(_analysis_binary_surface(p).sum()),
            "gt_surface_pixels": int(_analysis_binary_surface(g).sum()),
        }

    ps = _analysis_binary_surface(p)
    gs = _analysis_binary_surface(g)
    if not ps.any() or not gs.any():
        return {
            "asd_px": diag_px,
            "hd95_px": diag_px,
            "asd_heal_mm": diag_mm,
            "hd95_mm": diag_mm,
            "boundary_status": "surface_empty",
            "pred_surface_pixels": int(ps.sum()),
            "gt_surface_pixels": int(gs.sum()),
        }

    # Pixel-space legacy metric (spacing=1), retained for continuity with prior runs.
    dt_gt_px = distance_transform_edt(~gs)
    dt_pr_px = distance_transform_edt(~ps)
    d_px = np.concatenate([dt_gt_px[ps], dt_pr_px[gs]]).astype(np.float64)

    # HEAL/Taha-Hanbury-compatible physical/unit-space metric.
    dt_gt_mm = distance_transform_edt(~gs, sampling=(sy, sx))
    dt_pr_mm = distance_transform_edt(~ps, sampling=(sy, sx))
    d_mm = np.concatenate([dt_gt_mm[ps], dt_pr_mm[gs]]).astype(np.float64)

    return {
        "asd_px": float(d_px.mean()),
        "hd95_px": float(np.percentile(d_px, 95.0)),
        "asd_heal_mm": float(d_mm.mean()),
        "hd95_mm": float(np.percentile(d_mm, 95.0)),
        "boundary_status": "ok",
        "pred_surface_pixels": int(ps.sum()),
        "gt_surface_pixels": int(gs.sum()),
    }


def _analysis_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    spacing_mm: tuple[float, float],
) -> dict[str, float | int | str]:
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    out: dict[str, float | int | str] = {
        "dice": _safe_div(2.0 * tp, 2.0 * tp + fp + fn),
        "iou": _safe_div(tp, tp + fp + fn),
        "precision": _safe_div(tp, tp + fp),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_pixels": int(p.sum()),
        "gt_pixels": int(g.sum()),
        "pred_empty": int(not p.any()),
    }
    out.update(_analysis_surface_metrics(pred, gt, spacing_mm=spacing_mm))
    return out


def _save_mask_png(path: Path, arr: np.ndarray, *, binary: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        y = (arr > 0.5).astype(np.uint8) * 255
    else:
        y = np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), y)


def _save_signed_map(path_png: Path, path_npy: Path, arr: np.ndarray) -> None:
    """Save raw signed delta as .npy and a zero-centered display PNG."""
    path_png.parent.mkdir(parents=True, exist_ok=True)
    path_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_npy, arr.astype(np.float32))
    vis = np.round((np.clip(arr, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(str(path_png), vis)


def _mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    vals = np.asarray([float(r[key]) for r in rows if key in r], dtype=np.float64)
    if vals.size == 0:
        return 0.0, 0.0
    return float(vals.mean()), float(vals.std(ddof=0))


def _fmt_mean_std(mean: float, std: float, scale: float = 1.0, nd: int = 3) -> str:
    return f"{mean * scale:.{nd}f} ± {std * scale:.{nd}f}"


def _write_dict_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def _save_post_training_panel(
    out_path: Path,
    image_bgr: np.ndarray,
    gt: np.ndarray | None,
    initial_route: dict,
    final_route: dict,
    student_mask: np.ndarray,
    title: str,
) -> None:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    gt_show = (
        gt
        if gt is not None
        else np.zeros(student_mask.shape, dtype=np.uint8)
    )
    error = (
        np.abs(student_mask.astype(np.float32) - gt.astype(np.float32))
        if gt is not None
        else np.zeros(student_mask.shape, dtype=np.float32)
    )

    items = [
        (rgb, "Image", None),
        (gt_show, "GT (post-training only)", (0, 1)),
        (
            initial_route["o2o"],
            "Initial Teacher O2O",
            (0, 1),
        ),
        (
            initial_route["fused"],
            "Initial Teacher fused",
            (0, 1),
        ),
        (
            final_route["fused"],
            "Final EMA Teacher fused",
            (0, 1),
        ),
        (
            final_route["o2m_witness"],
            "Final Teacher O2M witnesses",
            (0, 1),
        ),
        (
            final_route["signed_delta"],
            "Final signed O2M-O2O delta",
            (-1, 1),
        ),
        (
            final_route["rescue"],
            "Final O2M rescue",
            (0, 1),
        ),
        (
            student_mask,
            "Final Student prediction",
            (0, 1),
        ),
        (
            error,
            "Absolute Student error",
            (0, 1),
        ),
    ]

    for ax, (arr, label, lim) in zip(axes.flat, items):
        if lim is None:
            ax.imshow(arr)
        else:
            ax.imshow(
                arr,
                vmin=lim[0],
                vmax=lim[1],
            )
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_automatic_post_training_trace(
    *,
    args,
    device: torch.device,
    images: list,
    final_teacher: nn.Module,
    final_student_wrapper,
    out_dir: Path,
) -> None:
    """Automatic post-training Teacher/Student audit and two paper-style tables.

    Target GT is first touched here, after all Stage-2 optimization and checkpoint
    serialization have finished. Therefore it cannot affect adaptation.
    """
    trace_dir = out_dir / "final_analysis"
    panel_dir = trace_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    mask_root = trace_dir / "masks"
    mask_dirs = {
        "gt": mask_root / "gt",
        "initial_o2o": mask_root / "initial_teacher_o2o",
        "initial_fused": mask_root / "initial_teacher_fused",
        "final_o2o": mask_root / "final_teacher_o2o",
        "final_fused": mask_root / "final_teacher_fused",
        "final_o2m": mask_root / "final_teacher_o2m_witness",
        "final_rescue": mask_root / "final_teacher_o2m_rescue",
        "final_evidence": mask_root / "final_teacher_evidence",
        "final_safe_bg": mask_root / "final_teacher_safe_background",
        "signed_delta_png": mask_root / "final_signed_delta_vis",
        "signed_delta_npy": mask_root / "final_signed_delta_npy",
        "student": mask_root / "final_student",
        "student_error": mask_root / "student_absolute_error",
    }
    for d in mask_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    analysis_images = images
    if args.analysis_max_images > 0:
        analysis_images = analysis_images[:args.analysis_max_images]

    gt_dir = Path(args.analysis_gt_masks).resolve() if args.analysis_gt_masks else None
    spacing_mm = (
        float(args.analysis_spacing_mm_y),
        float(args.analysis_spacing_mm_x),
    )

    initial_wrapper = YOLO(args.weights)
    initial_teacher = initial_wrapper.model.to(device).float().eval()
    for p in initial_teacher.parameters():
        p.requires_grad = False

    final_teacher.eval()
    final_student_wrapper.model.eval()

    rows: list[dict] = []

    def teacher_route(model, x):
        (_labels, _masks, _rel, _ps, routes) = generate_durr_pseudo_masks(
            teacher=model,
            weak_imgs=x,
            tau_o2o=args.tau_o2o,
            tau_o2m=args.tau_o2m,
            tau_no=args.tau_no,
            tau_dup=args.tau_dup,
            tau_match=args.durr_tau_match,
            max_witnesses=args.durr_max_witnesses,
            mask_threshold=args.mask_thr,
            stability_low=args.stability_low,
            stability_high=args.stability_high,
            reliability_threshold=args.mask_rel_thr,
            min_mask_pixels=args.min_mask_pixels,
            boundary_kernel=args.durr_boundary_kernel,
            route_gain=args.durr_route_gain,
            route_min_disagreement=args.durr_min_disagreement,
            rescue_conf=args.durr_rescue_conf,
            rescue_stability=args.durr_rescue_stability,
            rescue_consensus_iou=args.durr_rescue_consensus_iou,
            rescue_min_support=args.durr_rescue_min_support,
            evidence_conf=args.durr_evidence_conf,
            safe_bg_teacher_prob=args.durr_safe_bg_teacher_prob,
        )
        return routes[0]

    stages = {
        "initial_o2o": "Initial Teacher O2O",
        "initial_fused": "Initial Teacher Fused",
        "final_teacher": "Final EMA Teacher Fused",
        "student": "Final Student",
    }

    for idx, image_path in enumerate(analysis_images, 1):
        image_path = Path(image_path)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print("[WARN analysis] cannot read", image_path)
            continue

        x_cpu, meta = _analysis_letterbox(bgr, args.imgsz)
        x = x_cpu.unsqueeze(0).to(device)

        r0 = teacher_route(initial_teacher, x)
        rT = teacher_route(final_teacher, x)

        initial_vis = {
            "o2o": _analysis_restore(r0["o2o_union"], meta, binary=True),
            "fused": _analysis_restore(r0["fused_union"], meta, binary=True),
        }
        final_vis = {
            "o2o": _analysis_restore(rT["o2o_union"], meta, binary=True),
            "fused": _analysis_restore(rT["fused_union"], meta, binary=True),
            "o2m_witness": _analysis_restore(rT["o2m_witness"], meta, binary=False),
            "signed_delta": _analysis_restore(
                rT["signed_delta"], meta, binary=False, clip_range=None
            ),
            "rescue": _analysis_restore(rT["rescue_mask"], meta, binary=True),
            "evidence": _analysis_restore(rT["teacher_evidence"], meta, binary=False),
            "safe_bg": _analysis_restore(rT["safe_bg_mask"].float(), meta, binary=True),
        }

        result = final_student_wrapper.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.analysis_conf,
            device=args.device,
            verbose=False,
        )[0]

        gt = None
        if gt_dir is not None:
            gt_path = gt_dir / f"{image_path.stem}.png"
            gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt_raw is None:
                print("[WARN analysis] missing GT", gt_path)
            else:
                gt = (gt_raw > 0).astype(np.uint8)

        h = int(gt.shape[0]) if gt is not None else int(bgr.shape[0])
        w = int(gt.shape[1]) if gt is not None else int(bgr.shape[1])
        student_mask = _analysis_student_union(result, h, w)

        # Save masks automatically so no separate tracing script is needed.
        stem = image_path.stem
        if gt is not None:
            _save_mask_png(mask_dirs["gt"] / f"{stem}.png", gt, binary=True)
        _save_mask_png(mask_dirs["initial_o2o"] / f"{stem}.png", initial_vis["o2o"], binary=True)
        _save_mask_png(mask_dirs["initial_fused"] / f"{stem}.png", initial_vis["fused"], binary=True)
        _save_mask_png(mask_dirs["final_o2o"] / f"{stem}.png", final_vis["o2o"], binary=True)
        _save_mask_png(mask_dirs["final_fused"] / f"{stem}.png", final_vis["fused"], binary=True)
        _save_mask_png(mask_dirs["final_o2m"] / f"{stem}.png", final_vis["o2m_witness"])
        _save_mask_png(mask_dirs["final_rescue"] / f"{stem}.png", final_vis["rescue"], binary=True)
        _save_mask_png(mask_dirs["final_evidence"] / f"{stem}.png", final_vis["evidence"])
        _save_mask_png(mask_dirs["final_safe_bg"] / f"{stem}.png", final_vis["safe_bg"], binary=True)
        _save_signed_map(
            mask_dirs["signed_delta_png"] / f"{stem}.png",
            mask_dirs["signed_delta_npy"] / f"{stem}.npy",
            final_vis["signed_delta"],
        )
        _save_mask_png(mask_dirs["student"] / f"{stem}.png", student_mask, binary=True)
        if gt is not None:
            err = np.abs(student_mask.astype(np.float32) - gt.astype(np.float32))
            _save_mask_png(mask_dirs["student_error"] / f"{stem}.png", err)

        title = image_path.name
        row: dict = {
            "image": image_path.name,
            "final_teacher_rescue_pixels": int((final_vis["rescue"] > 0).sum()),
            "final_teacher_route_pixels": int((np.abs(final_vis["signed_delta"]) >= args.durr_min_disagreement).sum()),
            "final_teacher_expand_pixels": int((final_vis["signed_delta"] >= args.durr_min_disagreement).sum()),
            "final_teacher_shrink_pixels": int((final_vis["signed_delta"] <= -args.durr_min_disagreement).sum()),
            "final_teacher_safe_bg_fraction": float((final_vis["safe_bg"] > 0).mean()),
        }

        if gt is not None:
            stage_masks = {
                "initial_o2o": initial_vis["o2o"],
                "initial_fused": initial_vis["fused"],
                "final_teacher": final_vis["fused"],
                "student": student_mask,
            }
            metrics = {
                key: _analysis_metrics(mask, gt, spacing_mm=spacing_mm)
                for key, mask in stage_masks.items()
            }
            for stage, m in metrics.items():
                for key, value in m.items():
                    row[f"{stage}_{key}"] = value

            # Explicit deltas for quick diagnosis.
            comparisons = {
                "fusion": (metrics["initial_fused"], metrics["initial_o2o"]),
                "teacher_drift": (metrics["final_teacher"], metrics["initial_fused"]),
                "student_vs_initial": (metrics["student"], metrics["initial_fused"]),
                "student_vs_final_teacher": (metrics["student"], metrics["final_teacher"]),
            }
            for name, (a, b) in comparisons.items():
                for key in (
                    "dice", "iou", "precision", "sensitivity", "specificity",
                    "asd_heal_mm", "hd95_mm", "asd_px", "hd95_px",
                ):
                    row[f"{name}_delta_{key}"] = float(a[key]) - float(b[key])

            title = (
                f"{image_path.name} | "
                f"T0-fused={metrics['initial_fused']['dice']:.4f} | "
                f"Tfinal={metrics['final_teacher']['dice']:.4f} | "
                f"Student={metrics['student']['dice']:.4f} | "
                f"S-Tfinal={metrics['student']['dice']-metrics['final_teacher']['dice']:+.4f} | "
                f"ASD_HEAL={metrics['student']['asd_heal_mm']:.2f}"
            )

        _save_post_training_panel(
            panel_dir / f"{stem}.png",
            bgr,
            gt,
            initial_vis,
            final_vis,
            student_mask,
            title,
        )
        rows.append(row)

        if idx == 1 or idx % 50 == 0 or idx == len(analysis_images):
            print(f"[AUTO ANALYSIS] {idx}/{len(analysis_images)}", flush=True)

    # Always save per-image trace as CSV + JSON.
    _write_dict_csv(trace_dir / "per_image_trace.csv", rows)
    (trace_dir / "per_image_trace.json").write_text(
        json.dumps({"num_images": len(rows), "rows": rows}, indent=2),
        encoding="utf-8",
    )

    summary: dict = {
        "gt_used": gt_dir is not None,
        "gt_policy": "post-training only; never used for adaptation/model selection",
        "num_images": len(rows),
        "asd_protocol": {
            "reference": "HEAL BMVC 2025 -> Taha & Hanbury 2015",
            "definition": "symmetric pooled mean of bidirectional nearest-surface distances",
            "spacing_mm_y": spacing_mm[0],
            "spacing_mm_x": spacing_mm[1],
            "important_unit_note": (
                "CVC-ClinicDB/Kvasir image files do not contain calibrated physical spacing. "
                "Default (1.0,1.0) is an explicit unit-spacing convention; numerically it equals "
                "pixels and must not be interpreted as measured physical millimetres unless true "
                "spacing is supplied."
            ),
            "empty_policy": "one empty mask -> image diagonal; both empty -> 0",
        },
    }

    # ------------------------------------------------------------
    # TABLE 1: segmentation quality progression.
    # ------------------------------------------------------------
    table1_rows: list[dict] = []
    if rows and gt_dir is not None:
        table1_raw: list[dict] = []
        for stage, label in stages.items():
            r: dict = {"stage": label}
            for metric in (
                "dice", "iou", "precision", "sensitivity", "specificity",
                "asd_heal_mm", "hd95_mm", "asd_px", "hd95_px",
            ):
                mu, sd = _mean_std(rows, f"{stage}_{metric}")
                r[f"{metric}_mean"] = mu
                r[f"{metric}_std"] = sd
            r["pred_empty_count"] = int(sum(int(x.get(f"{stage}_pred_empty", 0)) for x in rows))
            table1_raw.append(r)

            table1_rows.append({
                "stage": label,
                "dice_pct": _fmt_mean_std(r["dice_mean"], r["dice_std"], scale=100.0, nd=2),
                "iou_pct": _fmt_mean_std(r["iou_mean"], r["iou_std"], scale=100.0, nd=2),
                "precision_pct": _fmt_mean_std(r["precision_mean"], r["precision_std"], scale=100.0, nd=2),
                "sensitivity_pct": _fmt_mean_std(r["sensitivity_mean"], r["sensitivity_std"], scale=100.0, nd=2),
                "specificity_pct": _fmt_mean_std(r["specificity_mean"], r["specificity_std"], scale=100.0, nd=2),
                "asd_heal_mm": _fmt_mean_std(r["asd_heal_mm_mean"], r["asd_heal_mm_std"], nd=3),
                "hd95_mm": _fmt_mean_std(r["hd95_mm_mean"], r["hd95_mm_std"], nd=3),
                "pred_empty": r["pred_empty_count"],
            })

        _write_dict_csv(trace_dir / "table1_quality_progression_raw.csv", table1_raw)
        _write_dict_csv(trace_dir / "table1_quality_progression.csv", table1_rows)
        t1_cols = [
            ("stage", "Stage"),
            ("dice_pct", "Dice % ↑"),
            ("iou_pct", "IoU % ↑"),
            ("precision_pct", "Precision % ↑"),
            ("sensitivity_pct", "Sensitivity % ↑"),
            ("specificity_pct", "Specificity % ↑"),
            ("asd_heal_mm", "ASD_HEAL (mm*) ↓"),
            ("hd95_mm", "HD95 ↓"),
            ("pred_empty", "Pred-empty"),
        ]
        (trace_dir / "table1_quality_progression.md").write_text(
            "# Table 1 — Segmentation quality progression\n\n"
            + _markdown_table(table1_rows, t1_cols), encoding="utf-8"
        )

        # --------------------------------------------------------
        # TABLE 2: adaptation diagnosis / deltas.
        # --------------------------------------------------------
        comparison_defs = [
            ("fusion", "Mask-DHF fusion: Initial Fused − Initial O2O"),
            ("teacher_drift", "EMA Teacher drift: Final Teacher − Initial Fused"),
            ("student_vs_initial", "Self-training: Final Student − Initial Fused"),
            ("student_vs_final_teacher", "Student gap: Final Student − Final EMA Teacher"),
        ]
        table2_rows: list[dict] = []
        table2_raw: list[dict] = []
        for prefix, label in comparison_defs:
            raw = {"comparison": label}
            for metric in (
                "dice", "iou", "precision", "sensitivity", "specificity",
                "asd_heal_mm", "hd95_mm", "asd_px", "hd95_px",
            ):
                mu, sd = _mean_std(rows, f"{prefix}_delta_{metric}")
                raw[f"delta_{metric}_mean"] = mu
                raw[f"delta_{metric}_std"] = sd

            dice_vals = np.asarray([float(x[f"{prefix}_delta_dice"]) for x in rows])
            asd_vals = np.asarray([float(x[f"{prefix}_delta_asd_heal_mm"]) for x in rows])
            tol = 1e-12
            raw.update({
                "dice_improved": int((dice_vals > tol).sum()),
                "dice_worsened": int((dice_vals < -tol).sum()),
                "dice_tied": int((np.abs(dice_vals) <= tol).sum()),
                # Lower ASD is better, hence negative delta = improvement.
                "asd_improved": int((asd_vals < -tol).sum()),
                "asd_worsened": int((asd_vals > tol).sum()),
                "asd_tied": int((np.abs(asd_vals) <= tol).sum()),
            })
            table2_raw.append(raw)
            table2_rows.append({
                "comparison": label,
                "delta_dice_pp": _fmt_mean_std(raw["delta_dice_mean"], raw["delta_dice_std"], scale=100.0, nd=2),
                "delta_iou_pp": _fmt_mean_std(raw["delta_iou_mean"], raw["delta_iou_std"], scale=100.0, nd=2),
                "delta_sens_pp": _fmt_mean_std(raw["delta_sensitivity_mean"], raw["delta_sensitivity_std"], scale=100.0, nd=2),
                "delta_spec_pp": _fmt_mean_std(raw["delta_specificity_mean"], raw["delta_specificity_std"], scale=100.0, nd=2),
                "delta_asd_mm": _fmt_mean_std(raw["delta_asd_heal_mm_mean"], raw["delta_asd_heal_mm_std"], nd=3),
                "delta_hd95_mm": _fmt_mean_std(raw["delta_hd95_mm_mean"], raw["delta_hd95_mm_std"], nd=3),
                "dice_I_W_T": f"{raw['dice_improved']}/{raw['dice_worsened']}/{raw['dice_tied']}",
                "asd_I_W_T": f"{raw['asd_improved']}/{raw['asd_worsened']}/{raw['asd_tied']}",
            })

        _write_dict_csv(trace_dir / "table2_adaptation_diagnostics_raw.csv", table2_raw)
        _write_dict_csv(trace_dir / "table2_adaptation_diagnostics.csv", table2_rows)
        t2_cols = [
            ("comparison", "Comparison"),
            ("delta_dice_pp", "ΔDice pp ↑"),
            ("delta_iou_pp", "ΔIoU pp ↑"),
            ("delta_sens_pp", "ΔSens pp ↑"),
            ("delta_spec_pp", "ΔSpec pp ↑"),
            ("delta_asd_mm", "ΔASD ↓"),
            ("delta_hd95_mm", "ΔHD95 ↓"),
            ("dice_I_W_T", "Dice I/W/T"),
            ("asd_I_W_T", "ASD I/W/T"),
        ]
        (trace_dir / "table2_adaptation_diagnostics.md").write_text(
            "# Table 2 — Adaptation diagnostics\n\n"
            "I/W/T = improved / worsened / tied. For ASD/HD95, negative delta is better.\n\n"
            + _markdown_table(table2_rows, t2_cols), encoding="utf-8"
        )

        summary["table1_quality_progression_raw"] = table1_raw
        summary["table2_adaptation_diagnostics_raw"] = table2_raw

    # Optional official Ultralytics post-training validation for final Student.
    # This is performed only when the user explicitly supplies a target data YAML.
    if args.analysis_data_yaml:
        print("[AUTO VAL] final Student official mask metrics...", flush=True)
        val_result = final_student_wrapper.val(
            data=args.analysis_data_yaml,
            imgsz=args.imgsz,
            batch=args.analysis_batch,
            device=args.device,
            conf=args.analysis_conf,
            verbose=False,
        )
        official = {}
        results_dict = getattr(val_result, "results_dict", {}) or {}
        for key, value in results_dict.items():
            try:
                official[str(key)] = float(value)
            except Exception:
                pass
        summary["official_final_student_val"] = official
        (trace_dir / "official_final_student_val.json").write_text(
            json.dumps(official, indent=2), encoding="utf-8"
        )

    (trace_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("[AUTO ANALYSIS SAVED]", trace_dir)
    if table1_rows:
        print("[TABLE 1]", trace_dir / "table1_quality_progression.md")
        print("[TABLE 2]", trace_dir / "table2_adaptation_diagnostics.md")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--weights", required=True)
    ap.add_argument("--target-images", required=True)
    ap.add_argument(
        "--out-dir",
        default="runs/seg/dense_sfseg/cvc_durr_v1_segmard_v2",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--ema", type=float, default=0.999)

    # Mask-DHF base.
    ap.add_argument("--tau-o2o", type=float, default=0.5)
    ap.add_argument("--tau-o2m", type=float, default=0.5)
    ap.add_argument("--tau-no", type=float, default=0.2)
    ap.add_argument("--tau-dup", type=float, default=0.7)
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--stability-low", type=float, default=0.40)
    ap.add_argument("--stability-high", type=float, default=0.60)
    ap.add_argument("--mask-rel-thr", type=float, default=0.744898)
    ap.add_argument("--min-mask-pixels", type=int, default=16)

    # DURR matching / boundary routing.
    ap.add_argument("--durr-tau-match", type=float, default=0.5)
    ap.add_argument("--durr-max-witnesses", type=int, default=5)
    ap.add_argument("--durr-boundary-kernel", type=int, default=5)
    ap.add_argument("--durr-route-gain", type=float, default=1.0)
    ap.add_argument(
        "--durr-min-disagreement",
        type=float,
        default=0.02,
    )

    # DURR O2M rescue.
    ap.add_argument("--durr-rescue-conf", type=float, default=0.80)
    ap.add_argument(
        "--durr-rescue-stability",
        type=float,
        default=0.80,
    )
    ap.add_argument(
        "--durr-rescue-consensus-iou",
        type=float,
        default=0.70,
    )
    ap.add_argument(
        "--durr-rescue-min-support",
        type=int,
        default=0,
        help=(
            "0 = high-conf+stable rescue may stand alone; "
            "1 = require at least one additional O2M mask with consensus IoU."
        ),
    )

    # Safe teacher evidence / hallucination.
    ap.add_argument("--durr-evidence-conf", type=float, default=0.10)
    ap.add_argument(
        "--durr-safe-bg-teacher-prob",
        type=float,
        default=0.10,
    )
    ap.add_argument(
        "--durr-hall-student-thr",
        type=float,
        default=0.80,
    )
    ap.add_argument(
        "--durr-hall-area-thr",
        type=float,
        default=0.10,
    )
    ap.add_argument(
        "--durr-hall-area-weight",
        type=float,
        default=0.25,
    )

    # DURR loss weights.
    ap.add_argument("--durr-lambda-dir", type=float, default=0.10)
    ap.add_argument("--durr-lambda-rescue", type=float, default=0.20)
    ap.add_argument("--durr-lambda-hall", type=float, default=0.10)
    ap.add_argument(
        "--durr-warmup-epochs",
        type=float,
        default=5.0,
    )

    # SegMARD-v2.
    ap.add_argument("--mard-lambda0", type=float, default=0.05)
    ap.add_argument("--mard-lambda-max", type=float, default=0.2)
    ap.add_argument("--mard-gamma", type=float, default=1.0)
    ap.add_argument("--mard-alpha", type=float, default=1.0)
    ap.add_argument("--mard-beta", type=float, default=0.1)
    ap.add_argument("--mard-warmup-epochs", type=float, default=5.0)
    ap.add_argument("--mard-gate-threshold", type=float, default=0.5)
    ap.add_argument("--mard-topk-boxes", type=int, default=15)
    ap.add_argument("--mard-fg-points", type=int, default=8)
    ap.add_argument("--mard-bg-points", type=int, default=128)
    ap.add_argument("--mard-eta", type=float, default=12.0)
    ap.add_argument("--mard-box-conf", type=float, default=0.5)
    add_segmard_args(ap)

    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--print-freq", type=int, default=10)
    ap.add_argument("--save-interval", type=int, default=10)

    # Automatic post-training trace. GT is read ONLY after all training,
    # checkpoint saving, and EMA updates are finished.
    ap.add_argument(
        "--analysis-gt-masks",
        default=None,
        help=(
            "Optional GT-mask directory for automatic POST-TRAINING "
            "Teacher/Student visualization. Never read during adaptation."
        ),
    )
    ap.add_argument(
        "--analysis-conf",
        type=float,
        default=0.25,
    )
    ap.add_argument(
        "--analysis-data-yaml",
        default=None,
        help=(
            "Optional target dataset YAML for automatic POST-TRAINING Ultralytics "
            "mask mAP validation. Labels are never read during adaptation."
        ),
    )
    ap.add_argument(
        "--analysis-batch",
        type=int,
        default=8,
        help="Batch size for optional post-training official val.",
    )
    ap.add_argument(
        "--analysis-spacing-mm-y",
        type=float,
        default=1.0,
        help=(
            "Physical row spacing for HEAL/Taha-Hanbury ASD. CVC/Kvasir do not "
            "ship calibrated spacing, so 1.0 is an explicit unit-spacing convention."
        ),
    )
    ap.add_argument(
        "--analysis-spacing-mm-x",
        type=float,
        default=1.0,
        help="Physical column spacing for HEAL/Taha-Hanbury ASD.",
    )
    ap.add_argument(
        "--analysis-max-images",
        type=int,
        default=0,
        help="0 = analyze all target images after training.",
    )

    args = ap.parse_args()

    if args.analysis_spacing_mm_y <= 0 or args.analysis_spacing_mm_x <= 0:
        raise ValueError("analysis spacing must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)

    images = list_images(
        Path(args.target_images).resolve()
    )
    dataset = TargetMTDataset(images, args.imgsz)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate,
    )

    (
        teacher,
        student,
        student_wrapper,
        criterion,
    ) = setup_teacher_student(
        args.weights,
        device,
        epochs=args.epochs,
    )

    # Keep an unfused wrapper available to serialize the EMA Teacher.
    teacher_wrapper = YOLO(args.weights)

    optimizer = optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    out_dir = Path(args.out_dir).resolve()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hook = SegmentInputFeatureHook(student)

    print("=" * 78)
    print("MEDRT-SFSEG STAGE-2 | DURR-v1 + SegMARD-v2")
    print("=" * 78)
    print("target images :", len(images))
    print("labels used   : NO")
    print("GT masks used : NO")
    print("epochs        :", args.epochs)
    print("batch         :", args.batch)
    print()
    print("DURR          : ON")
    print(
        "signed route  : match",
        args.durr_tau_match,
        "| band",
        args.durr_boundary_kernel,
        "| min|Δ|",
        args.durr_min_disagreement,
    )
    print(
        "O2M rescue    : conf",
        args.durr_rescue_conf,
        "| stab",
        args.durr_rescue_stability,
        "| support",
        args.durr_rescue_min_support,
    )
    print(
        "hall suppress : P_s>=",
        args.durr_hall_student_thr,
        "| area>",
        args.durr_hall_area_thr,
        "| safe Teacher P<",
        args.durr_safe_bg_teacher_prob,
    )
    print(
        "DURR lambdas  :",
        args.durr_lambda_dir,
        args.durr_lambda_rescue,
        args.durr_lambda_hall,
    )
    print(
        "SegMARD-v2    : hard-bg-ratio",
        args.segmard_hard_bg_ratio,
        "| reliability weighting",
        args.segmard_reliability_weighting,
    )

    global_step = 0

    try:
        for epoch in range(args.epochs):
            student.train()
            start = time.time()

            successful = 0
            skipped = 0

            totals = {
                "loss": 0.0,
                "sfseg": 0.0,
                "seg": 0.0,
                "semseg": 0.0,
                "mard": 0.0,
                "lambda_mard": 0.0,
                "dir": 0.0,
                "rescue": 0.0,
                "hall": 0.0,
                "lambda_durr": 0.0,
                "pseudo": 0,
                "anchors": 0,
                "mask_extras": 0,
                "reject_rel": 0,
                "matched_anchors": 0,
                "witnesses": 0,
                "rescue_instances": 0,
                "teacher_empty_images": 0,
                "dir_pixels": 0.0,
                "expand_pixels": 0.0,
                "shrink_pixels": 0.0,
                "hall_triggered": 0.0,
                "hall_pixels": 0.0,
                "fg_tokens": 0.0,
                "hard_bg_tokens": 0.0,
                "easy_bg_tokens": 0.0,
            }

            for batch_i, (weak, strong, _paths) in enumerate(
                loader,
                start=1,
            ):
                if (
                    args.max_batches > 0
                    and batch_i > args.max_batches
                ):
                    break

                weak = weak.to(device, non_blocking=True)
                strong = strong.to(device, non_blocking=True)

                (
                    labels,
                    masks,
                    reliabilities,
                    ps,
                    routes,
                ) = generate_durr_pseudo_masks(
                    teacher=teacher,
                    weak_imgs=weak,
                    tau_o2o=args.tau_o2o,
                    tau_o2m=args.tau_o2m,
                    tau_no=args.tau_no,
                    tau_dup=args.tau_dup,
                    tau_match=args.durr_tau_match,
                    max_witnesses=args.durr_max_witnesses,
                    mask_threshold=args.mask_thr,
                    stability_low=args.stability_low,
                    stability_high=args.stability_high,
                    reliability_threshold=args.mask_rel_thr,
                    min_mask_pixels=args.min_mask_pixels,
                    boundary_kernel=args.durr_boundary_kernel,
                    route_gain=args.durr_route_gain,
                    route_min_disagreement=args.durr_min_disagreement,
                    rescue_conf=args.durr_rescue_conf,
                    rescue_stability=args.durr_rescue_stability,
                    rescue_consensus_iou=args.durr_rescue_consensus_iou,
                    rescue_min_support=args.durr_rescue_min_support,
                    evidence_conf=args.durr_evidence_conf,
                    safe_bg_teacher_prob=args.durr_safe_bg_teacher_prob,
                )

                valid = [
                    i for i, x in enumerate(labels)
                    if x.shape[0] > 0
                ]
                empty = [
                    i for i, x in enumerate(labels)
                    if x.shape[0] == 0
                ]

                optimizer.zero_grad(set_to_none=True)

                zero = next(student.parameters()).new_zeros(())
                sfseg_loss = zero
                mard_loss = zero
                dir_loss = zero
                rescue_loss = zero
                hall_loss = zero
                lambda_mard = 0.0

                loss_items = torch.zeros(
                    5,
                    device=device,
                )
                mard_stats = {}
                dir_stats = {}
                rescue_stats = {}
                hall_stats = {}

                # ----------------------------------------------------
                # Standard pseudo-supervised images.
                # ----------------------------------------------------
                if valid:
                    strong_valid = strong[valid]
                    labels_valid = [labels[i] for i in valid]
                    masks_valid = [masks[i] for i in valid]
                    rel_valid = [reliabilities[i] for i in valid]
                    routes_valid = [routes[i] for i in valid]

                    hook.latest = None
                    outputs_valid = student(strong_valid)
                    feats = hook.latest
                    if feats is None:
                        raise RuntimeError(
                            "SegMARD feature hook captured no features"
                        )

                    pseudo_batch = build_pseudo_batch(
                        labels_valid,
                        masks_valid,
                        strong_valid.shape,
                    )
                    det_vec, loss_items = criterion(
                        outputs_valid,
                        pseudo_batch,
                    )
                    sfseg_loss = det_vec.sum()

                    mard_loss, mard_stats = compute_segmard_loss(
                        feats=feats,
                        pseudo_labels=labels_valid,
                        pseudo_masks=masks_valid,
                        pseudo_reliabilities=None,
                        h_pad=int(strong_valid.shape[2]),
                        w_pad=int(strong_valid.shape[3]),
                        args=args,
                    )

                    avg_conf = average_confidence(labels_valid)
                    lambda_mard = mard_weight(
                        args,
                        global_step,
                        len(loader),
                        avg_conf,
                    )

                    dir_loss, dir_stats = (
                        compute_directional_routing_loss(
                            outputs_valid,
                            routes_valid,
                        )
                    )
                    rescue_loss, rescue_stats = (
                        compute_rescue_loss(
                            outputs_valid,
                            routes_valid,
                        )
                    )

                # ----------------------------------------------------
                # Teacher-empty images are NOT discarded anymore.
                # They only receive safe hallucination suppression.
                # This needs a second Student forward only for the
                # usually-small empty subset.
                # ----------------------------------------------------
                if empty:
                    strong_empty = strong[empty]
                    routes_empty = [routes[i] for i in empty]

                    # Do not let this auxiliary forward overwrite the
                    # already captured SegMARD features used above.
                    outputs_empty = student(strong_empty)

                    hall_loss, hall_stats = (
                        compute_safe_hallucination_loss(
                            outputs_empty,
                            routes_empty,
                            student_threshold=(
                                args.durr_hall_student_thr
                            ),
                            area_threshold=(
                                args.durr_hall_area_thr
                            ),
                            area_weight=(
                                args.durr_hall_area_weight
                            ),
                        )
                    )

                lambda_durr = durr_ramp(
                    global_step,
                    len(loader),
                    args.durr_warmup_epochs,
                )

                total_loss = (
                    sfseg_loss
                    + lambda_mard * mard_loss
                    + lambda_durr
                    * (
                        args.durr_lambda_dir * dir_loss
                        + args.durr_lambda_rescue * rescue_loss
                        + args.durr_lambda_hall * hall_loss
                    )
                )

                has_signal = bool(valid) or (
                    float(hall_stats.get(
                        "hall_triggered_images",
                        0.0,
                    )) > 0
                )

                if not has_signal:
                    skipped += 1
                    global_step += 1
                    continue

                if not torch.isfinite(total_loss):
                    raise RuntimeError("Non-finite DURR total loss")

                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    args.grad_clip,
                )
                if not torch.isfinite(
                    torch.as_tensor(grad_norm)
                ):
                    raise RuntimeError("Non-finite gradient")
                optimizer.step()

                global_step += 1
                successful += 1

                pseudo_n = sum(x.shape[0] for x in labels)

                totals["loss"] += float(total_loss.detach())
                totals["sfseg"] += float(sfseg_loss.detach())
                totals["seg"] += float(loss_items[1].detach())
                totals["semseg"] += float(loss_items[4].detach())
                totals["mard"] += float(mard_loss.detach())
                totals["lambda_mard"] += lambda_mard
                totals["dir"] += float(dir_loss.detach())
                totals["rescue"] += float(rescue_loss.detach())
                totals["hall"] += float(hall_loss.detach())
                totals["lambda_durr"] += lambda_durr
                totals["pseudo"] += pseudo_n
                totals["anchors"] += int(ps["anchors"])
                totals["mask_extras"] += int(
                    ps["mask_dhf_extras"]
                )
                totals["reject_rel"] += int(
                    ps["rejected_reliability"]
                )
                totals["matched_anchors"] += int(
                    ps["durr_matched_anchors"]
                )
                totals["witnesses"] += int(
                    ps["durr_witnesses"]
                )
                totals["rescue_instances"] += int(
                    ps["durr_rescue_instances"]
                )
                totals["teacher_empty_images"] += int(
                    ps["durr_teacher_empty_images"]
                )

                totals["dir_pixels"] += float(
                    dir_stats.get("dir_pixels", 0.0)
                )
                totals["expand_pixels"] += float(
                    dir_stats.get(
                        "dir_expand_pixels",
                        0.0,
                    )
                )
                totals["shrink_pixels"] += float(
                    dir_stats.get(
                        "dir_shrink_pixels",
                        0.0,
                    )
                )
                totals["hall_triggered"] += float(
                    hall_stats.get(
                        "hall_triggered_images",
                        0.0,
                    )
                )
                totals["hall_pixels"] += float(
                    hall_stats.get("hall_pixels", 0.0)
                )

                for key in (
                    "fg_tokens",
                    "hard_bg_tokens",
                    "easy_bg_tokens",
                ):
                    totals[key] += float(
                        mard_stats.get(key, 0.0)
                    )

                if (
                    batch_i == 1
                    or batch_i % args.print_freq == 0
                    or batch_i == len(loader)
                ):
                    print(
                        f"[E{epoch+1:02d}] "
                        f"batch={batch_i:03d}/{len(loader):03d} "
                        f"pseudo={pseudo_n} "
                        f"loss={float(total_loss.detach()):.4f} "
                        f"sfseg={float(sfseg_loss.detach()):.4f} "
                        f"seg={float(loss_items[1]):.4f} "
                        f"mard={float(mard_loss.detach()):.4f} "
                        f"λM={lambda_mard:.4f} "
                        f"DURR(dir/res/hall)="
                        f"{float(dir_loss.detach()):.4f}/"
                        f"{float(rescue_loss.detach()):.4f}/"
                        f"{float(hall_loss.detach()):.4f} "
                        f"λD={lambda_durr:.3f} "
                        f"route={int(dir_stats.get('dir_pixels', 0))} "
                        f"E/S="
                        f"{int(dir_stats.get('dir_expand_pixels', 0))}/"
                        f"{int(dir_stats.get('dir_shrink_pixels', 0))} "
                        f"rescue={int(ps['durr_rescue_instances'])} "
                        f"empty={int(ps['durr_teacher_empty_images'])} "
                        f"hall={int(hall_stats.get('hall_triggered_images', 0))}/"
                        f"{int(hall_stats.get('hall_pixels', 0))} "
                        f"FG/HBG/EBG="
                        f"{int(mard_stats.get('fg_tokens', 0))}/"
                        f"{int(mard_stats.get('hard_bg_tokens', 0))}/"
                        f"{int(mard_stats.get('easy_bg_tokens', 0))} "
                        f"grad={float(grad_norm):.2f}",
                        flush=True,
                    )

            if successful == 0:
                raise RuntimeError(
                    "Epoch had no optimization batches"
                )

            scheduler.step()
            if hasattr(criterion, "update"):
                criterion.update()

            # Epoch-level Mean Teacher EMA.
            update_teacher_ema(
                teacher,
                student,
                args.ema,
            )

            denom = max(successful, 1)
            print()
            print(
                f"[Epoch {epoch+1:02d}] DONE "
                f"time={time.time()-start:.1f}s "
                f"valid={successful} "
                f"skipped={skipped} "
                f"pseudo={totals['pseudo']} "
                f"anchors={totals['anchors']} "
                f"mask_extra={totals['mask_extras']} "
                f"reject_rel={totals['reject_rel']} "
                f"DURRmatch={totals['matched_anchors']}/"
                f"{totals['witnesses']} "
                f"rescue={totals['rescue_instances']} "
                f"teacher_empty={totals['teacher_empty_images']} "
                f"hall={int(totals['hall_triggered'])}/"
                f"{int(totals['hall_pixels'])} "
                f"loss={totals['loss']/denom:.4f} "
                f"sfseg={totals['sfseg']/denom:.4f} "
                f"mard={totals['mard']/denom:.4f} "
                f"dir={totals['dir']/denom:.4f} "
                f"res={totals['rescue']/denom:.4f} "
                f"hallL={totals['hall']/denom:.4f} "
                f"route E/S="
                f"{int(totals['expand_pixels'])}/"
                f"{int(totals['shrink_pixels'])}",
                flush=True,
            )

            save_now = (
                (epoch + 1) % args.save_interval == 0
                or (epoch + 1) == args.epochs
            )
            if save_now:
                hook.close()
                student.eval()
                teacher.eval()

                student_ckpt = (
                    ckpt_dir
                    / f"durr_student_epoch_{epoch+1}.pt"
                )
                teacher_ckpt = (
                    ckpt_dir
                    / f"durr_teacher_ema_epoch_{epoch+1}.pt"
                )

                save_model(
                    student_wrapper,
                    student,
                    student_ckpt,
                )
                save_model(
                    teacher_wrapper,
                    teacher,
                    teacher_ckpt,
                )
                print("[SAVE Student]", student_ckpt)
                print("[SAVE Teacher]", teacher_ckpt)

                if epoch + 1 < args.epochs:
                    student.train()
                    hook = SegmentInputFeatureHook(student)

    finally:
        if (
            hook is not None
            and hook.handle is not None
        ):
            hook.close()

    metadata = {
        "method": "MedRT-SFSeg + DURR-v1 + SegMARD-v2",
        "source_free": True,
        "target_labels_used": False,
        "target_gt_masks_used": False,
        "initial_weights": str(
            Path(args.weights).resolve()
        ),
        "epochs": args.epochs,
        "durr": {
            "name": "Dual-head Uncertainty Reliability Routing",
            "signed_boundary_routing": True,
            "reliable_o2m_rescue": True,
            "safe_hallucination_suppression": True,
            "tau_match": args.durr_tau_match,
            "max_witnesses": args.durr_max_witnesses,
            "boundary_kernel": args.durr_boundary_kernel,
            "route_gain": args.durr_route_gain,
            "min_disagreement":
                args.durr_min_disagreement,
            "rescue_conf": args.durr_rescue_conf,
            "rescue_stability":
                args.durr_rescue_stability,
            "rescue_consensus_iou":
                args.durr_rescue_consensus_iou,
            "rescue_min_support":
                args.durr_rescue_min_support,
            "evidence_conf":
                args.durr_evidence_conf,
            "safe_bg_teacher_prob":
                args.durr_safe_bg_teacher_prob,
            "hall_student_threshold":
                args.durr_hall_student_thr,
            "hall_area_threshold":
                args.durr_hall_area_thr,
            "lambda_dir": args.durr_lambda_dir,
            "lambda_rescue":
                args.durr_lambda_rescue,
            "lambda_hall":
                args.durr_lambda_hall,
            "warmup_epochs":
                args.durr_warmup_epochs,
        },
        "mask_dhf": {
            "tau_o2o": args.tau_o2o,
            "tau_o2m": args.tau_o2m,
            "tau_no": args.tau_no,
            "tau_dup": args.tau_dup,
            "stability_low": args.stability_low,
            "stability_high": args.stability_high,
            "reliability_threshold":
                args.mask_rel_thr,
            "reliability_threshold_source":
                "Q25 of initial AdaBN Teacher target O2M extras; label-free",
        },
        "segmard": {
            "hard_bg_ratio":
                args.segmard_hard_bg_ratio,
            "erode_kernel":
                args.segmard_erode_kernel,
            "dilate_kernel":
                args.segmard_dilate_kernel,
            "reliability_weighting":
                args.segmard_reliability_weighting,
        },
        "ema_momentum": args.ema,
        "ema_frequency": "epoch",
        "deployment": (
            "adapted Student O2O inference only; "
            "DURR and SegMARD are training-only"
        ),
        "checkpoint_policy": (
            "both Student and EMA Teacher are saved at every save interval"
        ),
        "post_training_analysis": {
            "automatic": True,
            "two_tables": True,
            "teacher_and_student_masks_saved": True,
            "heal_asd_reference": "HEAL BMVC 2025 -> Taha & Hanbury 2015",
            "spacing_mm_y": args.analysis_spacing_mm_y,
            "spacing_mm_x": args.analysis_spacing_mm_x,
            "unit_note": (
                "CVC/Kvasir provide no calibrated physical spacing; default 1.0/1.0 "
                "is unit spacing unless true spacing is explicitly supplied."
            ),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage2_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Automatically generate Teacher+Student trace; no extra script is needed.
    # GT, when supplied, is first touched here after all training is over.
    run_automatic_post_training_trace(
        args=args,
        device=device,
        images=images,
        final_teacher=teacher,
        final_student_wrapper=student_wrapper,
        out_dir=out_dir,
    )

    print()
    print("[PASS] DURR Stage-2 completed")
    print(
        "[NOTE] Both final Student and final EMA Teacher checkpoints "
        "were saved automatically."
    )


if __name__ == "__main__":
    main()