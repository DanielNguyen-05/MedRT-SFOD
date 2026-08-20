#!/usr/bin/env python3
"""
Post-training segmentation analysis for MedRT-SFSeg.

This script is EVALUATION-ONLY: it reads target GT masks only after training is
finished and never feeds GT information back into adaptation.

Outputs
-------
<out-dir>/
  summary.json
  per_image_metrics.csv
  hard_cases.csv
  pred_masks/*.png
  plots/*.png
  visualizations/
      hard_by_dice/*.png
      hard_by_hd95/*.png
      hard_by_asd/*.png
      hard_by_fp/*.png
      hard_by_fn/*.png
      random/*.png

Metrics
-------
Overlap / pixel metrics:
  Dice, IoU, Precision, Sensitivity, Specificity
Boundary metrics (pixel units because CVC/Kvasir do not provide physical spacing):
  ASD  : symmetric average surface distance
  HD95 : 95th percentile of pooled bidirectional surface distances

Empty-mask convention
---------------------
- both prediction and GT empty: ASD=HD95=0
- exactly one empty: ASD=HD95=image diagonal (finite worst-case penalty)
  and boundary_status records the case explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


EPS = 1e-8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def div(a: float, b: float) -> float:
    return float(a / (b + EPS))


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def union_masks(result, h: int, w: int) -> np.ndarray:
    """Match the existing eval_source_seg.py union-mask semantics exactly."""
    if (
        result.masks is None
        or result.masks.data is None
        or result.masks.data.numel() == 0
    ):
        return np.zeros((h, w), dtype=np.uint8)

    masks = result.masks.data.detach().float()
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    merged = masks.amax(dim=0, keepdim=True).unsqueeze(0)
    if tuple(merged.shape[-2:]) != (h, w):
        merged = F.interpolate(
            merged,
            size=(h, w),
            mode="nearest",
        )

    return (
        (merged[0, 0] > 0.5)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


def overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    p = pred.astype(bool)
    g = gt.astype(bool)

    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())

    return {
        "dice": div(2 * tp, 2 * tp + fp + fn),
        "iou": div(tp, tp + fp + fn),
        "precision": div(tp, tp + fp),
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_pixels": int(p.sum()),
        "gt_pixels": int(g.sum()),
    }


def binary_surface(mask: np.ndarray) -> np.ndarray:
    """One-pixel inner surface using 3x3 erosion."""
    x = (mask > 0).astype(np.uint8)
    if not x.any():
        return np.zeros_like(x, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(x, kernel, iterations=1)
    return np.logical_and(x.astype(bool), ~eroded.astype(bool))


def distances_to_surface(surface: np.ndarray) -> np.ndarray:
    """Distance from every pixel to the nearest True surface pixel."""
    # OpenCV distanceTransform measures non-zero pixels to nearest zero pixel.
    # Therefore encode surface pixels as 0 and everything else as 1.
    source = (~surface.astype(bool)).astype(np.uint8)
    return cv2.distanceTransform(
        source,
        distanceType=cv2.DIST_L2,
        maskSize=cv2.DIST_MASK_PRECISE,
    )


def surface_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | str | int]:
    p = pred.astype(bool)
    g = gt.astype(bool)
    h, w = p.shape
    diagonal = float(math.hypot(h, w))

    p_any = bool(p.any())
    g_any = bool(g.any())

    if not p_any and not g_any:
        return {
            "asd": 0.0,
            "hd95": 0.0,
            "boundary_status": "both_empty",
            "pred_surface_pixels": 0,
            "gt_surface_pixels": 0,
        }

    if not p_any or not g_any:
        return {
            "asd": diagonal,
            "hd95": diagonal,
            "boundary_status": "pred_empty" if not p_any else "gt_empty",
            "pred_surface_pixels": int(binary_surface(p).sum()),
            "gt_surface_pixels": int(binary_surface(g).sum()),
        }

    ps = binary_surface(p)
    gs = binary_surface(g)

    # Defensive fallback for pathological one-pixel morphology cases.
    if not ps.any() or not gs.any():
        return {
            "asd": diagonal,
            "hd95": diagonal,
            "boundary_status": "surface_empty",
            "pred_surface_pixels": int(ps.sum()),
            "gt_surface_pixels": int(gs.sum()),
        }

    dt_to_gt = distances_to_surface(gs)
    dt_to_pred = distances_to_surface(ps)

    d_pred_to_gt = dt_to_gt[ps]
    d_gt_to_pred = dt_to_pred[gs]

    pooled = np.concatenate(
        [d_pred_to_gt.astype(np.float64), d_gt_to_pred.astype(np.float64)],
        axis=0,
    )

    return {
        "asd": float(pooled.mean()),
        "hd95": float(np.percentile(pooled, 95.0)),
        "boundary_status": "ok",
        "pred_surface_pixels": int(ps.sum()),
        "gt_surface_pixels": int(gs.sum()),
    }


def percentile_summary(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {}
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "max": float(x.max()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_single_plot(values: list[float], title: str, xlabel: str, out: Path) -> None:
    if plt is None or not values:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 4.5))
    ax = fig.add_subplot(111)
    ax.hist(values, bins=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Images")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_scatter(x: list[float], y: list[float], xlabel: str, ylabel: str, title: str, out: Path) -> None:
    if plt is None or not x or not y:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.0, 5.0))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=12, alpha=0.65)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """0=TN, 1=TP, 2=FP, 3=FN."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    out = np.zeros(gt.shape, dtype=np.uint8)
    out[np.logical_and(p, g)] = 1
    out[np.logical_and(p, ~g)] = 2
    out[np.logical_and(~p, g)] = 3
    return out


def contour_overlay(image_rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Draw GT and prediction contours on the image for visual diagnosis."""
    canvas = image_rgb.copy()
    gt_contours, _ = cv2.findContours(
        (gt > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    pred_contours, _ = cv2.findContours(
        (pred > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    # OpenCV draws in the array channel order. The exact colors are only a
    # visualization aid; titles identify the contours unambiguously.
    cv2.drawContours(canvas, gt_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(canvas, pred_contours, -1, (255, 0, 0), 2)
    return canvas


def save_case_panel(
    image_path: Path,
    gt_path: Path,
    pred_path: Path,
    row: dict[str, Any],
    out_path: Path,
) -> None:
    if plt is None:
        return

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or gt is None or pred is None:
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)
    err = make_error_map(pred, gt)
    overlay = contour_overlay(rgb, gt, pred)
    diff = np.abs(pred.astype(np.int16) - gt.astype(np.int16))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 6, figsize=(19, 3.6))
    axes[0].imshow(rgb)
    axes[0].set_title("Image")
    axes[1].imshow(gt, vmin=0, vmax=1)
    axes[1].set_title("GT")
    axes[2].imshow(pred, vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    axes[3].imshow(err, vmin=0, vmax=3)
    axes[3].set_title("Error: 1 TP / 2 FP / 3 FN")
    axes[4].imshow(overlay)
    axes[4].set_title("GT / Pred boundaries")
    axes[5].imshow(diff, vmin=0, vmax=1)
    axes[5].set_title("Absolute mask error")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{image_path.name} | "
        f"Dice={float(row['dice']):.4f}  "
        f"IoU={float(row['iou']):.4f}  "
        f"ASD={float(row['asd']):.2f}px  "
        f"HD95={float(row['hd95']):.2f}px"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def select_cases(rows: list[dict[str, Any]], topk: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    k = min(topk, len(rows))
    rng = random.Random(seed)
    random_rows = rng.sample(rows, k) if k > 0 else []
    return {
        "hard_by_dice": sorted(rows, key=lambda r: float(r["dice"]))[:k],
        "hard_by_hd95": sorted(rows, key=lambda r: float(r["hd95"]), reverse=True)[:k],
        "hard_by_asd": sorted(rows, key=lambda r: float(r["asd"]), reverse=True)[:k],
        "hard_by_fp": sorted(rows, key=lambda r: int(r["fp"]), reverse=True)[:k],
        "hard_by_fn": sorted(rows, key=lambda r: int(r["fn"]), reverse=True)[:k],
        "random": random_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt-masks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Compute metrics/CSV/plots only; skip per-case panels.",
    )
    args = ap.parse_args()

    model_path = Path(args.model).resolve()
    image_dir = Path(args.images).resolve()
    gt_dir = Path(args.gt_masks).resolve()
    out_dir = Path(args.out_dir).resolve()
    pred_dir = out_dir / "pred_masks"
    plot_dir = out_dir / "plots"
    vis_dir = out_dir / "visualizations"

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if not gt_dir.is_dir():
        raise FileNotFoundError(gt_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    print("=" * 78)
    print("MEDRT-SFSEG POST-TRAINING ANALYSIS")
    print("=" * 78)
    print("model        :", model_path)
    print("images       :", len(image_paths))
    print("GT used      : YES, EVALUATION ONLY")
    print("ASD/HD95 unit: pixels")
    print("output       :", out_dir)
    print()

    model = YOLO(str(model_path), task="segment")
    rows: list[dict[str, Any]] = []

    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for idx, image_path in enumerate(image_paths, start=1):
        gt_path = gt_dir / f"{image_path.stem}.png"
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(gt_path)
        gt = (gt > 0).astype(np.uint8)

        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]

        pred = union_masks(result, *gt.shape)
        cv2.imwrite(str(pred_dir / f"{image_path.stem}.png"), pred * 255)

        m = overlap_metrics(pred, gt)
        b = surface_metrics(pred, gt)

        image_pixels = int(gt.size)
        gt_pixels = int(m["gt_pixels"])
        pred_pixels = int(m["pred_pixels"])

        row: dict[str, Any] = {
            "image": image_path.name,
            "dice": float(m["dice"]),
            "iou": float(m["iou"]),
            "precision": float(m["precision"]),
            "sensitivity": float(m["sensitivity"]),
            "specificity": float(m["specificity"]),
            "asd": float(b["asd"]),
            "hd95": float(b["hd95"]),
            "boundary_status": str(b["boundary_status"]),
            "tp": int(m["tp"]),
            "fp": int(m["fp"]),
            "fn": int(m["fn"]),
            "tn": int(m["tn"]),
            "pred_pixels": pred_pixels,
            "gt_pixels": gt_pixels,
            "pred_area_fraction": div(pred_pixels, image_pixels),
            "gt_area_fraction": div(gt_pixels, image_pixels),
            "area_ratio_pred_to_gt": div(pred_pixels, gt_pixels),
            "fp_fraction_image": div(int(m["fp"]), image_pixels),
            "fn_fraction_gt": div(int(m["fn"]), gt_pixels),
            "pred_surface_pixels": int(b["pred_surface_pixels"]),
            "gt_surface_pixels": int(b["gt_surface_pixels"]),
        }
        rows.append(row)

        for key in totals:
            totals[key] += int(m[key])

        if idx == 1 or idx % 25 == 0 or idx == len(image_paths):
            print(
                f"[{idx:04d}/{len(image_paths):04d}] {image_path.name} "
                f"Dice={row['dice']:.4f} ASD={row['asd']:.2f}px HD95={row['hd95']:.2f}px"
            )

    metric_keys = [
        "dice", "iou", "precision", "sensitivity", "specificity", "asd", "hd95"
    ]
    macro = {
        key: float(np.mean([float(r[key]) for r in rows]))
        for key in metric_keys
    }

    tp, fp, fn, tn = totals["tp"], totals["fp"], totals["fn"], totals["tn"]
    global_metrics = {
        "dice": div(2 * tp, 2 * tp + fp + fn),
        "iou": div(tp, tp + fp + fn),
        "precision": div(tp, tp + fp),
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        **totals,
    }

    distributions = {
        key: percentile_summary([float(r[key]) for r in rows])
        for key in metric_keys
    }

    boundary_status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["boundary_status"])
        boundary_status_counts[status] = boundary_status_counts.get(status, 0) + 1

    params = sum(p.numel() for p in model.model.parameters())
    payload = {
        "model": str(model_path),
        "num_images": len(image_paths),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "params": int(params),
        "params_M": float(params / 1e6),
        "boundary_metric_unit": "pixels",
        "boundary_definition": {
            "surface": "one-pixel inner surface from 3x3 erosion",
            "asd": "mean of pooled bidirectional surface distances",
            "hd95": "95th percentile of pooled bidirectional surface distances",
            "one_empty_penalty": "image diagonal",
        },
        "medical_macro": macro,
        "medical_global": global_metrics,
        "distributions": distributions,
        "boundary_status_counts": boundary_status_counts,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_csv(out_dir / "per_image_metrics.csv", rows)

    selected = select_cases(rows, topk=args.topk, seed=args.seed)
    hard_rows: list[dict[str, Any]] = []
    for group, group_rows in selected.items():
        for rank, row in enumerate(group_rows, start=1):
            hard_rows.append({"group": group, "rank": rank, **row})
    write_csv(out_dir / "hard_cases.csv", hard_rows)

    save_single_plot(
        [float(r["dice"]) for r in rows],
        "Per-image Dice distribution",
        "Dice",
        plot_dir / "dice_hist.png",
    )
    save_single_plot(
        [float(r["asd"]) for r in rows],
        "Per-image ASD distribution",
        "ASD (pixels)",
        plot_dir / "asd_hist.png",
    )
    save_single_plot(
        [float(r["hd95"]) for r in rows],
        "Per-image HD95 distribution",
        "HD95 (pixels)",
        plot_dir / "hd95_hist.png",
    )
    save_scatter(
        [float(r["dice"]) for r in rows],
        [float(r["hd95"]) for r in rows],
        "Dice",
        "HD95 (pixels)",
        "Dice vs HD95",
        plot_dir / "dice_vs_hd95.png",
    )
    save_scatter(
        [float(r["asd"]) for r in rows],
        [float(r["hd95"]) for r in rows],
        "ASD (pixels)",
        "HD95 (pixels)",
        "ASD vs HD95",
        plot_dir / "asd_vs_hd95.png",
    )

    if not args.no_visualizations:
        row_by_image = {str(r["image"]): r for r in rows}
        for group, group_rows in selected.items():
            for rank, row in enumerate(group_rows, start=1):
                image_path = image_dir / str(row["image"])
                gt_path = gt_dir / f"{image_path.stem}.png"
                pred_path = pred_dir / f"{image_path.stem}.png"
                save_case_panel(
                    image_path=image_path,
                    gt_path=gt_path,
                    pred_path=pred_path,
                    row=row_by_image[image_path.name],
                    out_path=(
                        vis_dir
                        / group
                        / f"{rank:02d}_{image_path.stem}.png"
                    ),
                )

    print()
    print("=" * 78)
    print("POST-TRAINING SEGMENTATION ANALYSIS RESULTS")
    print("=" * 78)
    print(f"Macro Dice       : {macro['dice']:.6f}")
    print(f"Macro IoU        : {macro['iou']:.6f}")
    print(f"Macro Precision  : {macro['precision']:.6f}")
    print(f"Macro Sensitivity: {macro['sensitivity']:.6f}")
    print(f"Macro Specificity: {macro['specificity']:.6f}")
    print(f"Macro ASD        : {macro['asd']:.6f} px")
    print(f"Macro HD95       : {macro['hd95']:.6f} px")
    print(f"Params           : {params / 1e6:.3f} M")
    print("Boundary statuses:", boundary_status_counts)
    print("Saved summary    :", out_dir / "summary.json")
    print("Saved per-image  :", out_dir / "per_image_metrics.csv")
    print("Saved hard cases :", out_dir / "hard_cases.csv")
    if not args.no_visualizations:
        print("Saved visuals    :", vis_dir)
    print("[PASS] Post-training analysis completed")


if __name__ == "__main__":
    main()
