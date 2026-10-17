#!/usr/bin/env python3
"""
Multi-method, size-stratified, boundary-aware post-training evaluation.

Built for three specific requests on top of the existing
`analyze_seg_results.py` / `evaluate_durr_final.py` evaluators:

  1) Polyp-size-stratified table (Small / Medium / Large) across several
     checkpoints (e.g. Baseline / Mask-DHF / Mask-DHF+DURR / Full), reporting
     Dice, IoU, ASD, BF-score, Boundary IoU per size bin.

  2) Hallucination-suppression evidence: Precision, FP pixel rate, number of
     FP connected components, and FP area ratio, computed on the FULL target
     set and on a WEAK/NO-EVIDENCE subset (images whose GT polyp area is at
     or below a small threshold -- by default, GT-empty images). Intended to
     compare Mask-DHF vs Mask-DHF+DURR (or any pair of provided methods).

  3) Boundary-specific metrics: adds Boundary IoU and BF-score (in addition
     to the project's existing ASD/HD95) so the boundary-routing claim in the
     paper has a directly matching metric.

EVALUATION-ONLY. Target GT is read here only for post-hoc scoring, exactly
like `analyze_seg_results.py` / `evaluate_durr_final.py`; nothing here feeds
back into adaptation.

Usage
-----
python evaluate_stratified.py \
  --model baseline=runs/seg/source/frozen/kvasir_yolo26s_seg_source_best.pt \
  --model maskdhf=runs/seg/dense_sfseg/cvc_dense_60ep/checkpoints/dense_sfseg_epoch_60.pt \
  --model durr=runs/seg/dense_sfseg/cvc_durr_signed_60ep/checkpoints/durr_student_epoch_60.pt \
  --model full=runs/seg/dense_sfseg/cvc_full_60ep/checkpoints/durr_student_epoch_60.pt \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --out-dir runs/seg/analysis/stratified/k2c \
  --imgsz 640 --conf 0.25 --device 0

The --model flag is repeatable; the tag before '=' becomes the "Method"
column in the output tables (baseline/maskdhf/durr/full in the example,
but any tag string works, e.g. --model "Mask-DHF+DURR (signed)"=path.pt).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from boundary_metrics import (  # noqa: E402
    boundary_f_score,
    boundary_iou,
    false_positive_region_metrics,
)

EPS = 1e-8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BOUNDARY_VOXEL_SPACING = (1.0, 1.0)  # unit spacing; see project-wide ASD note.


def div(a: float, b: float) -> float:
    return float(a / (b + EPS))


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def union_masks(result, h: int, w: int) -> np.ndarray:
    """Identical semantics to analyze_seg_results.py / eval_source_seg.py."""
    if result.masks is None or result.masks.data is None or result.masks.data.numel() == 0:
        return np.zeros((h, w), dtype=np.uint8)
    masks = result.masks.data.detach().float()
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    merged = masks.amax(dim=0, keepdim=True).unsqueeze(0)
    if tuple(merged.shape[-2:]) != (h, w):
        merged = F.interpolate(merged, size=(h, w), mode="nearest")
    return (merged[0, 0] > 0.5).cpu().numpy().astype(np.uint8)


def overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    p, g = pred.astype(bool), gt.astype(bool)
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
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pred_pixels": int(p.sum()), "gt_pixels": int(g.sum()),
    }


def surface_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | str]:
    """HEAL-style ASD/HD95, kept byte-identical in convention to
    analyze_seg_results.py (unit spacing; NaN + exclude on empty masks)."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    p, g = pred.astype(bool), gt.astype(bool)
    p_any, g_any = bool(p.any()), bool(g.any())
    if not p_any or not g_any:
        status = "both_empty" if not p_any and not g_any else ("pred_empty" if not p_any else "gt_empty")
        return {"asd": float("nan"), "hd95": float("nan"), "boundary_status": status}

    def trace(x):
        eroded = binary_erosion(x)
        return np.logical_and(x, ~eroded)

    ps, gs = trace(p), trace(g)
    if not ps.any() or not gs.any():
        return {"asd": float("nan"), "hd95": float("nan"), "boundary_status": "surface_empty"}

    dt_gt = distance_transform_edt(~gs, sampling=BOUNDARY_VOXEL_SPACING)
    dt_pr = distance_transform_edt(~ps, sampling=BOUNDARY_VOXEL_SPACING)
    sd1 = float(dt_gt[ps].mean())
    sd2 = float(dt_pr[gs].mean())
    asd = (sd1 + sd2) / 3.0  # matches analyze_seg_results.py HEAL-repo formula.

    d_pred_to_gt = dt_gt[ps]
    d_gt_to_pred = dt_pr[gs]
    pooled = np.concatenate([d_pred_to_gt, d_gt_to_pred]).astype(np.float64)
    hd95 = float(np.percentile(pooled, 95.0))
    return {"asd": float(asd), "hd95": hd95, "boundary_status": "ok"}


def mean_std(values: list[float]) -> tuple[float, float, int, int]:
    x = np.asarray(values, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0, int(x.size)
    return float(finite.mean()), float(finite.std()), int(finite.size), int(x.size - finite.size)


def fmt(mu: float, sd: float, nd: int = 4, scale: float = 1.0) -> str:
    if not np.isfinite(mu):
        return "n/a"
    return f"{mu * scale:.{nd}f} ± {sd * scale:.{nd}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(r.get(k, "")) for k, _ in columns) + " |" for r in rows]
    return "\n".join([header, sep, *body]) + "\n"


def parse_model_args(values: list[str]) -> list[tuple[str, Path]]:
    out = []
    for v in values:
        if "=" not in v:
            raise ValueError(f"--model must be TAG=PATH, got: {v!r}")
        tag, path = v.split("=", 1)
        tag = tag.strip()
        p = Path(path.strip()).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        out.append((tag, p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", action="append", required=True,
        help="Repeatable TAG=PATH, e.g. --model baseline=weights.pt --model full=weights2.pt",
    )
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt-masks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--boundary-tolerance-px", type=float, default=2.0, help="BF-score pixel tolerance.")
    ap.add_argument("--boundary-iou-dilation-ratio", type=float, default=0.02)
    ap.add_argument("--min-fp-component-area", type=int, default=4)
    ap.add_argument(
        "--weak-evidence-max-gt-px", type=int, default=0,
        help="Images with GT positive-pixel count <= this are the 'weak/no polyp evidence' "
             "subset used for the hallucination-suppression comparison. Default 0 = strictly "
             "GT-empty images.",
    )
    ap.add_argument(
        "--size-thresholds", default=None,
        help="Optional 'T1,T2' GT-pixel thresholds for Small<=T1<Medium<=T2<Large. "
             "If omitted, thresholds are the 33rd/66th percentile of GT-positive-pixel "
             "counts over images with a nonempty GT mask, computed from this run and printed.",
    )
    args = ap.parse_args()

    models = parse_model_args(args.model)
    image_dir = Path(args.images).resolve()
    gt_dir = Path(args.gt_masks).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    print("=" * 78)
    print("STRATIFIED / BOUNDARY / HALLUCINATION EVALUATION")
    print("=" * 78)
    print("methods :", [t for t, _ in models])
    print("images  :", len(image_paths))
    print("out-dir :", out_dir)
    print()

    # ------------------------------------------------------------
    # Pass 1: load GT once, determine size bins from GT alone (same for
    # every method since GT does not depend on the model).
    # ------------------------------------------------------------
    gt_cache: dict[str, np.ndarray] = {}
    gt_pixel_counts: dict[str, int] = {}
    for image_path in image_paths:
        gt_path = gt_dir / f"{image_path.stem}.png"
        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_raw is None:
            raise FileNotFoundError(gt_path)
        gt = (gt_raw > 0).astype(np.uint8)
        gt_cache[image_path.stem] = gt
        gt_pixel_counts[image_path.stem] = int(gt.sum())

    nonempty_counts = np.asarray(
        [c for c in gt_pixel_counts.values() if c > 0], dtype=np.float64
    )
    if args.size_thresholds:
        t1_s, t2_s = args.size_thresholds.split(",")
        t1, t2 = float(t1_s), float(t2_s)
    elif nonempty_counts.size >= 3:
        t1 = float(np.quantile(nonempty_counts, 1.0 / 3.0))
        t2 = float(np.quantile(nonempty_counts, 2.0 / 3.0))
    else:
        t1 = t2 = 0.0

    def size_bin(gt_px: int) -> str | None:
        if gt_px <= args.weak_evidence_max_gt_px:
            return None  # not a polyp image; excluded from size table
        if gt_px <= t1:
            return "small"
        if gt_px <= t2:
            return "medium"
        return "large"

    print(f"[SIZE BINS] GT-positive-pixel thresholds: small<={t1:.1f}px, "
          f"medium<={t2:.1f}px, large>{t2:.1f}px "
          f"(n_nonempty={int(nonempty_counts.size)})")
    print(f"[WEAK EVIDENCE] gt_pixels <= {args.weak_evidence_max_gt_px} "
          f"-> {sum(1 for c in gt_pixel_counts.values() if c <= args.weak_evidence_max_gt_px)} "
          f"/ {len(gt_pixel_counts)} images")
    print()

    # ------------------------------------------------------------
    # Pass 2: per-method, per-image inference + full metric suite.
    # ------------------------------------------------------------
    all_rows: list[dict[str, Any]] = []

    for tag, model_path in models:
        print(f"--- method: {tag} ({model_path.name}) ---")
        model = YOLO(str(model_path), task="segment")

        for idx, image_path in enumerate(image_paths, start=1):
            stem = image_path.stem
            gt = gt_cache[stem]
            gt_px = gt_pixel_counts[stem]

            result = model.predict(
                source=str(image_path), imgsz=args.imgsz, conf=args.conf,
                device=args.device, verbose=False,
            )[0]
            pred = union_masks(result, *gt.shape)

            m = overlap_metrics(pred, gt)
            s = surface_metrics(pred, gt)
            biou = boundary_iou(pred, gt, dilation_ratio=args.boundary_iou_dilation_ratio)
            bf = boundary_f_score(pred, gt, tolerance_px=args.boundary_tolerance_px)
            fpm = false_positive_region_metrics(
                pred, gt, min_component_area=args.min_fp_component_area
            )

            row = {
                "method": tag,
                "image": image_path.name,
                "gt_pixels": gt_px,
                "size_bin": size_bin(gt_px) or "n/a",
                "weak_evidence": int(gt_px <= args.weak_evidence_max_gt_px),
                "dice": m["dice"], "iou": m["iou"],
                "precision": m["precision"], "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                "asd": s["asd"], "hd95": s["hd95"], "boundary_status": s["boundary_status"],
                "boundary_iou": biou["boundary_iou"],
                "bf_score": bf["bf_score"], "bf_precision": bf["bf_precision"], "bf_recall": bf["bf_recall"],
                "fp_pixel_rate": fpm["fp_pixel_rate"],
                "fp_area_ratio": fpm["fp_area_ratio"],
                "fp_cc_count": fpm["fp_cc_count"],
                "fp_cc_count_raw": fpm["fp_cc_count_raw"],
            }
            all_rows.append(row)

            if idx == 1 or idx % 50 == 0 or idx == len(image_paths):
                print(f"  [{idx:04d}/{len(image_paths):04d}] {image_path.name} "
                      f"Dice={row['dice']:.4f} BF={row['bf_score']:.4f} "
                      f"BIoU={row['boundary_iou']:.4f} FPrate={row['fp_pixel_rate']:.5f}")

    write_csv(out_dir / "per_image_metrics.csv", all_rows)
    (out_dir / "size_bin_thresholds.json").write_text(
        json.dumps({
            "gt_pixel_threshold_small_medium": t1,
            "gt_pixel_threshold_medium_large": t2,
            "weak_evidence_max_gt_px": args.weak_evidence_max_gt_px,
            "n_images_total": len(image_paths),
            "n_images_nonempty_gt": int(nonempty_counts.size),
        }, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Table 1: polyp-size-stratified Dice/IoU/ASD/BF-score/Boundary-IoU.
    # ------------------------------------------------------------
    size_order = ["small", "medium", "large", "all_polyps"]
    t1_rows_raw: list[dict[str, Any]] = []
    t1_rows_md: list[dict[str, Any]] = []
    for tag, _ in models:
        method_rows = [r for r in all_rows if r["method"] == tag]
        for bin_name in size_order:
            if bin_name == "all_polyps":
                subset = [r for r in method_rows if r["size_bin"] != "n/a"]
            else:
                subset = [r for r in method_rows if r["size_bin"] == bin_name]
            if not subset:
                continue
            raw: dict[str, Any] = {"method": tag, "size": bin_name, "n": len(subset)}
            md: dict[str, Any] = {"method": tag, "size": bin_name, "n": len(subset)}
            for metric, nd, scale in [
                ("dice", 2, 100.0), ("iou", 2, 100.0),
                ("asd", 3, 1.0), ("hd95", 3, 1.0),
                ("bf_score", 2, 100.0), ("boundary_iou", 2, 100.0),
            ]:
                mu, sd, n_valid, n_excl = mean_std([r[metric] for r in subset])
                raw[f"{metric}_mean"] = mu
                raw[f"{metric}_std"] = sd
                raw[f"{metric}_n_excluded"] = n_excl
                md[metric] = fmt(mu, sd, nd=nd, scale=scale)
            t1_rows_raw.append(raw)
            t1_rows_md.append(md)

    write_csv(out_dir / "table1_size_stratified_raw.csv", t1_rows_raw)
    write_csv(out_dir / "table1_size_stratified.csv", t1_rows_md)
    t1_cols = [
        ("method", "Method"), ("size", "Polyp size"), ("n", "N"),
        ("dice", "Dice % ↑"), ("iou", "IoU % ↑"),
        ("asd", "ASD ↓"), ("hd95", "HD95 ↓"),
        ("bf_score", "BF-score % ↑"), ("boundary_iou", "Boundary IoU % ↑"),
    ]
    (out_dir / "table1_size_stratified.md").write_text(
        "# Table 1 — Polyp-size-stratified evaluation\n\n"
        f"Size bins from GT-positive-pixel tertiles: small ≤ {t1:.0f}px, "
        f"medium ≤ {t2:.0f}px, large > {t2:.0f}px. "
        "`all_polyps` pools every image with a nonempty GT mask "
        "(GT-empty images are excluded from size stratification by definition).\n\n"
        + markdown_table(t1_rows_md, t1_cols),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Table 2: hallucination-suppression / false-positive comparison,
    # on the FULL set and on the weak/no-evidence subset.
    # ------------------------------------------------------------
    t2_rows_raw: list[dict[str, Any]] = []
    t2_rows_md: list[dict[str, Any]] = []
    for tag, _ in models:
        method_rows = [r for r in all_rows if r["method"] == tag]
        for subset_name, subset in [
            ("full_set", method_rows),
            ("weak_no_evidence", [r for r in method_rows if r["weak_evidence"] == 1]),
        ]:
            if not subset:
                continue
            raw: dict[str, Any] = {"method": tag, "subset": subset_name, "n": len(subset)}
            md: dict[str, Any] = {"method": tag, "subset": subset_name, "n": len(subset)}
            for metric, nd, scale in [
                ("precision", 2, 100.0),
                ("fp_pixel_rate", 4, 100.0),
                ("fp_cc_count", 3, 1.0),
                ("fp_area_ratio", 3, 1.0),
            ]:
                mu, sd, n_valid, n_excl = mean_std([r[metric] for r in subset])
                raw[f"{metric}_mean"] = mu
                raw[f"{metric}_std"] = sd
                raw[f"{metric}_n_valid"] = n_valid
                md[metric] = fmt(mu, sd, nd=nd, scale=scale)
            t2_rows_raw.append(raw)
            t2_rows_md.append(md)

    write_csv(out_dir / "table2_hallucination_fp_raw.csv", t2_rows_raw)
    write_csv(out_dir / "table2_hallucination_fp.csv", t2_rows_md)
    t2_cols = [
        ("method", "Method"), ("subset", "Subset"), ("n", "N"),
        ("precision", "Precision % ↑"),
        ("fp_pixel_rate", "FP pixel rate % ↓"),
        ("fp_cc_count", "# FP components ↓"),
        ("fp_area_ratio", "FP area ratio ↓ (n/a if GT empty)"),
    ]
    (out_dir / "table2_hallucination_fp.md").write_text(
        "# Table 2 — Hallucination / false-positive comparison\n\n"
        f"'weak_no_evidence' = images with GT positive-pixel count "
        f"<= {args.weak_evidence_max_gt_px} (default: strictly GT-empty). "
        "`fp_area_ratio` is undefined (n/a) on GT-empty images and excluded "
        "from that mean; use `fp_pixel_rate` / `fp_cc_count` for that subset. "
        "Compare e.g. the `maskdhf` and `full`/`durr` rows on the "
        "`weak_no_evidence` subset to evidence DURR's safe-hallucination-"
        "suppression mechanism.\n\n"
        + markdown_table(t2_rows_md, t2_cols),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Table 0: overall (unstratified) macro summary, for a quick sanity
    # cross-check against the paper's Table 1/2 numbers.
    # ------------------------------------------------------------
    t0_rows: list[dict[str, Any]] = []
    for tag, _ in models:
        method_rows = [r for r in all_rows if r["method"] == tag]
        row: dict[str, Any] = {"method": tag, "n": len(method_rows)}
        for metric, nd, scale in [
            ("dice", 2, 100.0), ("iou", 2, 100.0), ("precision", 2, 100.0),
            ("sensitivity", 2, 100.0), ("specificity", 2, 100.0),
            ("asd", 3, 1.0), ("hd95", 3, 1.0),
            ("bf_score", 2, 100.0), ("boundary_iou", 2, 100.0),
        ]:
            mu, sd, n_valid, n_excl = mean_std([r[metric] for r in method_rows])
            row[metric] = fmt(mu, sd, nd=nd, scale=scale)
        t0_rows.append(row)
    write_csv(out_dir / "table0_overall.csv", t0_rows)
    t0_cols = [
        ("method", "Method"), ("n", "N"),
        ("dice", "Dice % ↑"), ("iou", "IoU % ↑"), ("precision", "Precision % ↑"),
        ("sensitivity", "Sensitivity % ↑"), ("specificity", "Specificity % ↑"),
        ("asd", "ASD ↓"), ("hd95", "HD95 ↓"),
        ("bf_score", "BF-score % ↑"), ("boundary_iou", "Boundary IoU % ↑"),
    ]
    (out_dir / "table0_overall.md").write_text(
        "# Table 0 — Overall (unstratified) summary\n\n" + markdown_table(t0_rows, t0_cols),
        encoding="utf-8",
    )

    print()
    print("[SAVED]", out_dir / "table0_overall.md")
    print("[SAVED]", out_dir / "table1_size_stratified.md")
    print("[SAVED]", out_dir / "table2_hallucination_fp.md")
    print("[SAVED]", out_dir / "per_image_metrics.csv")
    print("[PASS] Stratified evaluation completed")


if __name__ == "__main__":
    main()
