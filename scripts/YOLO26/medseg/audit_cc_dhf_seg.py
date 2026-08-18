#!/usr/bin/env python3
"""
Label-free audit for CC-DHF (Consensus-and-Coverage Dual-Head Fusion).

Purpose
-------
Before changing Mask-DHF, measure whether native O2O/O2M segmentation branches
contain useful cross-head disagreement on target images.

This script uses TARGET IMAGES ONLY. It never reads target labels or GT masks.

For every valid O2O anchor, it:
1) reconstructs the O2O soft mask;
2) reconstructs O2M soft masks before Mask-DHF novelty rejection;
3) assigns each O2M candidate to its best same-class O2O anchor by box IoU;
4) keeps matches with IoU >= --tau-match;
5) measures matched-witness count, box/mask IoU, soft disagreement,
   interior disagreement, boundary disagreement, and reliability statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from smoke_stage2_mt_seg import (  # noqa: E402
    TargetMTDataset,
    collate,
    list_images,
    resolve_device,
    seed_everything,
    setup_teacher_student,
)
from mask_dhf_seg import (  # noqa: E402
    mask_probs_from_coefficients,
    mask_stability,
)

EPS = 1e-8


def binary_dilate(mask: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    if kernel <= 1:
        return mask.bool()
    if kernel % 2 == 0:
        raise ValueError("Boundary kernel must be odd")
    x = mask.float()[None, None]
    y = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return y[0, 0] > 0.5


def binary_erode(mask: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    if kernel <= 1:
        return mask.bool()
    if kernel % 2 == 0:
        raise ValueError("Boundary kernel must be odd")
    inv = (~mask.bool()).float()[None, None]
    y = F.max_pool2d(inv, kernel_size=kernel, stride=1, padding=kernel // 2)
    return ~(y[0, 0] > 0.5)


def binary_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.bool()
    b = b.bool()
    inter = (a & b).sum().float()
    union = (a | b).sum().float()
    return float((inter / union.clamp_min(1.0)).item())


def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0.0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0.0)
    )
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0.0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0.0)
    )
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp_min(EPS)


def quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    x = np.asarray(values, dtype=np.float64)
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "max": float(np.max(x)),
    }


def print_quantiles(name: str, values: list[float]) -> None:
    q = quantiles(values)
    print()
    print(name)
    if q is None:
        print("  EMPTY")
        return
    for key in ("count", "mean", "std", "min", "q10", "q25", "median", "q75", "q90", "max"):
        value = q[key]
        if key == "count":
            print(f"  {key:>6}: {int(value)}")
        else:
            print(f"  {key:>6}: {value:.6f}")


def masked_mean(values: torch.Tensor, region: torch.Tensor) -> float | None:
    region = region.bool()
    if not region.any():
        return None
    return float(values[region].mean().item())


def prepare_predictions(
    rows: torch.Tensor,
    proto: torch.Tensor,
    input_h: int,
    input_w: int,
    conf_thr: float,
    mask_thr: float,
    min_mask_pixels: int,
    stability_low: float,
    stability_high: float,
):
    """
    Return aligned rows, soft_probs, stability, reliability.

    Only confidence and minimum-mask-area filters are applied. We intentionally
    do NOT apply O2M novelty filtering or NMS because CC-DHF wants to inspect
    the overlapping O2M predictions that current Mask-DHF discards.
    """
    rows = rows[rows[:, 4] >= conf_thr]

    probs = mask_probs_from_coefficients(
        rows,
        proto,
        input_h,
        input_w,
    )

    binary = probs >= mask_thr

    if rows.numel():
        keep = binary.sum(dim=(1, 2)) >= min_mask_pixels
        rows = rows[keep]
        probs = probs[keep]

    stability = mask_stability(
        probs,
        low=stability_low,
        high=stability_high,
    )

    reliability = torch.sqrt(
        (rows[:, 4] * stability).clamp_min(0.0)
    )

    return rows, probs, stability, reliability


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Label-free O2O/O2M disagreement audit before CC-DHF."
    )
    ap.add_argument("--weights", required=True)
    ap.add_argument("--target-images", required=True)
    ap.add_argument(
        "--out-json",
        default="runs/seg/benchmarks/cc_dhf_audit_summary.json",
    )
    ap.add_argument(
        "--out-csv",
        default="runs/seg/benchmarks/cc_dhf_audit_pairs.csv",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--tau-o2o", type=float, default=0.5)
    ap.add_argument("--tau-o2m", type=float, default=0.5)
    ap.add_argument(
        "--tau-match",
        type=float,
        default=0.5,
        help="Minimum same-class box IoU for O2M consensus witness matching.",
    )
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--stability-low", type=float, default=0.40)
    ap.add_argument("--stability-high", type=float, default=0.60)
    ap.add_argument("--min-mask-pixels", type=int, default=16)
    ap.add_argument(
        "--boundary-kernel",
        type=int,
        default=3,
        help="Odd morphology kernel in prototype-mask space.",
    )
    ap.add_argument(
        "--max-witnesses-per-anchor",
        type=int,
        default=5,
        help="Retain at most top-K O2M witnesses per anchor for pair statistics; 0=unlimited.",
    )
    ap.add_argument("--print-freq", type=int, default=20)
    return ap.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    if args.boundary_kernel % 2 == 0:
        raise ValueError("--boundary-kernel must be odd")

    seed_everything(args.seed)
    device = resolve_device(args.device)

    target_images = list_images(Path(args.target_images).resolve())
    dataset = TargetMTDataset(target_images, args.imgsz)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate,
    )

    teacher, _student, _wrapper, _criterion = setup_teacher_student(
        args.weights,
        device,
        epochs=1,
    )
    teacher.eval()

    pair_rows: list[dict] = []

    total_o2o = 0
    total_o2m = 0
    anchors_with_match = 0
    raw_matched_o2m = 0
    match_count_per_anchor: list[float] = []

    box_ious: list[float] = []
    mask_ious: list[float] = []
    mad_all: list[float] = []
    mad_interior: list[float] = []
    mad_boundary: list[float] = []
    boundary_interior_ratio: list[float] = []

    best_box_ious: list[float] = []
    best_mask_ious: list[float] = []
    best_mad_all: list[float] = []
    best_mad_interior: list[float] = []
    best_mad_boundary: list[float] = []
    best_boundary_interior_ratio: list[float] = []

    o2o_rels: list[float] = []
    o2m_rels: list[float] = []
    o2o_stabs: list[float] = []
    o2m_stabs: list[float] = []

    for batch_i, (weak, _strong, paths) in enumerate(loader, start=1):
        weak = weak.to(device, non_blocking=True)

        outputs = teacher(weak, augment=False, visualize=False)
        if not (isinstance(outputs, tuple) and len(outputs) == 2):
            raise RuntimeError("Unexpected Teacher output")

        first, branches = outputs
        if not (isinstance(first, tuple) and len(first) == 2):
            raise RuntimeError("Expected ((O2O, proto), branches)")

        final_o2o, proto = first
        if not isinstance(branches, dict):
            raise RuntimeError("Missing Teacher branch dictionary")

        head = teacher.model[-1]
        decoded_o2m = head._inference(branches["one2many"]).permute(0, 2, 1)
        final_o2m = head.postprocess(decoded_o2m)

        input_h = int(weak.shape[2])
        input_w = int(weak.shape[3])

        for local_i in range(weak.shape[0]):
            path_str = str(paths[local_i])

            anchors, anchor_probs, anchor_stability, anchor_reliability = prepare_predictions(
                final_o2o[local_i],
                proto[local_i],
                input_h,
                input_w,
                args.tau_o2o,
                args.mask_thr,
                args.min_mask_pixels,
                args.stability_low,
                args.stability_high,
            )

            candidates, candidate_probs, candidate_stability, candidate_reliability = prepare_predictions(
                final_o2m[local_i],
                proto[local_i],
                input_h,
                input_w,
                args.tau_o2m,
                args.mask_thr,
                args.min_mask_pixels,
                args.stability_low,
                args.stability_high,
            )

            total_o2o += int(anchors.shape[0])
            total_o2m += int(candidates.shape[0])

            if anchors.numel() == 0 or candidates.numel() == 0:
                for _ in range(anchors.shape[0]):
                    match_count_per_anchor.append(0.0)
                continue

            ious = box_iou_matrix(candidates[:, :4], anchors[:, :4])
            same_class = candidates[:, 5].long()[:, None] == anchors[:, 5].long()[None, :]
            ious = torch.where(same_class, ious, torch.full_like(ious, -1.0))

            best_iou_for_candidate, best_anchor_idx = ious.max(dim=1)
            matched_candidate = best_iou_for_candidate >= args.tau_match

            for anchor_idx in range(anchors.shape[0]):
                candidate_ids = torch.where(
                    matched_candidate & (best_anchor_idx == anchor_idx)
                )[0]

                raw_count = int(candidate_ids.numel())
                match_count_per_anchor.append(float(raw_count))

                if raw_count == 0:
                    continue

                anchors_with_match += 1
                raw_matched_o2m += raw_count

                candidate_ids = candidate_ids[
                    torch.argsort(ious[candidate_ids, anchor_idx], descending=True)
                ]

                if args.max_witnesses_per_anchor > 0:
                    candidate_ids = candidate_ids[: args.max_witnesses_per_anchor]

                anchor_prob = anchor_probs[anchor_idx]
                anchor_mask = anchor_prob >= args.mask_thr
                interior = binary_erode(anchor_mask, args.boundary_kernel)
                dilated = binary_dilate(anchor_mask, args.boundary_kernel)
                boundary = dilated & (~interior)

                for rank, candidate_idx_t in enumerate(candidate_ids):
                    candidate_idx = int(candidate_idx_t.item())
                    candidate_prob = candidate_probs[candidate_idx]
                    candidate_mask = candidate_prob >= args.mask_thr

                    pair_region = binary_dilate(
                        anchor_mask | candidate_mask,
                        args.boundary_kernel,
                    )
                    abs_diff = (anchor_prob - candidate_prob).abs()

                    all_value = masked_mean(abs_diff, pair_region)
                    interior_value = masked_mean(abs_diff, interior)
                    boundary_value = masked_mean(abs_diff, boundary)

                    if all_value is None:
                        continue

                    mask_iou_value = binary_iou(anchor_mask, candidate_mask)
                    box_iou_value = float(ious[candidate_idx, anchor_idx].item())

                    ratio_value = None
                    if interior_value is not None and boundary_value is not None:
                        ratio_value = float(
                            boundary_value / max(interior_value, EPS)
                        )

                    o2o_rel = float(anchor_reliability[anchor_idx].item())
                    o2m_rel = float(candidate_reliability[candidate_idx].item())
                    o2o_stab = float(anchor_stability[anchor_idx].item())
                    o2m_stab = float(candidate_stability[candidate_idx].item())

                    row = {
                        "image": path_str,
                        "anchor_index": int(anchor_idx),
                        "witness_rank": int(rank),
                        "raw_witness_count_for_anchor": raw_count,
                        "o2o_conf": float(anchors[anchor_idx, 4].item()),
                        "o2m_conf": float(candidates[candidate_idx, 4].item()),
                        "o2o_stability": o2o_stab,
                        "o2m_stability": o2m_stab,
                        "o2o_reliability": o2o_rel,
                        "o2m_reliability": o2m_rel,
                        "box_iou": box_iou_value,
                        "mask_iou": mask_iou_value,
                        "mad_all": all_value,
                        "mad_interior": interior_value,
                        "mad_boundary": boundary_value,
                        "boundary_interior_ratio": ratio_value,
                    }
                    pair_rows.append(row)

                    box_ious.append(box_iou_value)
                    mask_ious.append(mask_iou_value)
                    mad_all.append(all_value)
                    if interior_value is not None:
                        mad_interior.append(interior_value)
                    if boundary_value is not None:
                        mad_boundary.append(boundary_value)
                    if ratio_value is not None:
                        boundary_interior_ratio.append(ratio_value)

                    o2o_rels.append(o2o_rel)
                    o2m_rels.append(o2m_rel)
                    o2o_stabs.append(o2o_stab)
                    o2m_stabs.append(o2m_stab)

                    if rank == 0:
                        best_box_ious.append(box_iou_value)
                        best_mask_ious.append(mask_iou_value)
                        best_mad_all.append(all_value)
                        if interior_value is not None:
                            best_mad_interior.append(interior_value)
                        if boundary_value is not None:
                            best_mad_boundary.append(boundary_value)
                        if ratio_value is not None:
                            best_boundary_interior_ratio.append(ratio_value)

        if batch_i == 1 or batch_i % args.print_freq == 0 or batch_i == len(loader):
            print(
                f"[{batch_i:03d}/{len(loader):03d}] "
                f"O2O={total_o2o} O2M={total_o2m} "
                f"matched_anchors={anchors_with_match} pairs={len(pair_rows)}",
                flush=True,
            )

    matched_anchor_rate = float(anchors_with_match) / max(float(total_o2o), 1.0)

    summary = {
        "source_free_audit": True,
        "target_labels_used": False,
        "target_gt_masks_used": False,
        "weights": str(Path(args.weights).resolve()),
        "target_images": str(Path(args.target_images).resolve()),
        "num_target_images": len(target_images),
        "settings": {
            "imgsz": args.imgsz,
            "tau_o2o": args.tau_o2o,
            "tau_o2m": args.tau_o2m,
            "tau_match": args.tau_match,
            "mask_thr": args.mask_thr,
            "stability_low": args.stability_low,
            "stability_high": args.stability_high,
            "min_mask_pixels": args.min_mask_pixels,
            "boundary_kernel": args.boundary_kernel,
            "max_witnesses_per_anchor": args.max_witnesses_per_anchor,
        },
        "counts": {
            "o2o_anchors": total_o2o,
            "o2m_valid_candidates": total_o2m,
            "anchors_with_match": anchors_with_match,
            "matched_anchor_rate": matched_anchor_rate,
            "raw_matched_o2m": raw_matched_o2m,
            "retained_pair_rows": len(pair_rows),
        },
        "match_count_per_anchor": quantiles(match_count_per_anchor),
        "all_retained_pairs": {
            "box_iou": quantiles(box_ious),
            "mask_iou": quantiles(mask_ious),
            "mad_all": quantiles(mad_all),
            "mad_interior": quantiles(mad_interior),
            "mad_boundary": quantiles(mad_boundary),
            "boundary_interior_ratio": quantiles(boundary_interior_ratio),
            "o2o_reliability": quantiles(o2o_rels),
            "o2m_reliability": quantiles(o2m_rels),
            "o2o_stability": quantiles(o2o_stabs),
            "o2m_stability": quantiles(o2m_stabs),
        },
        "best_witness_per_anchor": {
            "box_iou": quantiles(best_box_ious),
            "mask_iou": quantiles(best_mask_ious),
            "mad_all": quantiles(best_mad_all),
            "mad_interior": quantiles(best_mad_interior),
            "mad_boundary": quantiles(best_mad_boundary),
            "boundary_interior_ratio": quantiles(best_boundary_interior_ratio),
        },
    }

    out_json = Path(args.out_json).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_fields = [
        "image",
        "anchor_index",
        "witness_rank",
        "raw_witness_count_for_anchor",
        "o2o_conf",
        "o2m_conf",
        "o2o_stability",
        "o2m_stability",
        "o2o_reliability",
        "o2m_reliability",
        "box_iou",
        "mask_iou",
        "mad_all",
        "mad_interior",
        "mad_boundary",
        "boundary_interior_ratio",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(pair_rows)

    print()
    print("=" * 78)
    print("CC-DHF LABEL-FREE O2O/O2M AUDIT")
    print("=" * 78)
    print("target images       :", len(target_images))
    print("target labels used  : NO")
    print("target GT masks used: NO")
    print("O2O anchors         :", total_o2o)
    print("valid O2M candidates:", total_o2m)
    print("anchors with match  :", anchors_with_match)
    print("matched anchor rate :", f"{100.0 * matched_anchor_rate:.2f}%")
    print("raw matched O2M     :", raw_matched_o2m)
    print("retained pair rows  :", len(pair_rows))

    print_quantiles("O2M witness count per O2O anchor", match_count_per_anchor)

    print()
    print("-" * 78)
    print("ALL RETAINED MATCHED PAIRS")
    print("-" * 78)
    print_quantiles("Box IoU", box_ious)
    print_quantiles("Mask IoU", mask_ious)
    print_quantiles("Mean |P_o2o - P_o2m| over pair region", mad_all)
    print_quantiles("Interior disagreement", mad_interior)
    print_quantiles("Boundary disagreement", mad_boundary)
    print_quantiles("Boundary / interior disagreement ratio", boundary_interior_ratio)

    print()
    print("-" * 78)
    print("BEST O2M WITNESS PER MATCHED O2O ANCHOR")
    print("-" * 78)
    print_quantiles("Best-witness Box IoU", best_box_ious)
    print_quantiles("Best-witness Mask IoU", best_mask_ious)
    print_quantiles("Best-witness mean |P_o2o - P_o2m|", best_mad_all)
    print_quantiles("Best-witness interior disagreement", best_mad_interior)
    print_quantiles("Best-witness boundary disagreement", best_mad_boundary)
    print_quantiles("Best-witness boundary/interior ratio", best_boundary_interior_ratio)

    print()
    print("Saved JSON:", out_json)
    print("Saved CSV :", out_csv)
    print()
    print(
        "[PASS] Label-free CC-DHF audit completed. "
        "Do not modify training architecture until these statistics are interpreted."
    )


if __name__ == "__main__":
    main()