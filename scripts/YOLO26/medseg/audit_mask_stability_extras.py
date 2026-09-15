#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402

from audit_stage2_seg import (  # noqa: E402
    TargetAuditDataset,
    collate,
    list_images,
    resolve_device,
)

from smoke_stage2_mask_dhf_seg import (  # noqa: E402
    mask_probs_from_coefficients,
)


def stats(values):
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "p05": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def classwise_nms_indices(
    rows: torch.Tensor,
    iou_threshold: float,
):
    if rows.numel() == 0:
        return torch.zeros(
            0,
            dtype=torch.long,
            device=rows.device,
        )

    kept = []

    for cls in rows[:, 5].long().unique(
        sorted=True
    ):
        idx = torch.where(
            rows[:, 5].long() == cls
        )[0]

        local_keep = torchvision.ops.nms(
            rows[idx, :4],
            rows[idx, 4],
            iou_threshold,
        )

        kept.append(
            idx[local_keep]
        )

    if not kept:
        return torch.zeros(
            0,
            dtype=torch.long,
            device=rows.device,
        )

    return torch.cat(
        kept,
        dim=0,
    )


def mask_stability(
    probs: torch.Tensor,
    low_thr: float,
    high_thr: float,
):
    """
    IoU between mask thresholded at low/high levels.

    High mask is normally a subset of low mask.
    """

    low = (
        probs >= low_thr
    )

    high = (
        probs >= high_thr
    )

    inter = (
        low & high
    ).flatten(1).sum(1).float()

    union = (
        low | high
    ).flatten(1).sum(1).float()

    return (
        inter
        / union.clamp_min(1.0)
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--weights",
        required=True,
    )

    ap.add_argument(
        "--target-images",
        required=True,
    )

    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--device",
        default="0",
    )

    ap.add_argument(
        "--tau-o2o",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--tau-o2m",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--tau-no",
        type=float,
        default=0.2,
    )

    ap.add_argument(
        "--tau-dup",
        type=float,
        default=0.7,
    )

    ap.add_argument(
        "--mask-thr",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--stability-low",
        type=float,
        default=0.40,
    )

    ap.add_argument(
        "--stability-high",
        type=float,
        default=0.60,
    )

    ap.add_argument(
        "--min-mask-pixels",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--out",
        default=(
            "runs/seg/dense_sfseg/"
            "mask_stability_extras_audit.json"
        ),
    )

    args = ap.parse_args()

    device = resolve_device(
        args.device
    )

    image_root = Path(
        args.target_images
    ).resolve()

    images = list_images(
        image_root
    )

    dataset = TargetAuditDataset(
        images,
        args.imgsz,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
        collate_fn=collate,
    )

    wrapper = YOLO(
        args.weights
    )

    teacher = (
        wrapper.model
        .to(device)
        .float()
    )

    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad_(False)

    head = teacher.model[-1]

    total_anchors = 0
    total_candidates = 0
    total_valid_masks = 0
    total_box_novel = 0
    total_extras = 0

    box_conf_values = []
    stability_values = []
    reliability_values = []
    fg_conf_values = []

    # Keep reliability separated per image
    # so threshold simulation can measure coverage.
    extras_per_image = []

    print("=" * 72)
    print("MASK STABILITY AUDIT — BOX-DHF EXTRAS")
    print("=" * 72)

    print("images        :", len(images))
    print("labels loaded : NO")
    print("GT masks used : NO")
    print()
    print("tau_o2o       :", args.tau_o2o)
    print("tau_o2m       :", args.tau_o2m)
    print("tau_no        :", args.tau_no)
    print("tau_dup       :", args.tau_dup)
    print("mask_thr      :", args.mask_thr)
    print(
        "stability     :",
        args.stability_low,
        "vs",
        args.stability_high,
    )

    with torch.inference_mode():

        for batch_idx, (
            ims,
            _paths,
        ) in enumerate(
            loader,
            start=1,
        ):
            ims = ims.to(
                device,
                non_blocking=True,
            )

            outputs = teacher(
                ims,
                augment=False,
                visualize=False,
            )

            first, branches = outputs

            final_o2o, proto = first

            decoded_o2m = head._inference(
                branches["one2many"]
            ).permute(
                0,
                2,
                1,
            )

            final_o2m = head.postprocess(
                decoded_o2m
            )

            h = int(ims.shape[2])
            w = int(ims.shape[3])

            for i in range(
                ims.shape[0]
            ):
                anchors = final_o2o[i]

                candidates = final_o2m[i]

                anchors = anchors[
                    anchors[:, 4]
                    >= args.tau_o2o
                ]

                candidates = candidates[
                    candidates[:, 4]
                    >= args.tau_o2m
                ]

                total_anchors += int(
                    anchors.shape[0]
                )

                total_candidates += int(
                    candidates.shape[0]
                )

                if candidates.numel() == 0:
                    extras_per_image.append([])
                    continue

                probs, _ = (
                    mask_probs_from_coefficients(
                        candidates,
                        proto[i],
                        input_h=h,
                        input_w=w,
                    )
                )

                base_masks = (
                    probs
                    >= args.mask_thr
                )

                areas = base_masks.sum(
                    dim=(1, 2)
                )

                valid = (
                    areas
                    >= args.min_mask_pixels
                )

                candidates = candidates[
                    valid
                ]

                probs = probs[
                    valid
                ]

                total_valid_masks += int(
                    candidates.shape[0]
                )

                if candidates.numel() == 0:
                    extras_per_image.append([])
                    continue

                # -----------------------------------------------
                # Standard Box-DHF novelty gate FIRST
                # -----------------------------------------------

                if anchors.numel():
                    max_iou = box_iou(
                        candidates[:, :4],
                        anchors[:, :4],
                    ).max(
                        dim=1
                    ).values

                    novel = (
                        max_iou
                        <= args.tau_no
                    )

                    candidates = candidates[
                        novel
                    ]

                    probs = probs[
                        novel
                    ]

                total_box_novel += int(
                    candidates.shape[0]
                )

                if candidates.numel() == 0:
                    extras_per_image.append([])
                    continue

                # -----------------------------------------------
                # Standard DHF NMS
                # -----------------------------------------------

                keep = classwise_nms_indices(
                    candidates,
                    args.tau_dup,
                )

                candidates = candidates[
                    keep
                ]

                probs = probs[
                    keep
                ]

                total_extras += int(
                    candidates.shape[0]
                )

                if candidates.numel() == 0:
                    extras_per_image.append([])
                    continue

                # -----------------------------------------------
                # TRUE mask-aware quality signal
                # -----------------------------------------------

                stability = mask_stability(
                    probs,
                    low_thr=args.stability_low,
                    high_thr=args.stability_high,
                )

                base_mask = (
                    probs >= args.mask_thr
                ).float()

                fg_conf = (
                    (probs * base_mask)
                    .sum(dim=(1, 2))
                    /
                    base_mask
                    .sum(dim=(1, 2))
                    .clamp_min(1.0)
                )

                box_conf = candidates[:, 4]

                reliability = torch.sqrt(
                    (
                        box_conf
                        * stability
                    ).clamp_min(0.0)
                )

                box_conf_values.extend(
                    box_conf.cpu().tolist()
                )

                fg_conf_values.extend(
                    fg_conf.cpu().tolist()
                )

                stability_values.extend(
                    stability.cpu().tolist()
                )

                reliability_values.extend(
                    reliability.cpu().tolist()
                )

                extras_per_image.append(
                    reliability.cpu().tolist()
                )

            if (
                batch_idx == 1
                or batch_idx % 20 == 0
                or batch_idx == len(loader)
            ):
                print(
                    f"[Batch {batch_idx:03d}/{len(loader):03d}] "
                    f"anchors={total_anchors} "
                    f"candidates={total_candidates} "
                    f"novel={total_box_novel} "
                    f"extras={total_extras}"
                )

    stability_stats = stats(
        stability_values
    )

    reliability_stats = stats(
        reliability_values
    )

    box_conf_stats = stats(
        box_conf_values
    )

    fg_conf_stats = stats(
        fg_conf_values
    )

    # ------------------------------------------------------------
    # Label-free threshold simulations
    # ------------------------------------------------------------

    threshold_candidates = [
        0.70,
        0.75,
        0.80,
        0.85,
    ]

    if reliability_values:
        q25 = float(
            np.percentile(
                np.asarray(
                    reliability_values,
                    dtype=np.float64,
                ),
                25,
            )
        )

        threshold_candidates.append(
            q25
        )

    threshold_candidates = sorted(
        set(
            round(x, 6)
            for x in threshold_candidates
        )
    )

    simulations = {}

    for threshold in threshold_candidates:
        kept = 0
        images_extra = 0

        for values in extras_per_image:
            n = sum(
                float(v) >= threshold
                for v in values
            )

            kept += n

            if n > 0:
                images_extra += 1

        simulations[
            f"{threshold:.6f}"
        ] = {
            "extras_kept": int(kept),

            "extras_removed": int(
                total_extras - kept
            ),

            "keep_ratio": float(
                kept
                / max(
                    total_extras,
                    1,
                )
            ),

            "images_with_extra":
                int(images_extra),
        }

    result = {
        "images": len(images),

        "target_labels_used": False,
        "target_gt_masks_used": False,

        "anchors": total_anchors,
        "candidates": total_candidates,
        "valid_mask_candidates":
            total_valid_masks,

        "box_novel_before_nms":
            total_box_novel,

        "box_dhf_extras":
            total_extras,

        "box_conf":
            box_conf_stats,

        "foreground_mask_conf":
            fg_conf_stats,

        "mask_stability":
            stability_stats,

        "mask_reliability":
            reliability_stats,

        "threshold_simulation":
            simulations,
    }

    out = Path(
        args.out
    ).resolve()

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("EXTRA MASK-STABILITY AUDIT RESULTS")
    print("=" * 72)

    print(
        "anchors                :",
        total_anchors,
    )

    print(
        "O2M candidates         :",
        total_candidates,
    )

    print(
        "valid-mask candidates  :",
        total_valid_masks,
    )

    print(
        "box-novel before NMS   :",
        total_box_novel,
    )

    print(
        "Box-DHF extras         :",
        total_extras,
    )

    print()
    print(
        "box confidence         :",
        box_conf_stats,
    )

    print(
        "foreground mask conf   :",
        fg_conf_stats,
    )

    print(
        "mask stability         :",
        stability_stats,
    )

    print(
        "new mask reliability   :",
        reliability_stats,
    )

    print()
    print("Threshold simulation:")

    for threshold, row in (
        simulations.items()
    ):
        print(
            f"  r >= {threshold}: "
            f"keep={row['extras_kept']} "
            f"remove={row['extras_removed']} "
            f"ratio={row['keep_ratio']:.3f} "
            f"images_extra={row['images_with_extra']}"
        )

    print()
    print(
        "[PASS] extra mask-stability audit complete"
    )

    print(
        "saved:",
        out,
    )


if __name__ == "__main__":
    main()
