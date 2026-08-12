#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO  # noqa: E402

from smoke_stage2_mt_seg import (  # noqa: E402
    TargetMTDataset,
    collate,
    list_images,
    resolve_device,
    seed_everything,
)

from smoke_stage2_mask_dhf_seg import (  # noqa: E402
    generate_dhf_pseudo_masks,
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
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


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
        "--mask-rel-thr",
        type=float,
        default=0.65,
    )

    ap.add_argument(
        "--mask-no",
        type=float,
        default=0.20,
    )

    ap.add_argument(
        "--min-mask-pixels",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--device",
        default="0",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=29,
    )

    ap.add_argument(
        "--out",
        default=(
            "runs/seg/dense_sfseg/"
            "mask_dhf_audit.json"
        ),
    )

    args = ap.parse_args()

    seed_everything(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    image_root = Path(
        args.target_images
    ).resolve()

    images = list_images(
        image_root
    )

    dataset = TargetMTDataset(
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
        drop_last=False,
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

    totals = {
        "anchors": 0,
        "candidates": 0,
        "candidate_valid_mask": 0,
        "candidate_reliable": 0,
        "rejected_reliability": 0,
        "rejected_box_overlap": 0,
        "rejected_mask_overlap": 0,
        "extras": 0,
        "fused_after_mask_filter": 0,
    }

    mask_conf_values = []
    reliability_values = []

    images_with_pseudo = 0
    images_without_pseudo = 0

    print("=" * 72)
    print("MASK-DHF FULL TARGET AUDIT")
    print("=" * 72)

    print("images        :", len(images))
    print("labels loaded : NO")
    print("GT masks used : NO")

    print("tau_o2o       :", args.tau_o2o)
    print("tau_o2m       :", args.tau_o2m)
    print("tau_no        :", args.tau_no)
    print("tau_dup       :", args.tau_dup)
    print("mask_thr      :", args.mask_thr)
    print("mask_rel_thr  :", args.mask_rel_thr)
    print("mask_no       :", args.mask_no)

    with torch.inference_mode():
        for batch_idx, (
            weak,
            _strong,
            _paths,
        ) in enumerate(
            loader,
            start=1,
        ):
            weak = weak.to(
                device,
                non_blocking=True,
            )

            (
                labels,
                masks,
                batch_stats,
            ) = generate_dhf_pseudo_masks(
                teacher,
                weak,
                tau_o2o=args.tau_o2o,
                tau_o2m=args.tau_o2m,
                tau_no=args.tau_no,
                tau_dup=args.tau_dup,
                mask_threshold=args.mask_thr,
                min_mask_pixels=args.min_mask_pixels,
                mask_rel_threshold=args.mask_rel_thr,
                mask_no=args.mask_no,
            )

            for key in totals:
                totals[key] += int(
                    batch_stats.get(
                        key,
                        0,
                    )
                )

            mask_conf_values.extend(
                batch_stats.get(
                    "mask_conf_values",
                    [],
                )
            )

            reliability_values.extend(
                batch_stats.get(
                    "reliability_values",
                    [],
                )
            )

            for rows in labels:
                if rows.shape[0] > 0:
                    images_with_pseudo += 1
                else:
                    images_without_pseudo += 1

            if (
                batch_idx == 1
                or batch_idx % 20 == 0
                or batch_idx == len(loader)
            ):
                print(
                    f"[Batch {batch_idx:03d}/{len(loader):03d}] "
                    f"anchors={totals['anchors']} "
                    f"candidates={totals['candidates']} "
                    f"reliable={totals['candidate_reliable']} "
                    f"extra={totals['extras']} "
                    f"fused={totals['fused_after_mask_filter']}"
                )

    mask_conf_stats = stats(
        mask_conf_values
    )

    reliability_stats = stats(
        reliability_values
    )

    result = {
        "weights": str(
            Path(args.weights).resolve()
        ),
        "target_images": str(
            image_root
        ),
        "images": len(images),

        "target_labels_used": False,
        "target_gt_masks_used": False,

        "thresholds": {
            "tau_o2o": args.tau_o2o,
            "tau_o2m": args.tau_o2m,
            "tau_no": args.tau_no,
            "tau_dup": args.tau_dup,
            "mask_thr": args.mask_thr,
            "mask_rel_thr": args.mask_rel_thr,
            "mask_no": args.mask_no,
            "min_mask_pixels":
                args.min_mask_pixels,
        },

        **totals,

        "images_with_pseudo":
            images_with_pseudo,

        "images_without_pseudo":
            images_without_pseudo,

        "avg_fused_per_image":
            totals[
                "fused_after_mask_filter"
            ] / len(images),

        "mask_conf":
            mask_conf_stats,

        "mask_reliability":
            reliability_stats,
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
    print("MASK-DHF AUDIT RESULTS")
    print("=" * 72)

    for key in totals:
        print(
            f"{key:28s}: "
            f"{totals[key]}"
        )

    print(
        f"{'images_with_pseudo':28s}: "
        f"{images_with_pseudo}"
    )

    print(
        f"{'images_without_pseudo':28s}: "
        f"{images_without_pseudo}"
    )

    print()
    print(
        "mask confidence   :",
        mask_conf_stats,
    )

    print(
        "mask reliability  :",
        reliability_stats,
    )

    print()
    print(
        "avg fused/image   :",
        result[
            "avg_fused_per_image"
        ],
    )

    if (
        totals[
            "fused_after_mask_filter"
        ]
        == 0
    ):
        raise RuntimeError(
            "Mask-DHF removed every pseudo label"
        )

    print()
    print("[PASS] Mask-DHF audit complete")
    print("saved:", out)


if __name__ == "__main__":
    main()
