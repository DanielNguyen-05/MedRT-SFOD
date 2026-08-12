#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics.utils.metrics import box_iou  # noqa: E402

from smoke_stage2_mt_seg import (  # noqa: E402
    TargetMTDataset,
    build_pseudo_batch,
    collate,
    list_images,
    masks_from_coefficients,
    resolve_device,
    seed_everything,
    setup_teacher_student,
    update_teacher_ema,
)


# ================================================================
# DHF preserving segmentation mask coefficients
# ================================================================

def classwise_nms_full(
    rows: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """
    rows:
        [N, 6 + nm]

    NMS operates on box/conf/class but preserves
    the corresponding mask coefficients.
    """

    if rows.numel() == 0:
        return rows

    kept = []

    classes = rows[:, 5].long().unique(
        sorted=True
    )

    for cls in classes:
        idx = torch.where(
            rows[:, 5].long() == cls
        )[0]

        keep = torchvision.ops.nms(
            rows[idx, :4],
            rows[idx, 4],
            iou_threshold,
        )

        kept.append(
            rows[idx[keep]]
        )

    if not kept:
        return rows.new_zeros(
            (0, rows.shape[1])
        )

    return torch.cat(
        kept,
        dim=0,
    )


def dual_head_fusion_seg(
    one2one: torch.Tensor,
    one2many: torch.Tensor,
    tau_o2o: float,
    tau_o2m: float,
    tau_no: float,
    tau_dup: float,
):
    """
    Detection-style DHF extended so every kept row retains
    its mask coefficients.

    O2O:
        high-confidence anchors

    O2M:
        supplementary candidates

    An O2M candidate is added only when it has low overlap
    with every O2O anchor, followed by classwise NMS.
    """

    anchors = one2one[
        one2one[:, 4] >= tau_o2o
    ]

    candidates = one2many[
        one2many[:, 4] >= tau_o2m
    ]

    if candidates.numel() == 0:
        extras = candidates

    elif anchors.numel() == 0:
        extras = classwise_nms_full(
            candidates,
            tau_dup,
        )

    else:
        max_iou = box_iou(
            candidates[:, :4],
            anchors[:, :4],
        ).max(
            dim=1
        ).values

        extras = candidates[
            max_iou <= tau_no
        ]

        extras = classwise_nms_full(
            extras,
            tau_dup,
        )

    if anchors.numel() and extras.numel():
        fused = torch.cat(
            [anchors, extras],
            dim=0,
        )

    elif anchors.numel():
        fused = anchors

    else:
        fused = extras

    if fused.numel():
        fused = fused[
            torch.argsort(
                fused[:, 4],
                descending=True,
            )
        ]

    stats = {
        "anchors": int(
            anchors.shape[0]
        ),
        "candidates": int(
            candidates.shape[0]
        ),
        "extras": int(
            extras.shape[0]
        ),
        "fused_before_mask_filter": int(
            fused.shape[0]
        ),
    }

    return fused, stats


# ================================================================
# Teacher DHF pseudo masks
# ================================================================

@torch.no_grad()
def generate_dhf_pseudo_masks(
    teacher: nn.Module,
    weak_imgs: torch.Tensor,
    tau_o2o: float,
    tau_o2m: float,
    tau_no: float,
    tau_dup: float,
    mask_threshold: float,
    min_mask_pixels: int,
):
    teacher.eval()

    outputs = teacher(
        weak_imgs,
        augment=False,
        visualize=False,
    )

    if not (
        isinstance(outputs, tuple)
        and len(outputs) == 2
    ):
        raise RuntimeError(
            "Unexpected teacher output"
        )

    first, branches = outputs

    if not (
        isinstance(first, tuple)
        and len(first) == 2
    ):
        raise RuntimeError(
            "Expected ((O2O, proto), branches)"
        )

    final_o2o, proto = first

    if not isinstance(
        branches,
        dict,
    ):
        raise RuntimeError(
            "Teacher branches missing"
        )

    if "one2many" not in branches:
        raise RuntimeError(
            "Teacher O2M branch missing"
        )

    head = teacher.model[-1]

    raw_o2m = branches[
        "one2many"
    ]

    decoded_o2m = head._inference(
        raw_o2m
    ).permute(
        0,
        2,
        1,
    )

    final_o2m = head.postprocess(
        decoded_o2m
    )

    h = int(
        weak_imgs.shape[2]
    )

    w = int(
        weak_imgs.shape[3]
    )

    all_labels = []
    all_masks = []

    total_stats = {
        "anchors": 0,
        "candidates": 0,
        "extras": 0,
        "fused_before_mask_filter": 0,
        "fused_after_mask_filter": 0,
    }

    for i in range(
        weak_imgs.shape[0]
    ):
        o2o = final_o2o[i]

        o2m = final_o2m[i]

        # Remove zero-score padded top-k rows.
        o2o = o2o[
            o2o[:, 4] > 0
        ]

        o2m = o2m[
            o2m[:, 4] > 0
        ]

        fused, stats = (
            dual_head_fusion_seg(
                o2o,
                o2m,
                tau_o2o=tau_o2o,
                tau_o2m=tau_o2m,
                tau_no=tau_no,
                tau_dup=tau_dup,
            )
        )

        for key in (
            "anchors",
            "candidates",
            "extras",
            "fused_before_mask_filter",
        ):
            total_stats[key] += stats[key]

        if fused.numel() == 0:
            all_labels.append(
                fused
            )

            all_masks.append(
                proto.new_zeros(
                    (
                        0,
                        proto.shape[-2],
                        proto.shape[-1],
                    )
                )
            )

            continue

        masks = masks_from_coefficients(
            fused,
            proto[i],
            input_h=h,
            input_w=w,
            mask_threshold=mask_threshold,
        )

        areas = masks.sum(
            dim=(1, 2)
        )

        keep = (
            areas
            >= min_mask_pixels
        )

        fused = fused[keep]
        masks = masks[keep]

        total_stats[
            "fused_after_mask_filter"
        ] += int(
            fused.shape[0]
        )

        all_labels.append(
            fused
        )

        all_masks.append(
            masks
        )

    return (
        all_labels,
        all_masks,
        total_stats,
    )


# ================================================================
# Main smoke
# ================================================================

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
        default=4,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--max-batches",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
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
        "--min-mask-pixels",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
    )

    ap.add_argument(
        "--ema",
        type=float,
        default=0.999,
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
        "--out-dir",
        default=(
            "runs/seg/dense_sfseg/"
            "smoke_dhf_seg"
        ),
    )

    args = ap.parse_args()

    seed_everything(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    images = list_images(
        Path(
            args.target_images
        ).resolve()
    )

    dataset = TargetMTDataset(
        images,
        args.imgsz,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
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
        epochs=1,
    )

    optimizer = optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
    )

    print("=" * 72)
    print("STAGE-2 DHF SEGMENTATION SMOKE")
    print("=" * 72)

    print(
        "target images :",
        len(images),
    )

    print(
        "labels loaded : NO"
    )

    print(
        "GT masks used : NO"
    )

    print(
        "pseudo mode   : O2O + O2M DHF + pseudo masks"
    )

    print(
        "Mask-DHF reliability : OFF"
    )

    print(
        "MARD          : OFF"
    )

    print(
        "RASP          : OFF"
    )

    print(
        "tau_o2o       :",
        args.tau_o2o,
    )

    print(
        "tau_o2m       :",
        args.tau_o2m,
    )

    print(
        "tau_no        :",
        args.tau_no,
    )

    print(
        "tau_dup       :",
        args.tau_dup,
    )

    print(
        "max batches   :",
        args.max_batches,
    )

    successful = 0
    skipped = 0

    total_pseudo = 0
    total_anchor = 0
    total_candidate = 0
    total_extra = 0

    loss_values = []
    seg_values = []

    start = time.time()

    student.train()

    for batch_idx, (
        weak,
        strong,
        paths,
    ) in enumerate(
        loader,
        start=1,
    ):
        if (
            args.max_batches > 0
            and batch_idx
            > args.max_batches
        ):
            break

        weak = weak.to(
            device,
            non_blocking=True,
        )

        strong = strong.to(
            device,
            non_blocking=True,
        )

        (
            pseudo_labels,
            pseudo_masks,
            pseudo_stats,
        ) = generate_dhf_pseudo_masks(
            teacher,
            weak,
            tau_o2o=args.tau_o2o,
            tau_o2m=args.tau_o2m,
            tau_no=args.tau_no,
            tau_dup=args.tau_dup,
            mask_threshold=args.mask_thr,
            min_mask_pixels=(
                args.min_mask_pixels
            ),
        )

        valid_indices = [
            i
            for i, x
            in enumerate(
                pseudo_labels
            )
            if x.shape[0] > 0
        ]

        total_anchor += (
            pseudo_stats["anchors"]
        )

        total_candidate += (
            pseudo_stats["candidates"]
        )

        total_extra += (
            pseudo_stats["extras"]
        )

        if not valid_indices:
            skipped += 1

            print(
                f"[Batch {batch_idx:02d}] "
                "SKIP no pseudo masks "
                f"anchor={pseudo_stats['anchors']} "
                f"extra={pseudo_stats['extras']}"
            )

            continue

        strong_valid = strong[
            valid_indices
        ]

        labels_valid = [
            pseudo_labels[i]
            for i in valid_indices
        ]

        masks_valid = [
            pseudo_masks[i]
            for i in valid_indices
        ]

        n_pseudo = sum(
            x.shape[0]
            for x in labels_valid
        )

        total_pseudo += n_pseudo

        student.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        student_outputs = student(
            strong_valid
        )

        if not (
            isinstance(
                student_outputs,
                dict,
            )
            and "one2one"
            in student_outputs
            and "one2many"
            in student_outputs
        ):
            raise RuntimeError(
                "Invalid Student dual-head output"
            )

        batch = build_pseudo_batch(
            labels_valid,
            masks_valid,
            strong_valid.shape,
        )

        if batch is None:
            skipped += 1
            continue

        total_loss_vec, loss_items = (
            criterion(
                student_outputs,
                batch,
            )
        )

        if len(loss_items) < 5:
            raise RuntimeError(
                f"Expected five segmentation "
                f"loss components, got "
                f"{len(loss_items)}"
            )

        total_loss = (
            total_loss_vec.sum()
        )

        if not torch.isfinite(
            total_loss
        ):
            raise RuntimeError(
                f"Non-finite loss: "
                f"{total_loss}"
            )

        total_loss.backward()

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                args.grad_clip,
            )
        )

        if not torch.isfinite(
            torch.as_tensor(
                grad_norm
            )
        ):
            raise RuntimeError(
                f"Non-finite grad norm: "
                f"{grad_norm}"
            )

        optimizer.step()

        box_loss = float(
            loss_items[0]
            .detach()
            .cpu()
        )

        seg_loss = float(
            loss_items[1]
            .detach()
            .cpu()
        )

        cls_loss = float(
            loss_items[2]
            .detach()
            .cpu()
        )

        dfl_loss = float(
            loss_items[3]
            .detach()
            .cpu()
        )

        semseg_loss = float(
            loss_items[4]
            .detach()
            .cpu()
        )

        loss_float = float(
            total_loss
            .detach()
            .cpu()
        )

        loss_values.append(
            loss_float
        )

        seg_values.append(
            seg_loss
        )

        successful += 1

        mask_pixels = torch.cat(
            masks_valid,
            dim=0,
        ).sum(
            dim=(1, 2)
        )

        print(
            f"[Batch {batch_idx:02d}] "
            f"valid_img={len(valid_indices)}/{weak.shape[0]} "
            f"anchor={pseudo_stats['anchors']} "
            f"cand={pseudo_stats['candidates']} "
            f"extra={pseudo_stats['extras']} "
            f"pseudo={n_pseudo} "
            f"loss={loss_float:.4f} "
            f"box={box_loss:.4f} "
            f"seg={seg_loss:.4f} "
            f"cls={cls_loss:.4f} "
            f"dfl={dfl_loss:.4f} "
            f"semseg={semseg_loss:.4f} "
            f"grad_preclip={float(grad_norm):.4f} "
            f"mask_px_mean="
            f"{float(mask_pixels.float().mean()):.1f}"
        )

    if successful == 0:
        raise RuntimeError(
            "No successful optimization batches"
        )

    # Smoke represents one partial epoch:
    # update Teacher once, consistent with epoch-level EMA.
    update_teacher_ema(
        teacher,
        student,
        args.ema,
    )

    out_dir = Path(
        args.out_dir
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    smoke_ckpt = (
        out_dir
        / "student_DHF_SMOKE.pt"
    )

    student.eval()

    student_wrapper.model = student

    student_wrapper.save(
        str(smoke_ckpt)
    )

    meta = {
        "source_free": True,
        "target_labels_used": False,
        "target_gt_masks_used": False,

        "pseudo_mode":
            "o2o_o2m_box_dhf_with_pseudo_masks",

        "mask_dhf_reliability": False,
        "mard": False,
        "rasp": False,

        "tau_o2o": args.tau_o2o,
        "tau_o2m": args.tau_o2m,
        "tau_no": args.tau_no,
        "tau_dup": args.tau_dup,

        "successful_batches":
            successful,

        "skipped_batches":
            skipped,

        "anchors":
            total_anchor,

        "o2m_candidates":
            total_candidate,

        "dhf_extras":
            total_extra,

        "pseudo_instances":
            total_pseudo,

        "mean_total_loss":
            float(
                np.mean(
                    loss_values
                )
            ),

        "mean_seg_loss":
            float(
                np.mean(
                    seg_values
                )
            ),

        "elapsed_seconds":
            time.time() - start,

        "checkpoint":
            str(smoke_ckpt),
    }

    (
        out_dir
        / "smoke_metadata.json"
    ).write_text(
        json.dumps(
            meta,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("[PASS] DHF SEGMENTATION TRAINING PATH WORKS")
    print("=" * 72)

    print(
        "successful batches :",
        successful,
    )

    print(
        "skipped batches    :",
        skipped,
    )

    print(
        "O2O anchors        :",
        total_anchor,
    )

    print(
        "O2M candidates     :",
        total_candidate,
    )

    print(
        "DHF extras         :",
        total_extra,
    )

    print(
        "pseudo instances   :",
        total_pseudo,
    )

    print(
        "mean total loss    :",
        np.mean(loss_values),
    )

    print(
        "mean seg loss      :",
        np.mean(seg_values),
    )

    print()
    print(
        "SMOKE checkpoint only:",
        smoke_ckpt,
    )

    print(
        "DO NOT use this checkpoint "
        "for final reporting."
    )


if __name__ == "__main__":
    main()
