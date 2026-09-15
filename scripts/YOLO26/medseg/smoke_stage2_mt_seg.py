#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.ops import xyxy2xywh  # noqa: E402


IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(s: str):
    if s.lower() == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    if s.isdigit():
        return torch.device(f"cuda:{s}")

    return torch.device(s)


def make_divisible(x: int, divisor: int = 32):
    return int(
        math.ceil(float(x) / divisor) * divisor
    )


def list_images(root: Path):
    images = sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMG_EXTS
        ],
        key=lambda p: (
            int(p.stem)
            if p.stem.isdigit()
            else p.stem
        ),
    )

    if not images:
        raise RuntimeError(
            f"No images found: {root}"
        )

    return images


# ================================================================
# Target image-only dataset
# ================================================================

class TargetMTDataset(Dataset):

    def __init__(
        self,
        images: list[Path],
        imgsz: int,
    ):
        self.images = images
        self.imgsz = imgsz

    def __len__(self):
        return len(self.images)

    @staticmethod
    def strong_photo(im: np.ndarray):
        """
        Strong PHOTOMETRIC augmentation only.

        Geometry remains identical to weak image.
        This intentionally simplifies the first segmentation smoke test,
        so pseudo masks do not need geometric warping yet.
        """

        out = im.copy()

        if random.random() < 0.8:
            hsv = cv2.cvtColor(
                out,
                cv2.COLOR_RGB2HSV,
            ).astype(np.float32)

            hsv[..., 0] = np.clip(
                hsv[..., 0]
                + random.uniform(-0.10, 0.10) * 180,
                0,
                179,
            )

            hsv[..., 1] = np.clip(
                hsv[..., 1]
                + random.uniform(-0.20, 0.20) * 255,
                0,
                255,
            )

            hsv[..., 2] = np.clip(
                hsv[..., 2]
                + random.uniform(-0.20, 0.20) * 255,
                0,
                255,
            )

            out = cv2.cvtColor(
                hsv.astype(np.uint8),
                cv2.COLOR_HSV2RGB,
            )

        if random.random() < 0.5:
            alpha = random.uniform(
                0.85,
                1.15,
            )

            beta = random.uniform(
                -15,
                15,
            )

            out = np.clip(
                alpha * out.astype(np.float32)
                + beta,
                0,
                255,
            ).astype(np.uint8)

        if random.random() < 0.25:
            kernel = random.choice(
                [3, 5]
            )

            out = cv2.GaussianBlur(
                out,
                (kernel, kernel),
                0,
            )

        return np.ascontiguousarray(out)

    def __getitem__(self, idx):
        path = self.images[idx]

        im = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if im is None:
            raise RuntimeError(path)

        im = cv2.cvtColor(
            im,
            cv2.COLOR_BGR2RGB,
        )

        h, w = im.shape[:2]

        scale = self.imgsz / float(
            max(h, w)
        )

        nh = int(round(h * scale))
        nw = int(round(w * scale))

        im = cv2.resize(
            im,
            (nw, nh),
            interpolation=cv2.INTER_LINEAR,
        )

        # Shared geometry for Teacher and Student.
        if random.random() < 0.5:
            im = np.ascontiguousarray(
                np.fliplr(im)
            )

        weak = im.copy()
        strong = self.strong_photo(im)

        weak = (
            torch.from_numpy(
                np.ascontiguousarray(weak)
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        strong = (
            torch.from_numpy(
                np.ascontiguousarray(strong)
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        return weak, strong, str(path)


def collate(batch):
    weak, strong, paths = zip(*batch)

    max_h = make_divisible(
        max(x.shape[1] for x in weak),
        32,
    )

    max_w = make_divisible(
        max(x.shape[2] for x in weak),
        32,
    )

    def pad(images):
        result = []

        for x in images:
            c, h, w = x.shape

            canvas = torch.full(
                (c, max_h, max_w),
                114.0 / 255.0,
                dtype=x.dtype,
            )

            canvas[:, :h, :w] = x

            result.append(canvas)

        return torch.stack(
            result,
            dim=0,
        )

    return (
        pad(weak),
        pad(strong),
        list(paths),
    )


# ================================================================
# Model / criterion helpers
# ================================================================

def ensure_seg_loss_args(
    model: nn.Module,
    epochs: int,
):
    existing = getattr(
        model,
        "args",
        None,
    )

    if existing is None:
        existing = argparse.Namespace()

    elif isinstance(existing, dict):
        existing = argparse.Namespace(
            **existing
        )

    defaults = {
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "epochs": int(epochs),

        # Simpler representation:
        # one binary mask per pseudo instance.
        "overlap_mask": False,
    }

    for key, value in defaults.items():
        setattr(
            existing,
            key,
            value,
        )

    model.args = existing


def setup_teacher_student(
    checkpoint: str,
    device: torch.device,
    epochs: int,
):
    teacher_wrapper = YOLO(checkpoint)

    teacher = (
        teacher_wrapper.model
        .to(device)
        .float()
    )

    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad_(False)

    student_wrapper = YOLO(checkpoint)

    student = (
        student_wrapper.model
        .to(device)
        .float()
    )

    ensure_seg_loss_args(
        student,
        epochs,
    )

    student.criterion = (
        student.init_criterion()
    )

    for p in student.parameters():
        p.requires_grad_(True)

    head = student.model[-1]

    if type(head).__name__ != "Segment26":
        raise RuntimeError(
            f"Expected Segment26, got {type(head).__name__}"
        )

    if not getattr(
        student,
        "end2end",
        False,
    ):
        raise RuntimeError(
            "Student is not end-to-end"
        )

    return (
        teacher,
        student,
        student_wrapper,
        student.criterion,
    )


# ================================================================
# Teacher pseudo masks
# ================================================================

@torch.no_grad()
def masks_from_coefficients(
    rows: torch.Tensor,
    proto: torch.Tensor,
    input_h: int,
    input_w: int,
    mask_threshold: float = 0.5,
):
    """
    rows:
        [N, 6 + nm]

    proto:
        [nm, Hm, Wm]

    Returns:
        binary masks [N, Hm, Wm]
    """

    if rows.numel() == 0:
        return proto.new_zeros(
            (
                0,
                proto.shape[1],
                proto.shape[2],
            )
        )

    coeff = rows[:, 6:]

    logits = torch.einsum(
        "nc,chw->nhw",
        coeff,
        proto,
    )

    probs = logits.sigmoid()

    n, mh, mw = probs.shape

    boxes = rows[:, :4].clone()

    boxes[:, [0, 2]] *= (
        float(mw) / float(input_w)
    )

    boxes[:, [1, 3]] *= (
        float(mh) / float(input_h)
    )

    ys = torch.arange(
        mh,
        device=proto.device,
        dtype=boxes.dtype,
    ).view(1, mh, 1)

    xs = torch.arange(
        mw,
        device=proto.device,
        dtype=boxes.dtype,
    ).view(1, 1, mw)

    x1 = boxes[:, 0].view(-1, 1, 1)
    y1 = boxes[:, 1].view(-1, 1, 1)
    x2 = boxes[:, 2].view(-1, 1, 1)
    y2 = boxes[:, 3].view(-1, 1, 1)

    crop = (
        (xs >= x1)
        & (xs < x2)
        & (ys >= y1)
        & (ys < y2)
    )

    masks = (
        (probs >= mask_threshold)
        & crop
    )

    return masks.float()


@torch.no_grad()
def generate_o2o_pseudo_masks(
    teacher: nn.Module,
    weak_imgs: torch.Tensor,
    conf_threshold: float,
    mask_threshold: float,
    min_mask_pixels: int,
):
    """
    First smoke test deliberately uses O2O only.

    Mask-DHF comes AFTER this path is verified.
    """

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
            "Missing dual-head branch dictionary"
        )

    all_labels = []
    all_masks = []

    h = int(weak_imgs.shape[2])
    w = int(weak_imgs.shape[3])

    for i in range(
        weak_imgs.shape[0]
    ):
        rows = final_o2o[i]

        rows = rows[
            rows[:, 4]
            >= conf_threshold
        ]

        if rows.numel() == 0:
            all_labels.append(
                rows.new_zeros(
                    (
                        0,
                        final_o2o.shape[-1],
                    )
                )
            )

            all_masks.append(
                proto.new_zeros(
                    (
                        0,
                        proto.shape[2],
                        proto.shape[3],
                    )
                )
            )

            continue

        masks = masks_from_coefficients(
            rows,
            proto[i],
            input_h=h,
            input_w=w,
            mask_threshold=mask_threshold,
        )

        areas = masks.sum(
            dim=(1, 2)
        )

        keep = (
            areas >= min_mask_pixels
        )

        rows = rows[keep]
        masks = masks[keep]

        all_labels.append(rows)
        all_masks.append(masks)

    return all_labels, all_masks


# ================================================================
# Pseudo batch for Ultralytics segmentation criterion
# ================================================================

def build_pseudo_batch(
    labels: list[torch.Tensor],
    masks: list[torch.Tensor],
    input_shape,
):
    """
    Build pseudo targets for native YOLO26 segmentation loss.

    Source-free:
      - boxes come from Teacher pseudo predictions
      - instance masks come from Teacher mask coefficients + proto
      - semantic masks are constructed from those pseudo instances
      - NO target GT is used
    """

    device = labels[0].device

    h = int(input_shape[2])
    w = int(input_shape[3])

    norm = torch.tensor(
        [w, h, w, h],
        device=device,
        dtype=torch.float32,
    )

    all_batch_idx = []
    all_cls = []
    all_boxes = []
    all_masks = []

    for img_idx, (
        rows,
        instance_masks,
    ) in enumerate(
        zip(labels, masks)
    ):
        if rows.numel() == 0:
            continue

        if (
            rows.shape[0]
            != instance_masks.shape[0]
        ):
            raise RuntimeError(
                "Pseudo box/mask count mismatch"
            )

        n = rows.shape[0]

        all_batch_idx.append(
            torch.full(
                (n,),
                img_idx,
                device=device,
                dtype=torch.long,
            )
        )

        all_cls.append(
            rows[:, 5].long()
        )

        xywh = xyxy2xywh(
            rows[:, :4]
        )

        all_boxes.append(
            xywh / norm
        )

        all_masks.append(
            instance_masks.float()
        )

    if not all_boxes:
        return None

    batch_idx = torch.cat(
        all_batch_idx,
        dim=0,
    )

    cls = torch.cat(
        all_cls,
        dim=0,
    )

    bboxes = torch.cat(
        all_boxes,
        dim=0,
    )

    instance_masks = torch.cat(
        all_masks,
        dim=0,
    )

    # ------------------------------------------------------------
    # Proto26 semantic branch
    # ------------------------------------------------------------
    #
    # Current medical task:
    #     nc = 1
    #     class 0 = polyp
    #
    # v8SegmentationLoss converts sem_masks with:
    #
    #     F.one_hot(sem_masks, num_classes=nc)
    #
    # and, when overlap_mask=False, suppresses pixels outside
    # the instance pseudo masks.
    #
    # Therefore for the single-class polyp task, every semantic
    # foreground pixel has class id 0. A zero-valued class-index
    # map is correct; the pseudo instance masks determine which
    # pixels remain active.
    #
    # This uses NO target GT.
    # ------------------------------------------------------------

    if torch.any(cls != 0):
        raise RuntimeError(
            "This smoke test assumes the single-class "
            "polyp setting with class id 0."
        )

    mask_h = int(
        instance_masks.shape[-2]
    )

    mask_w = int(
        instance_masks.shape[-1]
    )

    sem_masks = torch.zeros(
        (
            len(labels),
            mask_h,
            mask_w,
        ),
        device=device,
        dtype=torch.long,
    )

    batch = {
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,

        # overlap_mask=False:
        # [total_instances, Hm, Wm]
        "masks": instance_masks,

        # Proto26 semantic branch:
        # [batch_size, Hm, Wm]
        "sem_masks": sem_masks,
    }

    return batch


# ================================================================
# EMA
# ================================================================

@torch.no_grad()
def update_teacher_ema(
    teacher: nn.Module,
    student: nn.Module,
    momentum: float,
):
    for tp, sp in zip(
        teacher.parameters(),
        student.parameters(),
    ):
        tp.data.mul_(
            momentum
        ).add_(
            sp.data,
            alpha=1.0 - momentum,
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
        "--conf",
        type=float,
        default=0.5,
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
            "smoke_mt_seg"
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
    print("STAGE-2 MEAN TEACHER SEGMENTATION SMOKE")
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
        "pseudo mode   : O2O masks only"
    )

    print(
        "Mask-DHF      : OFF"
    )

    print(
        "MARD          : OFF"
    )

    print(
        "RASP          : OFF"
    )

    print(
        "max batches   :",
        args.max_batches,
    )

    successful = 0
    skipped = 0
    total_pseudo = 0

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
        if batch_idx > args.max_batches:
            break

        weak = weak.to(
            device,
            non_blocking=True,
        )

        strong = strong.to(
            device,
            non_blocking=True,
        )

        pseudo_labels, pseudo_masks = (
            generate_o2o_pseudo_masks(
                teacher,
                weak,
                conf_threshold=args.conf,
                mask_threshold=args.mask_thr,
                min_mask_pixels=args.min_mask_pixels,
            )
        )

        valid_indices = [
            i
            for i, x in enumerate(
                pseudo_labels
            )
            if x.shape[0] > 0
        ]

        if not valid_indices:
            skipped += 1

            print(
                f"[Batch {batch_idx}] "
                "SKIP no pseudo masks"
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
                "Student training output "
                "is not dual-head dict"
            )

        if (
            "proto"
            not in student_outputs["one2one"]
        ):
            raise RuntimeError(
                "Student O2O proto missing"
            )

        if (
            "proto"
            not in student_outputs["one2many"]
        ):
            raise RuntimeError(
                "Student O2M proto missing"
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

        total_loss = (
            total_loss_vec.sum()
        )

        if not torch.isfinite(
            total_loss
        ):
            raise RuntimeError(
                f"Non-finite total loss: "
                f"{total_loss}"
            )

        if len(loss_items) < 5:
            raise RuntimeError(
                f"Unexpected segmentation loss vector: "
                f"len={len(loss_items)}"
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

        successful += 1

        loss_float = float(
            total_loss.detach().cpu()
        )

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

        loss_values.append(
            loss_float
        )

        seg_values.append(
            seg_loss
        )

        mask_pixels = torch.cat(
            masks_valid,
            dim=0,
        ).sum(
            dim=(1, 2)
        )

        print(
            f"[Batch {batch_idx:02d}] "
            f"valid_img={len(valid_indices)}/{weak.shape[0]} "
            f"pseudo={n_pseudo} "
            f"loss={loss_float:.4f} "
            f"box={box_loss:.4f} "
            f"seg={seg_loss:.4f} "
            f"cls={cls_loss:.4f} "
            f"dfl={dfl_loss:.4f} "
            f"semseg={semseg_loss:.4f} "
            f"grad={float(grad_norm):.4f} "
            f"mask_px_mean="
            f"{float(mask_pixels.float().mean()):.1f}"
        )

    if successful == 0:
        raise RuntimeError(
            "No successful optimization batch"
        )

    # Test the EMA path once after smoke optimization.
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
        / "student_SMOKE.pt"
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
        "pseudo_mode": "o2o_mask_only",
        "mask_dhf": False,
        "mard": False,
        "rasp": False,
        "max_batches": args.max_batches,
        "successful_batches": successful,
        "skipped_batches": skipped,
        "total_pseudo_instances": total_pseudo,
        "mean_total_loss": float(
            np.mean(loss_values)
        ),
        "mean_seg_loss": float(
            np.mean(seg_values)
        ),
        "elapsed_seconds": (
            time.time() - start
        ),
        "checkpoint": str(
            smoke_ckpt
        ),
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
    print("[PASS] MT SEGMENTATION LOSS PATH WORKS")
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
        "as a final experiment."
    )


if __name__ == "__main__":
    main()
