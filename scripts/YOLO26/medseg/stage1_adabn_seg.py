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
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# Force local repository Ultralytics
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402


IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def resolve_device(device: str) -> torch.device:
    if device.lower() == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        print("[WARN] CUDA unavailable -> using CPU")
        return torch.device("cpu")

    if device.isdigit():
        return torch.device(f"cuda:{device}")

    return torch.device(device)


def make_divisible(x: int, divisor: int = 32) -> int:
    return int(
        math.ceil(float(x) / divisor) * divisor
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_target_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"Target image directory not found: {root}"
        )

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
            f"No target images found under {root}"
        )

    return images


# ---------------------------------------------------------------------
# Image-only target dataset
# ---------------------------------------------------------------------

class TargetImageDataset(Dataset):
    """
    Image-only target-domain dataset.

    IMPORTANT:
        This dataset has no label path,
        no mask path,
        and no Ultralytics dataset YAML.

    Therefore Stage-1 AdaBN cannot accidentally load target GT.
    """

    def __init__(
        self,
        images: list[Path],
        imgsz: int = 640,
        flip_prob: float = 0.5,
    ):
        self.images = images
        self.imgsz = int(imgsz)
        self.flip_prob = float(flip_prob)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.images[idx]

        im = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if im is None:
            raise RuntimeError(
                f"Could not read image: {path}"
            )

        # BGR -> RGB
        im = cv2.cvtColor(
            im,
            cv2.COLOR_BGR2RGB,
        )

        h, w = im.shape[:2]

        # Long-edge resize.
        scale = self.imgsz / float(max(h, w))

        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        im = cv2.resize(
            im,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Weak-view geometry only.
        if (
            self.flip_prob > 0
            and random.random() < self.flip_prob
        ):
            im = np.ascontiguousarray(
                np.fliplr(im)
            )

        tensor = torch.from_numpy(
            np.ascontiguousarray(im)
        )

        tensor = (
            tensor
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        return tensor


def collate_target(
    batch: list[torch.Tensor],
) -> torch.Tensor:
    """
    Pad batch to stride-32 shape.

    For CVC 384x288 at imgsz=640:
        resized ~= 640x480
        batch tensor ~= [B, 3, 480, 640]
    """

    max_h = make_divisible(
        max(x.shape[1] for x in batch),
        32,
    )

    max_w = make_divisible(
        max(x.shape[2] for x in batch),
        32,
    )

    output = []

    for x in batch:
        c, h, w = x.shape

        canvas = torch.full(
            (c, max_h, max_w),
            114.0 / 255.0,
            dtype=x.dtype,
        )

        # Top-left placement to match weak-view pipeline.
        canvas[:, :h, :w] = x

        output.append(canvas)

    return torch.stack(
        output,
        dim=0,
    )


# ---------------------------------------------------------------------
# BN handling
# ---------------------------------------------------------------------

def get_bn_modules(
    model: nn.Module,
) -> list[tuple[str, nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(
            module,
            (
                nn.BatchNorm2d,
                nn.SyncBatchNorm,
            ),
        )
    ]


def freeze_except_bn_stats(
    model: nn.Module,
) -> int:
    """
    Freeze all learned parameters.

    Entire model is eval mode except BN modules,
    which are set to train mode so running_mean/running_var update.

    No gradients are enabled.
    """

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    bn_modules = get_bn_modules(model)

    for _, module in bn_modules:
        module.train()

    return len(bn_modules)


@torch.no_grad()
def capture_bn_stats(
    model: nn.Module,
) -> dict[str, dict[str, torch.Tensor]]:
    result = {}

    for name, module in get_bn_modules(model):
        result[name] = {
            "running_mean":
                module.running_mean.detach().cpu().clone(),
            "running_var":
                module.running_var.detach().cpu().clone(),
        }

    return result


@torch.no_grad()
def compute_bn_delta(
    before: dict,
    after: dict,
) -> dict:
    mean_deltas = []
    var_deltas = []
    changed_layers = 0

    for name in before:
        if name not in after:
            continue

        dm = (
            after[name]["running_mean"]
            - before[name]["running_mean"]
        ).abs().mean().item()

        dv = (
            after[name]["running_var"]
            - before[name]["running_var"]
        ).abs().mean().item()

        mean_deltas.append(dm)
        var_deltas.append(dv)

        if dm > 0.0 or dv > 0.0:
            changed_layers += 1

    return {
        "bn_layers": len(before),
        "changed_bn_layers": changed_layers,
        "mean_abs_running_mean_delta":
            float(np.mean(mean_deltas))
            if mean_deltas else 0.0,
        "mean_abs_running_var_delta":
            float(np.mean(var_deltas))
            if var_deltas else 0.0,
    }


# ---------------------------------------------------------------------
# Main AdaBN
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Stage-1 image-only AdaBN for "
            "YOLO26 medical segmentation"
        )
    )

    ap.add_argument(
        "--weights",
        required=True,
    )

    ap.add_argument(
        "--target-images",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        default=str(
            REPO_ROOT
            / "runs"
            / "seg"
            / "stage1"
            / "cvc_adabn"
        ),
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
        "--epochs",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--flip-prob",
        type=float,
        default=0.5,
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

    args = ap.parse_args()

    seed_everything(args.seed)

    device = resolve_device(args.device)

    weights = Path(args.weights).resolve()
    target_dir = Path(
        args.target_images
    ).resolve()

    out_dir = Path(args.out_dir).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ckpt_path = (
        out_dir
        / "yolo26s_seg_cvc_adabn.pt"
    )

    meta_path = (
        out_dir
        / "adabn_metadata.json"
    )

    print("=" * 72)
    print("STAGE 1 — YOLO26-S-SEG AdaBN")
    print("=" * 72)

    print("weights       :", weights)
    print("target images :", target_dir)
    print("output        :", ckpt_path)
    print("device        :", device)
    print("imgsz         :", args.imgsz)
    print("batch         :", args.batch)
    print("epochs        :", args.epochs)
    print("flip_prob     :", args.flip_prob)
    print("seed          :", args.seed)

    if not weights.exists():
        raise FileNotFoundError(
            weights
        )

    # -----------------------------------------------------------------
    # Source-free guardrail
    # -----------------------------------------------------------------

    print()
    print("[Protocol]")
    print("  target labels loaded : NO")
    print("  target masks loaded  : NO")
    print("  loss                 : NO")
    print("  backward             : NO")
    print("  optimizer            : NO")
    print("  BN running stats     : UPDATE")

    # -----------------------------------------------------------------
    # Target images only
    # -----------------------------------------------------------------

    images = list_target_images(
        target_dir
    )

    print()
    print(
        "target image count:",
        len(images),
    )

    dataset = TargetImageDataset(
        images=images,
        imgsz=args.imgsz,
        flip_prob=args.flip_prob,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
        collate_fn=collate_target,
        generator=generator,
    )

    # -----------------------------------------------------------------
    # Load frozen source checkpoint
    # -----------------------------------------------------------------

    print()
    print("[Load] source checkpoint")

    wrapper = YOLO(
        str(weights)
    )

    net = (
        wrapper.model
        .to(device)
        .float()
    )

    bn_count = freeze_except_bn_stats(
        net
    )

    print(
        "BatchNorm layers:",
        bn_count,
    )

    if bn_count == 0:
        raise RuntimeError(
            "No BatchNorm layers found. "
            "AdaBN cannot run on a fused model."
        )

    trainable = sum(
        p.numel()
        for p in net.parameters()
        if p.requires_grad
    )

    print(
        "trainable parameters:",
        trainable,
    )

    if trainable != 0:
        raise RuntimeError(
            "AdaBN guardrail failed: "
            "trainable parameters != 0"
        )

    bn_before = capture_bn_stats(
        net
    )

    # Save source BN buffers for reproducibility.
    torch.save(
        bn_before,
        out_dir / "source_bn_stats.pt",
    )

    # -----------------------------------------------------------------
    # AdaBN passes
    # -----------------------------------------------------------------

    print()
    print("[AdaBN] start")

    total_seen = 0

    start = time.time()

    with torch.inference_mode():
        for epoch in range(
            1,
            args.epochs + 1,
        ):
            freeze_except_bn_stats(net)

            epoch_seen = 0
            epoch_start = time.time()

            for batch_idx, ims in enumerate(
                loader,
                start=1,
            ):
                ims = ims.to(
                    device,
                    non_blocking=True,
                )

                _ = net(ims)

                bs = int(
                    ims.shape[0]
                )

                epoch_seen += bs
                total_seen += bs

                if (
                    batch_idx == 1
                    or batch_idx % 20 == 0
                    or batch_idx == len(loader)
                ):
                    print(
                        f"[Epoch {epoch}/{args.epochs}] "
                        f"batch={batch_idx}/{len(loader)} "
                        f"seen={epoch_seen}/{len(dataset)} "
                        f"shape={tuple(ims.shape)}"
                    )

            print(
                f"[Epoch {epoch}] "
                f"images={epoch_seen} "
                f"time={time.time() - epoch_start:.2f}s"
            )

    elapsed = (
        time.time() - start
    )

    # -----------------------------------------------------------------
    # Verify BN statistics changed
    # -----------------------------------------------------------------

    bn_after = capture_bn_stats(
        net
    )

    delta = compute_bn_delta(
        bn_before,
        bn_after,
    )

    print()
    print("[BN delta]")

    for k, v in delta.items():
        print(
            f"{k}: {v}"
        )

    if (
        delta["changed_bn_layers"] == 0
    ):
        raise RuntimeError(
            "No BN running statistics changed."
        )

    # -----------------------------------------------------------------
    # Save adapted checkpoint
    # -----------------------------------------------------------------

    # Put model in eval mode before serialization.
    net.eval()

    wrapper.model = net

    wrapper.save(
        str(ckpt_path)
    )

    if not ckpt_path.exists():
        raise RuntimeError(
            "Adapted checkpoint was not created"
        )

    # Load-back verification.
    check_wrapper = YOLO(
        str(ckpt_path)
    )

    check_bn = len(
        get_bn_modules(
            check_wrapper.model
        )
    )

    if check_bn == 0:
        raise RuntimeError(
            "Saved checkpoint reload contains no BN layers"
        )

    metadata = {
        "stage": "stage1_adabn_seg",
        "source_weights": str(weights),
        "target_images": str(target_dir),
        "target_image_count": len(images),
        "target_labels_used": False,
        "target_masks_used": False,
        "loss_used": False,
        "backward_used": False,
        "optimizer_used": False,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "epochs": args.epochs,
        "flip_prob": args.flip_prob,
        "seed": args.seed,
        "device": str(device),
        "total_images_forwarded": total_seen,
        "elapsed_seconds": elapsed,
        "checkpoint": str(ckpt_path),
        **delta,
    }

    meta_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("[PASS] Stage-1 AdaBN completed")
    print("=" * 72)

    print(
        "checkpoint:",
        ckpt_path,
    )

    print(
        "metadata  :",
        meta_path,
    )


if __name__ == "__main__":
    main()
