"""Stage 0/1 AdaBN/optional RC with source-free protocol guardrails.

Changes vs the uploaded version:
- preprocessing now matches Stage-2 weak-view geometry: long-edge resize,
  optional shared-style flip, then top-left batch padding to /32.  The old
  center-padded fixed square changed the BN input distribution between stages.
- no colour jitter by default during AdaBN; ``--strong_aug`` is explicitly an
  optional experiment rather than silently changing every target image.
- target-label model selection is renamed ``--oracle_early_stop`` and loudly
  marked diagnostic-only.  Main source-free results must leave it OFF.
- evaluation is task-aware: segmentation checkpoints use ``metrics.seg`` when
  available instead of assuming ``metrics.box`` everywhere.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from ultralytics import YOLO  # noqa: E402

MODEL_FAMILY = "YOLO26"
MODEL_TAG = "yolo26"


class DetectInputFeatureHook:  # compatibility for older pickles
    def _hook_fn(self, *args, **kwargs):
        pass


def resolve_device(s: str) -> torch.device:
    if s.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if s.isdigit():
        return torch.device(f"cuda:{s}")
    return torch.device(s)


def val_device_arg(device: torch.device) -> str:
    return str(device.index if device.type == "cuda" and device.index is not None else device)


def make_divisible(x: int, divisor: int = 32) -> int:
    return int(math.ceil(float(x) / divisor) * divisor)


def list_images_from_yaml(data_yaml: str) -> list[str]:
    with open(data_yaml, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    root = Path(y.get("path", "."))
    train_path = root / y["train"]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs: list[str] = []
    if train_path.is_file():
        for line in train_path.read_text().splitlines():
            p = Path(line.strip())
            if p.suffix.lower() in exts and p.exists():
                imgs.append(str(p))
    else:
        for dp, _, files in os.walk(train_path):
            imgs.extend(str(Path(dp) / fn) for fn in files if Path(fn).suffix.lower() in exts)
    return sorted(imgs)


class TargetImgDataset(data.Dataset):
    def __init__(self, img_paths: list[str], img_size: int, strong_aug: bool = False):
        self.imgs = img_paths
        self.img_size = int(img_size)
        self.strong_aug = bool(strong_aug)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.imgs[idx]
        im = cv2.imread(path)
        if im is None:
            raise FileNotFoundError(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        scale = self.img_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Match Stage-2 weak view: resize + horizontal flip only by default.
        if random.random() < 0.5:
            im = np.ascontiguousarray(np.fliplr(im))

        if self.strong_aug:
            if random.random() < 0.8:
                hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 0] = np.clip(hsv[..., 0] + random.uniform(-0.15, 0.15) * 180, 0, 179)
                hsv[..., 1] = np.clip(hsv[..., 1] + random.uniform(-0.2, 0.2) * 255, 0, 255)
                hsv[..., 2] = np.clip(hsv[..., 2] + random.uniform(-0.2, 0.2) * 255, 0, 255)
                im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            if random.random() < 0.5:
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-20, 20)
                im = np.clip(alpha * im.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        return torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0


def collate(batch: list[torch.Tensor]) -> torch.Tensor:
    max_h = make_divisible(max(x.shape[1] for x in batch), 32)
    max_w = make_divisible(max(x.shape[2] for x in batch), 32)
    out = []
    for x in batch:
        c, h, w = x.shape
        canvas = torch.full((c, max_h, max_w), 114 / 255.0, dtype=x.dtype)
        canvas[:, :h, :w] = x
        out.append(canvas)
    return torch.stack(out, 0)


def dump_bn_priors_from_model(model: nn.Module, out_path: Path) -> dict[str, list]:
    priors = {
        k: v.detach().cpu().tolist()
        for k, v in model.state_dict().items()
        if k.endswith("running_mean") or k.endswith("running_var")
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(priors))
    print(f"[Stage 0] saved {len(priors)} BN tensors -> {out_path}")
    return priors


def bn_priors_to_device(priors: dict[str, list], device: torch.device):
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in priors.items()}


@torch.no_grad()
def blend_bn_running_stats(model: nn.Module, priors: dict[str, torch.Tensor], alpha: float) -> int:
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            continue
        for attr in ("running_mean", "running_var"):
            key = f"{name}.{attr}" if name else attr
            if key in priors and getattr(module, attr, None) is not None:
                current = getattr(module, attr)
                current.copy_((1.0 - alpha) * current + alpha * priors[key])
                count += 1
    return count


def set_bn_train_no_grad(model: nn.Module) -> None:
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.train()


@torch.no_grad()
def restore_bn_buffers(model: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    for name, module in model.named_modules():
        if not isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            continue
        for attr in ("running_mean", "running_var", "num_batches_tracked"):
            key = f"{name}.{attr}" if name else attr
            target = getattr(module, attr, None)
            if key in state and target is not None:
                target.copy_(state[key].to(device))


def extract_task_metrics(metrics) -> tuple[str, float, float, float]:
    seg = getattr(metrics, "seg", None)
    if seg is not None and hasattr(seg, "map50"):
        return "seg", float(seg.map), float(seg.map50), float(seg.map75)
    box = getattr(metrics, "box", None)
    if box is not None and hasattr(box, "map50"):
        return "box", float(box.map), float(box.map50), float(box.map75)
    return "unknown", 0.0, 0.0, 0.0


def validate(wrapper: YOLO, net: nn.Module, args, device: torch.device, epoch_idx: int) -> float:
    wrapper.model = net
    metrics = wrapper.val(
        data=args.data, imgsz=args.imgsz, batch=args.batch * 2, conf=0.001,
        iou=0.6, device=val_device_arg(device), verbose=False, plots=False,
    )
    task, _, map50, _ = extract_task_metrics(metrics)
    print(f"[ORACLE validation] epoch={epoch_idx} task={task} mAP50={map50:.4f}")
    set_bn_train_no_grad(net)
    return map50


def main(args) -> None:
    if args.oracle_early_stop:
        print("[WARNING] --oracle_early_stop reads TARGET validation labels. Use only for diagnostics/upper bounds, NEVER for main SFDA/SFOD model selection.")

    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = YOLO(args.weights)
    net = wrapper.model.to(device).float()

    priors_path = Path(args.bn_priors_out) if args.bn_priors_out else out_dir / f"{MODEL_TAG}_stage0_bn_priors.json"
    priors = bn_priors_to_device(dump_bn_priors_from_model(net, priors_path), device)

    imgs = list_images_from_yaml(args.data)
    if not imgs:
        raise RuntimeError(f"No target training images found in {args.data}")
    ds = TargetImgDataset(imgs, args.imgsz, args.strong_aug)
    loader = data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                             pin_memory=True, collate_fn=collate, drop_last=False)

    set_bn_train_no_grad(net)
    best_map50, best_epoch, best_state = -1.0, -1, None

    def run_epoch(epoch_idx: int, blend_alpha: float) -> bool:
        nonlocal best_map50, best_epoch, best_state
        set_bn_train_no_grad(net)
        t0 = time.time()
        n_img = 0
        for ims in loader:
            ims = ims.to(device, non_blocking=True)
            with torch.no_grad():
                _ = net(ims)
            n_img += ims.shape[0]
        n_blend = blend_bn_running_stats(net, priors, blend_alpha) if blend_alpha > 0 else 0
        print(f"[Stage1 epoch {epoch_idx}] imgs={n_img} blend_alpha={blend_alpha:.4f} blended={n_blend} time={time.time()-t0:.1f}s")

        if args.oracle_early_stop and epoch_idx % args.val_interval == 0:
            score = validate(wrapper, net, args, device, epoch_idx)
            if score > best_map50:
                best_map50, best_epoch = score, epoch_idx
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if epoch_idx - best_epoch >= args.patience:
                print("[ORACLE early stop] patience reached")
                return True
        return False

    stopped = False
    print("[Stage 1A] AdaBN")
    for epoch in range(args.epochs_adabn):
        if run_epoch(epoch, 0.0):
            stopped = True
            break

    if not stopped and args.epochs_rc > 0:
        print("[Stage 1B] optional RC")
        for rc_epoch in range(args.epochs_rc):
            alpha = args.alpha0 if args.epochs_rc <= 1 else args.alpha0 * 0.5 * (1 + math.cos(math.pi * rc_epoch / (args.epochs_rc - 1)))
            if run_epoch(args.epochs_adabn + rc_epoch, alpha):
                break

    if args.oracle_early_stop and best_state is not None:
        restore_bn_buffers(net, best_state, device)
        print(f"[ORACLE] restored epoch {best_epoch}; do not use this checkpoint as a main source-free result")

    dataset_name = Path(args.data).stem
    ckpt = out_dir / f"{MODEL_TAG}_stage1_adabnrc_{dataset_name}.pt"
    wrapper.model = net
    wrapper.save(str(ckpt))
    print(f"[Stage 1] saved -> {ckpt}")

    if args.eval:
        rows = []
        for label, path in (("source", args.weights), ("stage1_adapted", str(ckpt))):
            m = YOLO(path).val(data=args.data, imgsz=args.imgsz, batch=args.batch * 2, conf=0.001,
                               iou=0.6, device=val_device_arg(device), verbose=False, plots=(label != "source"))
            task, map_, map50, map75 = extract_task_metrics(m)
            rows.append((label, task, map_, map50, map75))
        csv_path = out_dir / f"{MODEL_TAG}_stage1_evaluation_comparison.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["model", "metric_task", "map", "map50", "map75"]); w.writerows(rows)
        print(f"[Eval] saved -> {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="YOLO26 Stage0 BN priors + Stage1 source-free AdaBN/optional RC")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bn_priors_out", default=None)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs_adabn", type=int, default=2)
    ap.add_argument("--epochs_rc", type=int, default=0)
    ap.add_argument("--alpha0", type=float, default=0.10)
    ap.add_argument("--strong_aug", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--oracle_early_stop", action="store_true", help="DIAGNOSTIC ONLY: target-label model selection; invalid for main source-free results")
    ap.add_argument("--val_interval", type=int, default=1)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--device", default="0")
    main(ap.parse_args())
