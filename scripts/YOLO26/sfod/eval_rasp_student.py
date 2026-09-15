"""Evaluate either an effective masked RASP student or a compact exported model.

Target labels are used here only for final reporting/diagnostics, never for model
selection inside Stage-2 adaptation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO  # noqa: E402
from rasp_pruning import pruner_from_state  # noqa: E402
from stage2_rtsfod_yolo26 import resolve_device, val_device_arg  # noqa: E402


def main(args):
    device = resolve_device(args.device)
    wrapper = YOLO(args.model)
    model = wrapper.model.to(device).float()
    pruner = None

    if args.state:
        state = torch.load(args.state, map_location=device, weights_only=False)
        if "student_state" in state:
            model.load_state_dict(state["student_state"], strict=True)
        rasp_state = state.get("rasp", state)
        if not rasp_state.get("enabled", False):
            raise RuntimeError("--state does not contain enabled RASP masks")
        pruner = pruner_from_state(model, rasp_state["pruner"])
        pruner.load_state_dict(rasp_state["pruner"], strict=True)
        pruner.install()
        print(f"Evaluating masked latent student, hidden sparsity={pruner.sparsity():.3f}")
    else:
        print("Evaluating model directly (expected compact export or dense baseline).")

    wrapper.model = model
    metrics = wrapper.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=val_device_arg(device),
        plots=args.plots,
        verbose=True,
    )
    box = metrics.box
    print(f"mAP50={float(box.map50):.6f} mAP50-95={float(box.map):.6f}")

    if pruner is not None:
        pruner.uninstall()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--state", default="", help="Optional RASP training state to apply masks to a latent checkpoint")
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--plots", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
