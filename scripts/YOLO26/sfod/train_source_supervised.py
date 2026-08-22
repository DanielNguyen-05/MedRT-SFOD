"""Source-supervised training (Stage -1) — replaces train_city_detection.py,
train_polyp_detection.py, and train_polyp_segmentation.py.

Why one script instead of three: those three scripts each hand-rolled a
`nn.Sequential` toy model and a manual loss, and one of them
(`train_polyp_detection.py`) computed `loss = outputs.mean()` — a loss that
never referenced the loaded ground-truth boxes at all, so "training" only
pushed outputs toward zero and never learned detection. Reimplementing
YOLO's target assignment (O2O/O2M matching, DFL, box/cls loss weighting) by
hand is exactly the kind of thing that produces bugs like that; Ultralytics'
own `YOLO(cfg).train(data=...)` already implements this correctly, so this
script is a thin, correct wrapper around it instead of a second
reimplementation.

This script performs the *source-supervised* pretraining step that has to
happen before Stage 0/1 (AdaBN) and Stage 2 (DHF+MARD+/-CARD) — i.e. it
trains a normal detector/segmenter on a domain's *labeled* data, producing
the checkpoint that `stage0_stage1_adabn_rc_yolo26.py --weights <this>.pt`
then adapts to an unlabeled target domain. It does not implement source-free
adaptation itself.

Usage:
    # Cityscapes source-only detection
    python train_source_supervised.py \
        --model-cfg ultralytics/cfg/models/26/yolo26-lite.yaml \
        --data ultralytics/cfg/datasets/c2f_example.yaml \
        --task detect --epochs 100 --imgsz 1024 --out-dir runs/city_source

    # Polyp detection (after polyp_kvasir_to_yolo.py --task detect)
    python train_source_supervised.py \
        --model-cfg ultralytics/cfg/models/26/yolo26-lite.yaml \
        --data /path/to/datasets/polyp_detect/dataset_detect.yaml \
        --task detect --epochs 100 --imgsz 640 --out-dir runs/polyp_detect_source

    # Polyp segmentation (after polyp_kvasir_to_yolo.py --task segment)
    python train_source_supervised.py \
        --model-cfg ultralytics/cfg/models/26/yolo26-lite-seg.yaml \
        --data /path/to/datasets/polyp_seg/dataset_seg.yaml \
        --task segment --epochs 100 --imgsz 640 --out-dir runs/polyp_seg_source
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402  (requires ultralytics/data/ to be patched in, see README)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Source-supervised training for city or polyp domains, detect or segment")
    parser.add_argument("--model-cfg", type=str, default=None, help="Model YAML used only when --weights is not supplied")
    parser.add_argument("--weights", type=str, default=None, help="Pretrained checkpoint, e.g. yolo26m.pt")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML (e.g. from polyp_kvasir_to_yolo.py)")
    parser.add_argument("--task", type=str, choices=["detect", "segment"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0", help="CUDA index, comma-separated indices, or 'cpu'")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.weights:
        print(f"[train_source_supervised] pretrained weights={args.weights}")
        model = YOLO(args.weights, task=args.task)
    elif args.model_cfg:
        print(f"[train_source_supervised] WARNING: building from YAML / scratch: {args.model_cfg}")
        model = YOLO(args.model_cfg, task=args.task)
    else:
        raise ValueError("Provide either --weights or --model-cfg")

    print(f"[train_source_supervised] task={args.task} data={args.data}")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(out_dir.parent),
        name=out_dir.name,
        resume=args.resume,
        exist_ok=True,
    )
    print(f"[train_source_supervised] Done. Checkpoints under {out_dir}/weights/ (best.pt, last.pt).")
    print(
        "[train_source_supervised] Next step: feed the produced best.pt into "
        "stage0_stage1_adabn_rc_yolo26.py --weights <best.pt> on the UNLABELED target-domain "
        "data to start source-free adaptation."
    )


if __name__ == "__main__":
    main()
