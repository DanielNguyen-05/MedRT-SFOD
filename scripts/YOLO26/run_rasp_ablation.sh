#!/usr/bin/env bash
set -euo pipefail

# Example only. Override these environment variables before running.
: "${STAGE1_MODEL:?Set STAGE1_MODEL=/path/to/stage1.pt}"
: "${TARGET_YAML:?Set TARGET_YAML=/path/to/foggy_cityscapes.yaml}"
OUT_ROOT="${OUT_ROOT:-runs/rasp_c2f}"
DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-1024}"
BATCH="${BATCH:-4}"

# A) Frozen baseline implementation from the original project.
PYTHONPATH="$PWD" python scripts/YOLO26/stage2_rtsfod_yolo26.py \
  --stage1_model "$STAGE1_MODEL" --data "$TARGET_YAML" \
  --out_dir "$OUT_ROOT/baseline" --device "$DEVICE" --imgsz "$IMGSZ" --batch "$BATCH"

# B) Main RASP method. No target-label --eval during adaptation.
PYTHONPATH="$PWD" python scripts/YOLO26/stage2_rasp_rtsfod_yolo26.py \
  --stage1_model "$STAGE1_MODEL" --data "$TARGET_YAML" \
  --out_dir "$OUT_ROOT/rasp" --device "$DEVICE" --imgsz "$IMGSZ" --batch "$BATCH" \
  --rasp_enable
