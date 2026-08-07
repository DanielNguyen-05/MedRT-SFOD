#!/usr/bin/env bash
# Convert raw Kvasir-SEG -> YOLO format, then run source-supervised training.
#
# Rewritten: the previous version of this script hardcoded
# `/Users/dangnguyen/Desktop/MedRT-SFOD`, so it only ran on one specific
# machine, and it called `train_polyp_detection.py`, whose loss never used
# the ground-truth boxes (see architecture_benchmark.py's docstring / the
# code review). It now:
#   1. Resolves the repo root relative to this script's own location.
#   2. Converts the raw Kvasir-SEG download to YOLO format via
#      polyp_kvasir_to_yolo.py (fixing the "only bbox[0]" multi-polyp bug).
#   3. Trains with train_source_supervised.py, which wraps the real
#      Ultralytics YOLO(cfg).train() API instead of a hand-rolled loss.
#
# Usage:
#   ./run_polyp_training.sh /path/to/raw/Kvasir-SEG [detect|segment]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_KVASIR_SRC="${1:?Usage: $0 <path-to-raw-Kvasir-SEG> [detect|segment]}"
TASK="${2:-detect}"
DATASETS_ROOT="${DATASETS_ROOT:-${REPO_ROOT}/../datasets}"   # sibling of the repo, override via env var
CONVERTED_DST="${DATASETS_ROOT}/polyp_${TASK}"

if [[ "${TASK}" == "detect" ]]; then
  MODEL_CFG="ultralytics/cfg/models/26/yolo26-lite.yaml"
  DATA_YAML="${CONVERTED_DST}/dataset_detect.yaml"
elif [[ "${TASK}" == "segment" ]]; then
  MODEL_CFG="ultralytics/cfg/models/26/yolo26-lite-seg.yaml"
  DATA_YAML="${CONVERTED_DST}/dataset_seg.yaml"
else
  echo "TASK must be 'detect' or 'segment', got: ${TASK}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "[run_polyp_training] Converting ${RAW_KVASIR_SRC} -> ${CONVERTED_DST} (task=${TASK})"
python3 scripts/YOLO26/polyp_kvasir_to_yolo.py \
  --src "${RAW_KVASIR_SRC}" \
  --dst "${CONVERTED_DST}" \
  --task "${TASK}"

echo "[run_polyp_training] Training (source-supervised, task=${TASK})"
python3 scripts/YOLO26/train_source_supervised.py \
  --model-cfg "${MODEL_CFG}" \
  --data "${DATA_YAML}" \
  --task "${TASK}" \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --out-dir "runs/polyp_${TASK}_source"
