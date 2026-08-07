#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f scripts/YOLO26/stage2_rtsfod_yolo26.py ]]; then
  echo "ERROR: run this from the existing MedRT-SFOD repository root." >&2
  echo "Missing scripts/YOLO26/stage2_rtsfod_yolo26.py" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
cp "$HERE"/scripts/YOLO26/rasp_pruning.py scripts/YOLO26/
cp "$HERE"/scripts/YOLO26/stage2_rasp_rtsfod_yolo26.py scripts/YOLO26/
cp "$HERE"/scripts/YOLO26/inspect_rasp_prunable_groups.py scripts/YOLO26/
cp "$HERE"/scripts/YOLO26/export_rasp_compact.py scripts/YOLO26/
cp "$HERE"/scripts/YOLO26/eval_rasp_student.py scripts/YOLO26/
cp "$HERE"/scripts/YOLO26/test_rasp_smoke.py scripts/YOLO26/
cp "$HERE"/scripts/YOLO26/run_rasp_ablation.sh scripts/YOLO26/
cp "$HERE"/requirements-rasp.txt ./requirements-rasp.txt
cp "$HERE"/README_RASP.md ./README_RASP.md
mkdir -p colab
cp "$HERE"/colab/05_rasp_audit_train.ipynb colab/
cp "$HERE"/colab/06_rasp_export_eval.ipynb colab/

echo "RASP overlay installed. Next:"
echo "  pip install -r requirements-rasp.txt"
echo '  PYTHONPATH="$PWD" python scripts/YOLO26/test_rasp_smoke.py'
