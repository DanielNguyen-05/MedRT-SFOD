# MedRT-SFOD — Medical Source-Free Segmentation Pipeline

> Repository root: `~/MedRT-SFOD`  
> Source: **Kvasir-SEG**  
> Target: **CVC-ClinicDB**  
> Model: **YOLO26-S-Seg**  
> Main adaptation: **AdaBN → Mean Teacher → Mask-DHF → MARD**  
> Compression: **RASP-SFSeg → physical compaction**

BraTS2023: xem [hướng dẫn train/val/test](docs/BraTS2023.md) và
[notebook GPU](colab/07_BraTS2023_Train_Val_Test.ipynb) cho baseline YOLO26-S-Seg
với split theo bệnh nhân, bản sửa MEN và đánh giá mask gốc theo từng nhóm GLI/MEN/PED.

---

# 0. Environment

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"
```

```bash
python - <<'PY'
import torch
import ultralytics

print("VIRTUAL_ENV  :", __import__("os").environ.get("VIRTUAL_ENV"))
print("torch        :", torch.__version__)
print("cuda         :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu          :", torch.cuda.get_device_name(0))
print("ultralytics  :", ultralytics.__file__)
PY
```

RASP dependencies:

```bash
python - <<'PY'
mods = ["sklearn", "kneed", "torch_pruning"]
for mod in mods:
    try:
        m = __import__(mod)
        print("[OK]", mod, getattr(m, "__version__", "available"))
    except Exception as e:
        print("[MISSING]", mod, e)
PY
```

If missing:

```bash
python -m pip install scikit-learn kneed torch-pruning thop
```

---

# 1. Project pre-flight

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"

python - <<'PY'
from pathlib import Path

required = [
    "scripts/YOLO26/medseg/prepare_kvasir.py",
    "scripts/YOLO26/medseg/train_source_seg.py",
    "scripts/YOLO26/medseg/check_yolo26s_seg.py",
    "scripts/YOLO26/medseg/eval_source_seg.py",
    "scripts/YOLO26/medseg/prepare_cvc_clinicdb.py",
    "scripts/YOLO26/medseg/stage1_adabn_seg.py",
    "scripts/YOLO26/medseg/audit_stage2_seg.py",
    "scripts/YOLO26/medseg/mask_dhf_seg.py",
    "scripts/YOLO26/medseg/audit_mask_dhf_seg.py",
    "scripts/YOLO26/medseg/audit_mask_stability_extras.py",
    "scripts/YOLO26/medseg/stage2_dense_sfseg.py",
    "scripts/YOLO26/medseg/inspect_rasp_prunable_groups_seg.py",
    "scripts/YOLO26/medseg/stage3_rasp_sfseg.py",
    "scripts/YOLO26/rasp_pruning.py",
    "scripts/YOLO26/export_rasp_compact.py",
]

missing = [p for p in required if not Path(p).exists()]
if missing:
    print("MISSING:")
    for p in missing:
        print(" -", p)
    raise SystemExit(1)

print("[PASS] required scripts found")
PY
```

---

# 2. Dataset layout

## 2.1. Raw Kvasir-SEG

```text
dataset/Kvasir-SEG/
├── images/
├── masks/
└── kavsir_bboxes.json
```

## 2.2. Prepared Kvasir-SEG

```text
dataset/Kvasir-SEG-YOLO26/
├── images/
│   ├── target/
│   ├── train/
│   └── val/
├── labels/
│   ├── target/
│   ├── train/
│   └── val/
├── gt_masks/
│   ├── target/
│   ├── train/
│   └── val/
├── split_train.txt
├── split_val.txt
└── dataset_seg.yaml
```

## 2.3. Raw CVC-ClinicDB

```text
dataset/CVC-ClinicDB/
├── PNG/
|   ├── Original/
|   └── Ground Truth/
├── TIF/
|   ├── Original/
|   └── Ground Truth/
├── class_dict.csv
└── metadata.csv
```

## 2.4. Prepared CVC-ClinicDB

```text
dataset/CVC-ClinicDB-YOLO26/
├── images/
│   ├── target/
│   ├── train/
│   └── val/
├── labels/
│   ├── target/
│   ├── train/
│   └── val/
├── gt_masks/
│   ├── target/
│   ├── train/
│   └── val/
├── split_train.txt
├── split_val.txt
└── dataset_seg.yaml
```

---

# 3. Prepare Kvasir-SEG

Skip this section if `dataset/Kvasir-SEG-YOLO26/` already contains the canonical prepared split.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"

python scripts/YOLO26/medseg/prepare_kvasir.py \
  --src dataset/Kvasir-SEG \
  --dst dataset/Kvasir-SEG-YOLO26 \
  --val-fraction 0.20 \
  --seed 29 \
  --min-contour-area 15 \
  --overwrite
```

Check:

```bash
find dataset/Kvasir-SEG-YOLO26/images/train -type f | wc -l
find dataset/Kvasir-SEG-YOLO26/images/val   -type f | wc -l
find dataset/Kvasir-SEG-YOLO26/labels/train -type f | wc -l
find dataset/Kvasir-SEG-YOLO26/labels/val   -type f | wc -l
```

Expected canonical split:

```text
train = 800
val   = 200
seed  = 29
```

---

# 4. YOLO26-S-Seg pre-flight

```bash
python scripts/YOLO26/medseg/check_yolo26s_seg.py \
  --weights yolo26s-seg.pt \
  --data dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml
```

Expected:

```text
model   : SegmentationModel
head    : Segment26
end2end : True
nm      : 32
npr     : 128
[PASS] YOLO26-S-Seg ready
```

--

## Evaluate base line on 2 datasets:

Evaluate on Kvasir
```bash
python scripts/YOLO26/medseg/eval_source_seg.py \
  --model yolo26s-seg.pt \
  --data dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml \
  --images dataset/Kvasir-SEG-YOLO26/images/target \
  --gt-masks dataset/Kvasir-SEG-YOLO26/gt_masks/target \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --conf 0.25 \
  --method "YOLO26-Seg-S Baseline (Kvasir)" \
  --warmup 100 \
  --iters 500 \
  --precision fp32 \
  --out runs/seg/benchmarks/yolo26s_seg_baseline_kvasir_metrics.json \
  2>&1 | tee logs/seg/benchmarks/yolo26s_seg_baseline_kvasir_eval.log
```

Evaluate on CVC-ClinicDB:

```bash
python scripts/YOLO26/medseg/eval_source_seg.py \
  --model yolo26s-seg.pt \
  --data dataset/CVC-ClinicDB-YOLO26/dataset_seg.yaml \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --conf 0.25 \
  --method "YOLO26-Seg-S Baseline (CVC-ClinicDB)" \
  --warmup 100 \
  --iters 500 \
  --precision fp32 \
  --out runs/seg/benchmarks/yolo26s_seg_baseline_cvc_metrics.json \
  2>&1 | tee logs/seg/benchmarks/yolo26s_seg_baseline_cvc_eval.log
```

---

# 5. Train source model

Supervised on CVC-ClinicDB
```bash
mkdir -p runs/seg/source logs/seg/source

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/train_source_seg.py \
  --weights yolo26s-seg.pt \
  --data dataset/CVC-ClinicDB-YOLO26/dataset_seg.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --device 0 \
  --seed 29 \
  --project runs/seg/source \
  --name cvc_yolo26s_seg \
> logs/seg/source/cvc_yolo26s_seg.log 2>&1 &

echo "PID=$!"
```


Supervised on Kvasir
```bash
mkdir -p runs/seg/source logs/seg/source

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/train_source_seg.py \
  --weights yolo26s-seg.pt \
  --data dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --device 0 \
  --seed 29 \
  --project runs/seg/source \
  --name kvasir_yolo26s_seg \
> logs/seg/source/kvasir_yolo26s_seg.log 2>&1 &

echo "PID=$!"
```

---

# 6. Domain-shift validation

Kvasir supversied -> test on CVC-ClinicDB
```bash
mkdir -p runs/seg/benchmarks logs/seg/benchmarks

python scripts/YOLO26/medseg/eval_source_seg.py \
  --model runs/seg/source/kvasir_yolo26s_seg/weights/best.pt \
  --data dataset/CVC-ClinicDB-YOLO26/dataset_seg.yaml \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target\
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --conf 0.25 \
  --method "YOLO26-Seg-S Source (Supervised on Kvasir -> Test on CVC)" \
  --warmup 100 \
  --iters 500 \
  --precision fp32 \
  --out runs/seg/benchmarks/k2c_yolo26s_seg_source_metrics.json \
  2>&1 | tee logs/seg/benchmarks/k2c_yolo26s_seg_supervised_eval.log
```

CVC-ClinicDB supversied -> test on Kvasir
```bash
mkdir -p runs/seg/benchmarks logs/seg/benchmarks

python scripts/YOLO26/medseg/eval_source_seg.py \
  --model runs/seg/source/cvc_yolo26s_seg/weights/best.pt \
  --data dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml \
  --images dataset/Kvasir-SEG-YOLO26/images/target \
  --gt-masks dataset/Kvasir-SEG-YOLO26/gt_masks/target \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --conf 0.25 \
  --method "YOLO26-Seg-S Source (Supervised on CVC -> Test on Kvasir)" \
  --warmup 100 \
  --iters 500 \
  --precision fp32 \
  --out runs/seg/benchmarks/c2k_yolo26s_seg_supervised_metrics.json \
  2>&1 | tee logs/seg/benchmarks/c2k_yolo26s_seg_supervised_eval.log
```

---

# 7. Freeze source checkpoint

```bash
mkdir -p runs/seg/source/frozen

cp \
  runs/seg/source/kvasir_yolo26s_seg/weights/best.pt \
  runs/seg/source/frozen/kvasir_yolo26s_seg_source_best.pt

cp \
  dataset/Kvasir-SEG-YOLO26/split_train.txt \
  runs/seg/source/frozen/

cp \
  dataset/Kvasir-SEG-YOLO26/split_val.txt \
  runs/seg/source/frozen/

sha256sum \
  runs/seg/source/frozen/kvasir_yolo26s_seg_source_best.pt \
  | tee runs/seg/source/frozen/kvasir_yolo26s_seg_source_best.sha256
```

Canonical source checkpoint:

```text
runs/seg/source/frozen/kvasir_yolo26s_seg_source_best.pt
```

If CVC -> Kvasir:
```bash
mkdir -p runs/seg/source/frozen

cp \
  runs/seg/source/cvc_yolo26s_seg/weights/best.pt \
  runs/seg/source/frozen/cvc_yolo26s_seg_source_best.pt

cp \
  dataset/CVC-ClinicDB-YOLO26/split_train.txt \
  runs/seg/source/frozen/

cp \
  dataset/CVC-ClinicDB-YOLO26/split_val.txt \
  runs/seg/source/frozen/

sha256sum \
  runs/seg/source/frozen/cvc_yolo26s_seg_source_best.pt \
  | tee runs/seg/source/frozen/cvc_yolo26s_seg_source_best.sha256
```

---

# 8. Prepare CVC-ClinicDB target

```bash
python scripts/YOLO26/medseg/prepare_cvc_clinicdb.py \
  --src dataset/CVC-ClinicDB/PNG \
  --dst dataset/CVC-ClinicDB-YOLO26 \
  --min-contour-area 15 \
  --overwrite
```
Check:

```bash
find dataset/CVC-ClinicDB-YOLO26/images/target -type f | wc -l
find dataset/CVC-ClinicDB-YOLO26/labels/target -type f | wc -l
find dataset/CVC-ClinicDB-YOLO26/gt_masks/target -type f | wc -l
```

Expected:

```text
images   = 612
labels   = 612
gt_masks = 612
```

If we train CVC-ClinicDB -> Kvasir. We run this and arrange the folder later.
```bash
python scripts/YOLO26/medseg/prepare_cvc_to_kvasir_reverse.py   
  --cvc-raw dataset/CVC-ClinicDB/PNG   
  --kvasir-raw dataset/Kvasir-SEG   
  --cvc-source-out dataset/CVC-ClinicDB-YOLO26-SOURCE   
  --kvasir-target-out dataset/Kvasir-SEG-YOLO26-TARGET   
  --val-fraction 0.20   
  --seed 29   
  --min-contour-area 15   
  --overwrite
```

---

# 10. Stage 1 — AdaBN on target images only


K2C
```bash
mkdir -p runs/seg/stage1/cvc_adabn logs/seg/stage1

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/stage1_adabn_seg.py \
  --weights runs/seg/source/frozen/kvasir_yolo26s_seg_source_best.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --out-dir runs/seg/stage1/cvc_adabn \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --epochs 2 \
  --device 0 \
  --seed 29 \
> logs/seg/stage1/cvc_adabn.log 2>&1 &

echo "PID=$!"
```

C2K
```bash
mkdir -p runs/seg/stage1/kvasir_adabn logs/seg/stage1

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/stage1_adabn_seg.py \
  --weights runs/seg/source/frozen/cvc_yolo26s_seg_source_best.pt \
  --target-images dataset/Kvasir-SEG-YOLO26/images/target \
  --out-dir runs/seg/stage1/kvasir_adabn \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --epochs 2 \
  --device 0 \
  --seed 29 \
> logs/seg/stage1/kvasir_adabn.log 2>&1 &
```

---

# 11. Evaluate AdaBN checkpoint

K2C
```bash
python scripts/YOLO26/medseg/eval_source_seg.py \
  --model runs/seg/stage1/cvc_adabn/yolo26s_seg_cvc_adabn.pt \
  --data dataset/CVC-ClinicDB-YOLO26/dataset_seg.yaml \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --conf 0.25 \
  --out runs/seg/benchmarks/cvc_clinicdb_adabn_metrics.json \
2>&1 | tee logs/seg/benchmarks/cvc_clinicdb_adabn_eval.log
```

C2K
```bash
python scripts/YOLO26/medseg/eval_source_seg.py \
  --model runs/seg/stage1/kvasir_adabn/yolo26s_seg_kvasir_adabn.pt \
  --data dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml \
  --images dataset/Kvasir-SEG-YOLO26/images/target \
  --gt-masks dataset/Kvasir-SEG-YOLO26/gt_masks/target \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --conf 0.25 \
  --out runs/seg/benchmarks/kvasir_adabn_metrics.json \
2>&1 | tee logs/seg/benchmarks/kvasir_adabn_eval.log
```

---

# 12. Freeze AdaBN checkpoint

K2C
```bash
mkdir -p runs/seg/stage1/frozen

cp \
  runs/seg/stage1/cvc_adabn/yolo26s_seg_cvc_adabn.pt \
  runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt

sha256sum \
  runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
  | tee runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.sha256
```

C2K
```bash
mkdir -p runs/seg/stage1/frozen

cp \
  runs/seg/stage1/kvasir_adabn/yolo26s_seg_kvasir_adabn.pt \
  runs/seg/stage1/frozen/yolo26s_seg_kvasir_adabn.pt

sha256sum \
  runs/seg/stage1/frozen/yolo26s_seg_kvasir_adabn.pt \
  | tee runs/seg/stage1/frozen/yolo26s_seg_kvasir_adabn.sha256
```

---

# 13. Stage-2 dual-head segmentation audit

K2C
```bash
mkdir -p runs/seg/dense_sfseg logs/seg/dense_sfseg

python scripts/YOLO26/medseg/audit_stage2_seg.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --device 0 \
2>&1 | tee logs/seg/dense_sfseg/stage2_seg_pseudolabel_audit_k2c.log
```

C2K
```bash
mkdir -p runs/seg/dense_sfseg logs/seg/dense_sfseg

python scripts/YOLO26/medseg/audit_stage2_seg.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_kvasir_adabn.pt \
  --target-images dataset/Kvasir-SEG-YOLO26/images/target \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --device 0 \
2>&1 | tee logs/seg/dense_sfseg/stage2_seg_pseudolabel_audit_c2k.log
```

---

# 14. Full-target Mask-DHF audit

K2C
```bash
python scripts/YOLO26/medseg/audit_mask_dhf_seg.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --tau-o2o 0.5 \
  --tau-o2m 0.5 \
  --tau-no 0.2 \
  --tau-dup 0.7 \
  --mask-thr 0.5 \
  --mask-rel-thr 0.65 \
  --mask-no 0.20 \
  --min-mask-pixels 16 \
  --device 0 \
  --seed 29 \
  --out runs/seg/dense_sfseg/mask_dhf_audit_kc2.json \
2>&1 | tee logs/seg/dense_sfseg/mask_dhf_audit_k2c.log
```

C2K
```bash
python scripts/YOLO26/medseg/audit_mask_dhf_seg.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_kvasir_adabn.pt \
  --target-images dataset/Kvasir-SEG-YOLO26/images/target \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --tau-o2o 0.5 \
  --tau-o2m 0.5 \
  --tau-no 0.2 \
  --tau-dup 0.7 \
  --mask-thr 0.5 \
  --mask-rel-thr 0.65 \
  --mask-no 0.20 \
  --min-mask-pixels 16 \
  --device 0 \
  --seed 29 \
  --out runs/seg/dense_sfseg/mask_dhf_audit_c2k.json \
2>&1 | tee logs/seg/dense_sfseg/mask_dhf_audit_c2k.log
```

---

# 15. Mask-stability audit on Box-DHF extras

K2C
```bash
python scripts/YOLO26/medseg/audit_mask_stability_extras.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --device 0 \
  --tau-o2o 0.5 \
  --tau-o2m 0.5 \
  --tau-no 0.2 \
  --tau-dup 0.7 \
  --mask-thr 0.5 \
  --stability-low 0.40 \
  --stability-high 0.60 \
  --min-mask-pixels 16 \
  --out runs/seg/dense_sfseg/mask_stability_extras_audit.json \
2>&1 | tee logs/seg/dense_sfseg/mask_stability_extras_audit_k2c.log
```

C2K
```bash
python scripts/YOLO26/medseg/audit_mask_stability_extras.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_kvasir_adabn.pt \
  --target-images dataset/Kvasir-SEG-YOLO26/images/target \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --device 0 \
  --tau-o2o 0.5 \
  --tau-o2m 0.5 \
  --tau-no 0.2 \
  --tau-dup 0.7 \
  --mask-thr 0.5 \
  --stability-low 0.40 \
  --stability-high 0.60 \
  --min-mask-pixels 16 \
  --out runs/seg/dense_sfseg/mask_stability_extras_audit.json \
2>&1 | tee logs/seg/dense_sfseg/mask_stability_extras_audit_c2k.log
```

---

# 16. Dense MedRT-SFSeg — 60 epochs

## 16a. Training

```bash
mkdir -p \
  runs/seg/dense_sfseg/cvc_dense_60ep \
  logs/seg/dense_sfseg

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/stage2_dense_sfseg.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --out-dir runs/seg/dense_sfseg/cvc_dense_60ep \
  --imgsz 640 \
  --batch 4 \
  --workers 4 \
  --epochs 60 \
  --lr 1e-4 \
  --grad-clip 10 \
  --ema 0.999 \
  --tau-o2o 0.5 \
  --tau-o2m 0.5 \
  --tau-no 0.2 \
  --tau-dup 0.7 \
  --mask-thr 0.5 \
  --stability-low 0.40 \
  --stability-high 0.60 \
  --mask-rel-thr 0.744898 \
  --min-mask-pixels 16 \
  --mard-lambda0 0.05 \
  --mard-lambda-max 0.2 \
  --mard-warmup-epochs 5 \
  --mard-gamma 1.0 \
  --mard-alpha 1.0 \
  --mard-beta 0.1 \
  --mard-topk-boxes 15 \
  --mard-fg-points 8 \
  --mard-bg-points 128 \
  --mard-eta 12 \
  --mard-box-conf 0.5 \
  --device 0 \
  --seed 29 \
  --print-freq 20 \
  --save-interval 10 \
> logs/seg/dense_sfseg/cvc_dense_60ep.log 2>&1 &

echo "PID=$!"
```

Monitor:

```bash
tail -f logs/seg/dense_sfseg/cvc_dense_60ep.log
```

Final Dense checkpoint:

```text
runs/seg/dense_sfseg/cvc_dense_60ep/checkpoints/dense_sfseg_epoch_60.pt
```

## 16b. Evaluate Dense MedRT-SFSeg

```bash
nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/eval_source_seg.py \
  --model runs/seg/dense_sfseg/cvc_dense_60ep/checkpoints/dense_sfseg_epoch_60.pt \
  --data dataset/CVC-ClinicDB-YOLO26/dataset_seg.yaml \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --conf 0.25 \
  --out runs/seg/benchmarks/cvc_clinicdb_dense_sfseg_epoch60_metrics.json \
> logs/seg/benchmarks/cvc_clinicdb_dense_sfseg_epoch60_eval.log 2>&1 &

echo "PID=$!"
```

Monitor:

```bash
tail -f logs/seg/benchmarks/cvc_clinicdb_dense_sfseg_epoch60_eval.log
```

---

# 17. Mask-DHF + SegMARD-v2 - 60 epochs (best)

## 17a. Training

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"

mkdir -p \
  runs/seg/dense_sfseg/cvc_segmard_v2_60ep \
  logs/seg/dense_sfseg

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/stage2_dense_sfseg.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --out-dir runs/seg/dense_sfseg/cvc_segmard_v2_60ep \
  --imgsz 640 \
  --batch 4 \
  --workers 4 \
  --epochs 60 \
  --lr 1e-4 \
  --grad-clip 10 \
  --ema 0.999 \
  --tau-o2o 0.5 \
  --tau-o2m 0.5 \
  --tau-no 0.2 \
  --tau-dup 0.7 \
  --mask-thr 0.5 \
  --stability-low 0.40 \
  --stability-high 0.60 \
  --mask-rel-thr 0.744898 \
  --min-mask-pixels 16 \
  --mard-lambda0 0.05 \
  --mard-lambda-max 0.2 \
  --mard-warmup-epochs 5 \
  --mard-gate-threshold 0.5 \
  --mard-gamma 1.0 \
  --mard-alpha 1.0 \
  --mard-beta 0.1 \
  --mard-topk-boxes 15 \
  --mard-fg-points 8 \
  --mard-bg-points 128 \
  --mard-eta 12 \
  --mard-box-conf 0.5 \
  --mard-mode mask \
  --segmard-erode-kernel 1 \
  --segmard-dilate-kernel 1 \
  --segmard-hard-bg-ratio 0.5 \
  --device 0 \
  --seed 29 \
  --print-freq 20 \
  --save-interval 10 \
> logs/seg/dense_sfseg/cvc_segmard_v2_60ep.log 2>&1 &

echo "PID=$!"
```

## 17b. Evaluation

```bash
nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/eval_source_seg.py \
  --model runs/seg/dense_sfseg/cvc_segmard_v2_60ep/checkpoints/dense_sfseg_epoch_60.pt \
  --data dataset/CVC-ClinicDB-YOLO26/dataset_seg.yaml \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --conf 0.25 \
  --out runs/seg/benchmarks/cvc_clinicdb_segmard_v2_metrics.json \
> logs/seg/dense_sfseg/cvc_segmard_v2_eval.log 2>&1 &

echo "PID=$!"
```
---

# 18. BDL-v1 + SegMARD-v2

## 18a. Training

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"

mkdir -p \
  runs/seg/dense_sfseg/cvc_bdl_v1_segmard_v2_60ep \
  logs/seg/dense_sfseg

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/stage2_dense_sfseg.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --out-dir runs/seg/dense_sfseg/cvc_bdl_v1_segmard_v2_60ep \
  --imgsz 640 \
  --batch 4 \
  --workers 4 \
  --epochs 60 \
  --lr 1e-4 \
  --grad-clip 10 \
  --ema 0.999 \
  --tau-o2o 0.5 \
  --tau-o2m 0.5 \
  --tau-no 0.2 \
  --tau-dup 0.7 \
  --mask-thr 0.5 \
  --stability-low 0.40 \
  --stability-high 0.60 \
  --mask-rel-thr 0.744898 \
  --min-mask-pixels 16 \
  --dhf-mode bdl \
  --bdl-tau-match 0.5 \
  --bdl-max-witnesses 5 \
  --bdl-boundary-kernel 3 \
  --mard-lambda0 0.05 \
  --mard-lambda-max 0.2 \
  --mard-warmup-epochs 5 \
  --mard-gate-threshold 0.5 \
  --mard-gamma 1.0 \
  --mard-alpha 1.0 \
  --mard-beta 0.1 \
  --mard-topk-boxes 15 \
  --mard-fg-points 8 \
  --mard-bg-points 128 \
  --mard-eta 12 \
  --mard-box-conf 0.5 \
  --mard-mode mask \
  --segmard-erode-kernel 1 \
  --segmard-dilate-kernel 1 \
  --segmard-hard-bg-ratio 0.5 \
  --device 0 \
  --seed 29 \
  --print-freq 20 \
  --save-interval 10 \
> logs/seg/dense_sfseg/cvc_bdl_v1_segmard_v2_60ep.log 2>&1 &

echo "PID=$!"
```

## 18b. Evaluation

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"

mkdir -p \
  runs/seg/benchmarks \
  logs/seg/benchmarks

python scripts/YOLO26/medseg/eval_source_seg.py \
  --model runs/seg/dense_sfseg/cvc_bdl_v1_segmard_v2_60ep/checkpoints/dense_sfseg_epoch_60.pt \
  --data dataset/CVC-ClinicDB-YOLO26/dataset_seg.yaml \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --conf 0.25 \
  --out runs/seg/benchmarks/cvc_clinicdb_bdl_v1_segmard_v2_metrics.json \
2>&1 | tee logs/seg/benchmarks/cvc_bdl_v1_segmard_v2_eval.log
```

## 18c. Analysis BDL-v1 with ASD & HD95

```bash
python scripts/YOLO26/medseg/analyze_seg_results.py \
  --model runs/seg/dense_sfseg/cvc_bdl_v1_segmard_v2_60ep/checkpoints/dense_sfseg_epoch_60.pt \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --out-dir runs/seg/analysis/cvc_bdl_v1_segmard_v2 \
  --imgsz 640 \
  --device 0 \
  --conf 0.25 \
  --topk 20 \
  --seed 29 \
2>&1 | tee logs/seg/benchmarks/cvc_bdl_v1_analysis.log
```

if you want to analysis on SegMARD-v2 baseline
```
python scripts/YOLO26/medseg/analyze_seg_results.py \
  --model runs/seg/dense_sfseg/cvc_segmard_v2_60ep/checkpoints/dense_sfseg_epoch_60.pt \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --out-dir runs/seg/analysis/cvc_segmard_v2 \
  --imgsz 640 \
  --device 0 \
  --conf 0.25 \
  --topk 20 \
  --seed 29 \
2>&1 | tee logs/seg/benchmarks/cvc_segmard_v2_boundary_analysis.log
```

---

# 19. Main DURR-v1 + SegMARD-v2 — 60 epochs

K2C
```bash
mkdir -p runs/seg/dense_sfseg logs/seg/dense_sfseg

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/stage2_dense_sfseg_durr.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --out-dir runs/seg/dense_sfseg/cvc_durr_v1_segmard_v2_60ep \
  --imgsz 640 \
  --batch 4 \
  --workers 4 \
  --epochs 60 \
  --lr 1e-4 \
  --device 0 \
  --seed 29 \
  --tau-o2o 0.5 \
  --tau-o2m 0.5 \
  --tau-no 0.2 \
  --tau-dup 0.7 \
  --mask-thr 0.5 \
  --stability-low 0.40 \
  --stability-high 0.60 \
  --mask-rel-thr 0.744898 \
  --min-mask-pixels 16 \
  --durr-tau-match 0.5 \
  --durr-max-witnesses 5 \
  --durr-boundary-kernel 5 \
  --durr-route-gain 1.0 \
  --durr-min-disagreement 0.02 \
  --durr-rescue-conf 0.80 \
  --durr-rescue-stability 0.80 \
  --durr-rescue-consensus-iou 0.70 \
  --durr-rescue-min-support 0 \
  --durr-evidence-conf 0.10 \
  --durr-safe-bg-teacher-prob 0.10 \
  --durr-hall-student-thr 0.80 \
  --durr-hall-area-thr 0.10 \
  --durr-hall-area-weight 0.25 \
  --durr-lambda-dir 0.10 \
  --durr-lambda-rescue 0.20 \
  --durr-lambda-hall 0.10 \
  --durr-warmup-epochs 5 \
  --mard-lambda0 0.05 \
  --mard-lambda-max 0.20 \
  --mard-gamma 1.0 \
  --mard-alpha 1.0 \
  --mard-beta 0.1 \
  --mard-warmup-epochs 5 \
  --mard-gate-threshold 0.5 \
  --mard-topk-boxes 15 \
  --mard-fg-points 8 \
  --mard-bg-points 128 \
  --mard-eta 12 \
  --mard-box-conf 0.5 \
  --segmard-erode-kernel 1 \
  --segmard-dilate-kernel 1 \
  --segmard-hard-bg-ratio 0.5 \
  --save-interval 10 \
  --analysis-gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --analysis-data-yaml dataset/CVC-ClinicDB-YOLO26/dataset_seg.yaml \
  --analysis-batch 8 \
  --analysis-conf 0.25 \
  --analysis-max-images 0 \
> logs/seg/dense_sfseg/cvc_durr_v1_segmard_v2_60ep.log 2>&1 &

echo "PID=$!"
```

C2K
```bash
mkdir -p runs/seg/dense_sfseg logs/seg/dense_sfseg

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/stage2_dense_sfseg_durr.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_kvasir_adabn.pt \
  --target-images dataset/Kvasir-SEG-YOLO26/images/target \
  --out-dir runs/seg/dense_sfseg/kvasir_durr_v1_segmard_v2_60ep \
  --imgsz 640 \
  --batch 4 \
  --workers 4 \
  --epochs 60 \
  --lr 1e-4 \
  --device 0 \
  --seed 29 \
  --tau-o2o 0.5 \
  --tau-o2m 0.5 \
  --tau-no 0.2 \
  --tau-dup 0.7 \
  --mask-thr 0.5 \
  --stability-low 0.40 \
  --stability-high 0.60 \
  --mask-rel-thr 0.753071 \
  --min-mask-pixels 16 \
  --durr-tau-match 0.5 \
  --durr-max-witnesses 5 \
  --durr-boundary-kernel 5 \
  --durr-route-gain 1.0 \
  --durr-min-disagreement 0.02 \
  --durr-rescue-conf 0.80 \
  --durr-rescue-stability 0.80 \
  --durr-rescue-consensus-iou 0.70 \
  --durr-rescue-min-support 0 \
  --durr-evidence-conf 0.10 \
  --durr-safe-bg-teacher-prob 0.10 \
  --durr-hall-student-thr 0.80 \
  --durr-hall-area-thr 0.10 \
  --durr-hall-area-weight 0.25 \
  --durr-lambda-dir 0.10 \
  --durr-lambda-rescue 0.20 \
  --durr-lambda-hall 0.10 \
  --durr-warmup-epochs 5 \
  --mard-lambda0 0.05 \
  --mard-lambda-max 0.20 \
  --mard-gamma 1.0 \
  --mard-alpha 1.0 \
  --mard-beta 0.1 \
  --mard-warmup-epochs 5 \
  --mard-gate-threshold 0.5 \
  --mard-topk-boxes 15 \
  --mard-fg-points 8 \
  --mard-bg-points 128 \
  --mard-eta 12 \
  --mard-box-conf 0.5 \
  --segmard-erode-kernel 1 \
  --segmard-dilate-kernel 1 \
  --segmard-hard-bg-ratio 0.5 \
  --save-interval 10 \
  --analysis-gt-masks dataset/Kvasir-SEG-YOLO26/gt_masks/target \
  --analysis-data-yaml dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml \
  --analysis-batch 8 \
  --analysis-conf 0.25 \
  --analysis-max-images 0 \
> logs/seg/dense_sfseg/cvc_kvasir_v1_segmard_v2_60ep.log 2>&1 &

echo "PID=$!"
```


---

# 21. DURR checkpoints

Expected:

```text
runs/seg/dense_sfseg/cvc_durr_v1_segmard_v2_60ep/checkpoints/
├── durr_student_epoch_10.pt
├── durr_teacher_ema_epoch_10.pt
├── ...
├── durr_student_epoch_60.pt
└── durr_teacher_ema_epoch_60.pt
```

Final Student:

```text
runs/seg/dense_sfseg/cvc_durr_v1_segmard_v2_60ep/checkpoints/durr_student_epoch_60.pt
```

Final EMA Teacher:

```text
runs/seg/dense_sfseg/cvc_durr_v1_segmard_v2_60ep/checkpoints/durr_teacher_ema_epoch_60.pt
```

Metadata:

```text
runs/seg/dense_sfseg/cvc_durr_v1_segmard_v2_60ep/stage2_metadata.json
```

---

# 22. Automatic Teacher + Student visualization

Nếu main command có:

```text
--analysis-gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target
```

thì **không cần chạy thêm visualization script**.

Expected:

```text
runs/seg/dense_sfseg/cvc_durr_v1_segmard_v2_60ep/final_analysis/
├── panels/
├── per_image_trace.csv
└── summary.json
```

Panel:

```text
Image
GT
Initial Teacher O2O
Initial Teacher fused
Final EMA Teacher fused
Final Teacher O2M witnesses
Final signed delta
Final O2M rescue
Final Student prediction
Absolute Student error
```

GT chỉ được load sau training.

---

# 23. Official final Student evaluation

K2C
```bash
mkdir -p \
  runs/seg/benchmarks/cvc_durr_v1_final \
  logs/seg/benchmarks

python scripts/YOLO26/medseg/evaluate_durr_final.py \
  --model runs/seg/dense_sfseg/cvc_durr_v1_segmard_v2_60ep/checkpoints/durr_student_epoch_60.pt \
  --data dataset/CVC-ClinicDB-YOLO26/dataset_seg.yaml \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --benchmark-image dataset/CVC-ClinicDB-YOLO26/images/target \
  --out-dir runs/seg/benchmarks/cvc_durr_v1_final \
  --imgsz 640 \
  --eval-batch 8 \
  --device 0 \
  --conf 0.25 \
  --warmup 100 \
  --iters 500 \
  --precision fp32 \
2>&1 | tee logs/seg/benchmarks/cvc_durr_v1_final.log
```

C2K
```bash
mkdir -p \
  runs/seg/benchmarks/kvasir_durr_v1_final \
  logs/seg/benchmarks

python scripts/YOLO26/medseg/evaluate_durr_final.py \
  --model runs/seg/dense_sfseg/kvasir_durr_v1_segmard_v2_60ep/checkpoints/durr_student_epoch_60.pt \
  --data dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml \
  --images dataset/Kvasir-SEG-YOLO26/images/target \
  --gt-masks dataset/Kvasir-SEG-YOLO26/gt_masks/target \
  --out-dir runs/seg/benchmarks/kvasir_durr_v1_final \
  --imgsz 640 \
  --eval-batch 8 \
  --device 0 \
  --conf 0.25 \
  --warmup 100 \
  --iters 500 \
  --precision fp32 \
  2>&1 | tee logs/seg/benchmarks/kvasir_durr_v1_final.log
```

---

# 30. Freeze final DURR artifacts

Sau official evaluation:

```bash
mkdir -p runs/seg/dense_sfseg/frozen

cp \
  runs/seg/dense_sfseg/cvc_durr_v1_segmard_v2_60ep/checkpoints/durr_student_epoch_60.pt \
  runs/seg/dense_sfseg/frozen/cvc_durr_v1_student_epoch60.pt

cp \
  runs/seg/dense_sfseg/cvc_durr_v1_segmard_v2_60ep/checkpoints/durr_teacher_ema_epoch_60.pt \
  runs/seg/dense_sfseg/frozen/cvc_durr_v1_teacher_ema_epoch60.pt

cp \
  runs/seg/dense_sfseg/cvc_durr_v1_segmard_v2_60ep/stage2_metadata.json \
  runs/seg/dense_sfseg/frozen/

sha256sum \
  runs/seg/dense_sfseg/frozen/cvc_durr_v1_student_epoch60.pt \
  | tee runs/seg/dense_sfseg/frozen/cvc_durr_v1_student_epoch60.sha256
```
