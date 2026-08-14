# MedRT-SFOD — Medical Source-Free Segmentation Pipeline

> Repository root: `~/MedRT-SFOD`  
> Source: **Kvasir-SEG**  
> Target: **CVC-ClinicDB**  
> Model: **YOLO26-S-Seg**  
> Main adaptation: **AdaBN → Mean Teacher → Mask-DHF → MARD**  
> Compression: **RASP-SFSeg → physical compaction**

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
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── gt_masks/
│   ├── train/
│   └── val/
├── split_train.txt
├── split_val.txt
└── dataset_seg.yaml
```

## 2.3. Raw CVC-ClinicDB

```text
dataset/CVC-ClinicDB/
└── PNG/
    ├── Original/
    └── Ground Truth/
```

## 2.4. Prepared CVC-ClinicDB

```text
dataset/CVC-ClinicDB-YOLO26/
├── images/
│   └── target/
├── labels/
│   └── target/
├── gt_masks/
│   └── target/
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

---

# 5. Train source model — Kvasir-SEG

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

Monitor:

```bash
tail -f logs/seg/source/kvasir_yolo26s_seg.log
```

Source checkpoint:

```text
runs/seg/source/kvasir_yolo26s_seg/weights/best.pt
```

---

# 6. Source validation — Kvasir-SEG

```bash
mkdir -p runs/seg/benchmarks logs/seg/benchmarks

python scripts/YOLO26/medseg/eval_source_seg.py \
  --model runs/seg/source/kvasir_yolo26s_seg/weights/best.pt \
  --data dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml \
  --images dataset/Kvasir-SEG-YOLO26/images/val \
  --gt-masks dataset/Kvasir-SEG-YOLO26/gt_masks/val \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --conf 0.25 \
  --out runs/seg/benchmarks/kvasir_yolo26s_seg_source_metrics.json \
2>&1 | tee logs/seg/benchmarks/kvasir_yolo26s_seg_source_eval.log
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

---

# 9. Source-only target baseline — CVC-ClinicDB

```bash
python scripts/YOLO26/medseg/eval_source_seg.py \
  --model runs/seg/source/frozen/kvasir_yolo26s_seg_source_best.pt \
  --data dataset/CVC-ClinicDB-YOLO26/dataset_seg.yaml \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --conf 0.25 \
  --out runs/seg/benchmarks/cvc_clinicdb_source_only_metrics.json \
2>&1 | tee logs/seg/benchmarks/cvc_clinicdb_source_only.log
```

---

# 10. Stage 1 — AdaBN on target images only

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

Monitor:

```bash
tail -f logs/seg/stage1/cvc_adabn.log
```

Expected checkpoint:

```text
runs/seg/stage1/cvc_adabn/yolo26s_seg_cvc_adabn.pt
```

---

# 11. Evaluate AdaBN checkpoint

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

---

# 12. Freeze AdaBN checkpoint

```bash
mkdir -p runs/seg/stage1/frozen

cp \
  runs/seg/stage1/cvc_adabn/yolo26s_seg_cvc_adabn.pt \
  runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt

sha256sum \
  runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
  | tee runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.sha256
```

Canonical Stage-2 initialization:

```text
runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt
```

---

# 13. Stage-2 dual-head segmentation audit

```bash
mkdir -p runs/seg/dense_sfseg logs/seg/dense_sfseg

python scripts/YOLO26/medseg/audit_stage2_seg.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --device 0 \
2>&1 | tee logs/seg/dense_sfseg/stage2_seg_pseudolabel_audit.log
```

---

# 14. Full-target Mask-DHF audit

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
  --out runs/seg/dense_sfseg/mask_dhf_audit.json \
2>&1 | tee logs/seg/dense_sfseg/mask_dhf_audit.log
```

---

# 15. Mask-stability audit on Box-DHF extras

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
2>&1 | tee logs/seg/dense_sfseg/mask_stability_extras_audit.log
```

Canonical Mask-DHF reliability threshold:

```text
0.744898
```

---

# 16. Dense MedRT-SFSeg — 60 epochs

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

---

# 17. Evaluate Dense MedRT-SFSeg

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

# 18. Freeze Dense model

```bash
mkdir -p runs/seg/dense_sfseg/frozen

cp \
  runs/seg/dense_sfseg/cvc_dense_60ep/checkpoints/dense_sfseg_epoch_60.pt \
  runs/seg/dense_sfseg/frozen/cvc_dense_sfseg_epoch60.pt

cp \
  runs/seg/dense_sfseg/cvc_dense_60ep/stage2_metadata.json \
  runs/seg/dense_sfseg/frozen/

cp \
  runs/seg/benchmarks/cvc_clinicdb_dense_sfseg_epoch60_metrics.json \
  runs/seg/dense_sfseg/frozen/

sha256sum \
  runs/seg/dense_sfseg/frozen/cvc_dense_sfseg_epoch60.pt \
  | tee runs/seg/dense_sfseg/frozen/cvc_dense_sfseg_epoch60.sha256
```

Canonical Dense checkpoint:

```text
runs/seg/dense_sfseg/frozen/cvc_dense_sfseg_epoch60.pt
```

---

# 19. RASP-SFSeg structural audit

```bash
mkdir -p runs/seg/rasp_sfseg logs/seg/rasp_sfseg

python scripts/YOLO26/medseg/inspect_rasp_prunable_groups_seg.py \
  --model runs/seg/dense_sfseg/frozen/cvc_dense_sfseg_epoch60.pt \
  --device 0 \
  --imgsz 640 \
  --audit-imgsz 256 \
  --min-hidden 16 \
  --min-keep-ratio 0.50 \
  --round-to 8 \
  --depgraph \
  --out runs/seg/rasp_sfseg/rasp_seg_structural_audit.json \
2>&1 | tee logs/seg/rasp_sfseg/rasp_seg_structural_audit.log
```

Expected canonical audit:

```text
eligible groups   = 13
eligible channels = 880
DepGraph safe     = 13/13
```

---

# 20. RASP-SFSeg — 60 epochs

RASP main run starts from the same AdaBN checkpoint as the Dense main run.

```bash
mkdir -p \
  runs/seg/rasp_sfseg/cvc_rasp_60ep \
  logs/seg/rasp_sfseg

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/stage3_rasp_sfseg.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --out-dir runs/seg/rasp_sfseg/cvc_rasp_60ep \
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
  --rasp_enable \
  --rasp_warmup_epochs 5 \
  --rasp_cycle_epochs 3 \
  --rasp_reliability_threshold 0.50 \
  --rasp_importance_beta 0.90 \
  --rasp_min_hidden 16 \
  --rasp_min_keep_ratio 0.50 \
  --rasp_round_to 8 \
  --rasp_gmm_posterior 0.80 \
  --rasp_gmm_bic_gain 0 \
  --rasp_gmm_min_separation 1.0 \
  --rasp_gmm_min_samples 16 \
  --rasp_cost_gamma 1.0 \
  --rasp_max_step_cost_fraction 0.05 \
  --device 0 \
  --seed 29 \
  --print-freq 20 \
  --save-interval 10 \
> logs/seg/rasp_sfseg/cvc_rasp_60ep.log 2>&1 &

echo "PID=$!"
```

Monitor:

```bash
tail -f logs/seg/rasp_sfseg/cvc_rasp_60ep.log
```

Pruning events:

```bash
grep "status=pruned" logs/seg/rasp_sfseg/cvc_rasp_60ep.log
```

All RASP epoch summaries:

```bash
grep "RASP(" logs/seg/rasp_sfseg/cvc_rasp_60ep.log
```

Final history:

```bash
tail -10 runs/seg/rasp_sfseg/cvc_rasp_60ep/rasp_history.jsonl
```

Final RASP files:

```text
runs/seg/rasp_sfseg/cvc_rasp_60ep/checkpoints/
├── yolo26s_rasp_sfseg_latent_epoch_60.pt
├── rasp_sfseg_state_epoch_60.pt
└── rasp_sfseg_state_latest.pt
```

---

# 21. Inspect final RASP state

```bash
python - <<'PY'
import torch

STATE = (
    "runs/seg/rasp_sfseg/cvc_rasp_60ep/checkpoints/"
    "rasp_sfseg_state_epoch_60.pt"
)

s = torch.load(
    STATE,
    map_location="cpu",
    weights_only=False,
)

print("epoch       :", s.get("epoch"))
print("global_step :", s.get("global_step"))
print("format      :", s.get("format"))

r = s["rasp"]
print("enabled     :", r.get("enabled"))
print("events      :", r.get("prune_events"))
print("last prune  :", r.get("last_prune_epoch"))
print("stats       :", r.get("current_stats"))

pruner = r["pruner"]
print()
print("PER-GROUP MASKS")
print("-" * 80)

total = 0
pruned = 0
for name, row in pruner["groups"].items():
    mask = row["mask"].bool()
    h = int(mask.numel())
    p = int((~mask).sum().item())
    total += h
    pruned += p
    print(f"{name:55s} hidden={h:4d} pruned={p:4d} s={p/max(h,1):.3f}")

print()
print("total hidden :", total)
print("total pruned :", pruned)
print("sparsity     :", pruned / max(total, 1))
PY
```

---

# 22. Physical compact export

```bash
export RASP_OUT="$HOME/MedRT-SFOD/runs/seg/rasp_sfseg/cvc_rasp_60ep"
export RASP_STATE="$RASP_OUT/checkpoints/rasp_sfseg_state_epoch_60.pt"
export RASP_LATENT="$RASP_OUT/checkpoints/yolo26s_rasp_sfseg_latent_epoch_60.pt"

rm -f \
  "$RASP_OUT/yolo26s_rasp_sfseg_compact.pt" \
  "$RASP_OUT/yolo26s_rasp_sfseg_compact.pt.report.json"

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/export_rasp_compact.py \
  --state "$RASP_STATE" \
  --latent_model "$RASP_LATENT" \
  --out "$RASP_OUT/yolo26s_rasp_sfseg_compact.pt" \
  --device 0 \
  --verify \
  --verify_imgsz 256 \
  --verify_tol 2e-4 \
  --count_macs \
> logs/seg/rasp_sfseg/cvc_rasp_export_compact.log 2>&1 &

echo "PID=$!"
```

Monitor:

```bash
tail -f logs/seg/rasp_sfseg/cvc_rasp_export_compact.log
```

Compact model:

```text
runs/seg/rasp_sfseg/cvc_rasp_60ep/yolo26s_rasp_sfseg_compact.pt
```

Compact report:

```text
runs/seg/rasp_sfseg/cvc_rasp_60ep/yolo26s_rasp_sfseg_compact.pt.report.json
```

---

# 23. Final compact evaluation — CVC-ClinicDB

```bash
nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/eval_source_seg.py \
  --model runs/seg/rasp_sfseg/cvc_rasp_60ep/yolo26s_rasp_sfseg_compact.pt \
  --data dataset/CVC-ClinicDB-YOLO26/dataset_seg.yaml \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --conf 0.25 \
  --out runs/seg/benchmarks/cvc_clinicdb_rasp_compact_metrics.json \
> logs/seg/benchmarks/cvc_clinicdb_rasp_compact_eval.log 2>&1 &

echo "PID=$!"
```

Monitor:

```bash
tail -f logs/seg/benchmarks/cvc_clinicdb_rasp_compact_eval.log
```

---

# 24. Freeze final compact model

```bash
mkdir -p runs/seg/rasp_sfseg/frozen

cp \
  runs/seg/rasp_sfseg/cvc_rasp_60ep/yolo26s_rasp_sfseg_compact.pt \
  runs/seg/rasp_sfseg/frozen/cvc_rasp_sfseg_final.pt

cp \
  runs/seg/rasp_sfseg/cvc_rasp_60ep/yolo26s_rasp_sfseg_compact.pt.report.json \
  runs/seg/rasp_sfseg/frozen/

cp \
  runs/seg/benchmarks/cvc_clinicdb_rasp_compact_metrics.json \
  runs/seg/rasp_sfseg/frozen/

sha256sum \
  runs/seg/rasp_sfseg/frozen/cvc_rasp_sfseg_final.pt \
  | tee runs/seg/rasp_sfseg/frozen/cvc_rasp_sfseg_final.sha256
```

Canonical final model:

```text
runs/seg/rasp_sfseg/frozen/cvc_rasp_sfseg_final.pt
```

---

# 25. Same-protocol FP32 latency/FPS benchmark

## 25.1. Dense model

```bash
python - <<'PY'
import time
import json
import torch
from ultralytics import YOLO

MODEL = "runs/seg/dense_sfseg/frozen/cvc_dense_sfseg_epoch60.pt"
OUT = "runs/seg/benchmarks/cvc_dense_fp32_latency.json"
DEVICE = torch.device("cuda:0")
B = 1
H = W = 640
WARMUP = 100
RUNS = 500

m = YOLO(MODEL).model.to(DEVICE).float().eval()
x = torch.randn(B, 3, H, W, device=DEVICE)

with torch.inference_mode():
    for _ in range(WARMUP):
        _ = m(x)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(RUNS):
        _ = m(x)
    end.record()
    torch.cuda.synchronize()

ms = float(start.elapsed_time(end)) / RUNS
fps = 1000.0 * B / ms

result = {
    "model": MODEL,
    "precision": "FP32",
    "batch": B,
    "imgsz": [H, W],
    "warmup": WARMUP,
    "runs": RUNS,
    "latency_ms": ms,
    "fps": fps,
}

print(json.dumps(result, indent=2))
open(OUT, "w").write(json.dumps(result, indent=2))
PY
```

## 25.2. Compact model

```bash
python - <<'PY'
import json
import torch
from ultralytics import YOLO

MODEL = "runs/seg/rasp_sfseg/frozen/cvc_rasp_sfseg_final.pt"
OUT = "runs/seg/benchmarks/cvc_rasp_compact_fp32_latency.json"
DEVICE = torch.device("cuda:0")
B = 1
H = W = 640
WARMUP = 100
RUNS = 500

m = YOLO(MODEL).model.to(DEVICE).float().eval()
x = torch.randn(B, 3, H, W, device=DEVICE)

with torch.inference_mode():
    for _ in range(WARMUP):
        _ = m(x)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(RUNS):
        _ = m(x)
    end.record()
    torch.cuda.synchronize()

ms = float(start.elapsed_time(end)) / RUNS
fps = 1000.0 * B / ms

result = {
    "model": MODEL,
    "precision": "FP32",
    "batch": B,
    "imgsz": [H, W],
    "warmup": WARMUP,
    "runs": RUNS,
    "latency_ms": ms,
    "fps": fps,
}

print(json.dumps(result, indent=2))
open(OUT, "w").write(json.dumps(result, indent=2))
PY
```

---

# 26. TensorRT FP16 export

Dense:

```bash
yolo export \
  model=runs/seg/dense_sfseg/frozen/cvc_dense_sfseg_epoch60.pt \
  format=engine \
  imgsz=640 \
  batch=1 \
  half=True \
  device=0
```

Compact:

```bash
yolo export \
  model=runs/seg/rasp_sfseg/frozen/cvc_rasp_sfseg_final.pt \
  format=engine \
  imgsz=640 \
  batch=1 \
  half=True \
  device=0
```

---

# 27. Main artifacts

```text
SOURCE
runs/seg/source/frozen/
├── kvasir_yolo26s_seg_source_best.pt
├── kvasir_yolo26s_seg_source_best.sha256
├── split_train.txt
└── split_val.txt

STAGE 1
runs/seg/stage1/frozen/
└── yolo26s_seg_cvc_adabn.pt

DENSE MEDRT-SFSEG
runs/seg/dense_sfseg/frozen/
├── cvc_dense_sfseg_epoch60.pt
├── cvc_dense_sfseg_epoch60.sha256
├── stage2_metadata.json
└── cvc_clinicdb_dense_sfseg_epoch60_metrics.json

RASP-SFSEG
runs/seg/rasp_sfseg/cvc_rasp_60ep/checkpoints/
├── yolo26s_rasp_sfseg_latent_epoch_60.pt
├── rasp_sfseg_state_epoch_60.pt
└── rasp_sfseg_state_latest.pt

FINAL COMPACT
runs/seg/rasp_sfseg/frozen/
├── cvc_rasp_sfseg_final.pt
├── cvc_rasp_sfseg_final.sha256
├── yolo26s_rasp_sfseg_compact.pt.report.json
└── cvc_clinicdb_rasp_compact_metrics.json
```

---

# 28. Main benchmark files

```text
runs/seg/benchmarks/
├── kvasir_yolo26s_seg_source_metrics.json
├── cvc_clinicdb_source_only_metrics.json
├── cvc_clinicdb_adabn_metrics.json
├── cvc_clinicdb_dense_sfseg_epoch60_metrics.json
├── cvc_clinicdb_rasp_compact_metrics.json
├── cvc_dense_fp32_latency.json
└── cvc_rasp_compact_fp32_latency.json
```

---

# 29. Pipeline order

```text
RAW Kvasir-SEG
    ↓
Prepare YOLO26 segmentation dataset
    ↓
YOLO26-S-Seg source training — 100 epochs
    ↓
Source validation
    ↓
Freeze source checkpoint
    ↓
RAW CVC-ClinicDB
    ↓
Prepare CVC target dataset
    ↓
Source-only target evaluation
    ↓
AdaBN — target images only
    ↓
Freeze AdaBN checkpoint
    ↓
Dual-head pseudo-label audit
    ↓
Mask-DHF audit
    ↓
Mask-stability audit
    ↓
Dense MedRT-SFSeg — 60 epochs
    ↓
Dense target evaluation
    ↓
Freeze Dense checkpoint
    ↓
RASP-SFSeg structural / DepGraph audit
    ↓
RASP-SFSeg — 60 epochs
    ↓
Physical compact export
    ↓
Masked-vs-compact numerical equivalence
    ↓
Compact CVC evaluation
    ↓
Freeze final compact checkpoint
    ↓
FP32 latency / FPS
    ↓
TensorRT FP16 export / benchmark
```

---

# 30. Source-free protocol checklist

```text
[ ] Source Kvasir labels used only for supervised source training
[ ] CVC target images used for AdaBN / Stage-2 / RASP adaptation
[ ] CVC target labels NOT read during adaptation
[ ] CVC GT masks NOT read during adaptation
[ ] No target-GT early stopping
[ ] No target-GT pruning decision
[ ] No target-GT pruning-ratio selection
[ ] No target-GT best-epoch selection
[ ] Dense official checkpoint = epoch 60
[ ] RASP official checkpoint = epoch 60
[ ] Physical compaction equivalence PASS before reporting Params/MACs
[ ] Dense and compact latency measured with same hardware/input/batch/precision
```

