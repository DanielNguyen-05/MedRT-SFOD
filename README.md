# RASPT-SFOD / -SFOD — Hướng dẫn chạy project hoàn chỉnh

> **Project root:** `~/MedRT-SFOD`  
> **Task:** Source-Free Object Detection — Cityscapes → Foggy Cityscapes  
> **Detector:** YOLO26-M  
> **Baseline:** RT-SFOD  
> **Proposed method:** RASP-SFOD

---

# 1. Cấu trúc project

Project được đặt tại:

```text
~/MedRT-SFOD/
├── .venv/
├── dataset/
├── logs/
├── runs/
├── scripts/
│   └── YOLO26/
├── ultralytics/
├── requirements-rasp.txt
└── yolo26m.pt
```

Các script chính:

```text
scripts/YOLO26/
├── train_source_supervised.py
├── stage0_stage1_adabn_rc_yolo26.py
├── audit_rtsfod_yolo26.py
├── stage2_rtsfod_yolo26.py
├── rasp_pruning.py
├── inspect_rasp_prunable_groups.py
├── test_rasp_smoke.py
├── stage2_rasp_rtsfod_yolo26.py
├── export_rasp_compact.py
└── eval_rasp_student.py
```

---

# 2. Dataset

## 2.1. Source domain

```text
Cityscapes
train: 2975 images
val:   500 images
```

YAML:

```text
dataset/c2f_yolo/cityscapes/cityscapes.yaml
```

## 2.2. Target domain

```text
Foggy Cityscapes beta = 0.02
train: 2975 images
val:   500 images
```

YAML:

```text
dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml
```

## 2.3. Classes

```text
0 person
1 rider
2 car
3 truck
4 bus
5 train
6 motorcycle
7 bicycle
```

---

# 3. Quy trình chạy toàn bộ project

```text
YOLO26-M COCO pretrained
        ↓
Source training trên Clear Cityscapes
        ↓
Stage-1 AdaBN trên Foggy train
        ↓
RT-SFOD pseudo-label audit
        ↓
RT-SFOD smoke test
        ↓
Dense RT-SFOD 60 epochs
        ↓
RASP smoke test
        ↓
RASP structural / DepGraph audit
        ↓
RASP pruning-event smoke
        ↓
RASP-SFOD 60 epochs
        ↓
Final RASP state check
        ↓
Physical compact export
        ↓
Numerical equivalence verification
        ↓
Compact model evaluation trên Foggy val
        ↓
Dense vs compact comparison
```

---

# 4. Khởi tạo môi trường

## Mục đích

Kích hoạt môi trường Python của project và đảm bảo toàn bộ script sử dụng code local trong `~/MedRT-SFOD`.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"
```

Kiểm tra môi trường:

```bash
python - <<'PY'
import torch
import ultralytics

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

print("ultralytics:", ultralytics.__version__)
print("ultralytics_path:", ultralytics.__file__)
PY
```

Cài dependency nếu cần:

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements-rasp.txt
python -m pip install thop torch-pruning scikit-learn kneed
```

---

# 5. Kiểm tra project trước khi chạy

## Mục đích

Xác nhận các script, dataset YAML và pretrained checkpoint cần thiết đều tồn tại.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

python - <<'PY'
from pathlib import Path

required = [
    "yolo26m.pt",
    "scripts/YOLO26/train_source_supervised.py",
    "scripts/YOLO26/stage0_stage1_adabn_rc_yolo26.py",
    "scripts/YOLO26/audit_rtsfod_yolo26.py",
    "scripts/YOLO26/stage2_rtsfod_yolo26.py",
    "scripts/YOLO26/rasp_pruning.py",
    "scripts/YOLO26/inspect_rasp_prunable_groups.py",
    "scripts/YOLO26/test_rasp_smoke.py",
    "scripts/YOLO26/stage2_rasp_rtsfod_yolo26.py",
    "scripts/YOLO26/export_rasp_compact.py",
    "scripts/YOLO26/eval_rasp_student.py",
    "dataset/c2f_yolo/cityscapes/cityscapes.yaml",
    "dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml",
]

missing = [p for p in required if not Path(p).exists()]

if missing:
    print("Missing files:")
    for p in missing:
        print(" -", p)
    raise SystemExit(1)

print("[OK] Project files are ready.")
PY
```

---

# 6. Kiểm tra pretrained YOLO26-M

## Mục đích

Xác nhận checkpoint `yolo26m.pt` là YOLO26-M COCO pretrained dùng để khởi tạo source detector.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

PYTHONPATH="$PWD" python - <<'PY'
from pathlib import Path
from ultralytics import YOLO

CKPT = Path("yolo26m.pt")

model = YOLO(str(CKPT), task="detect")
net = model.model

params = sum(p.numel() for p in net.parameters())

print("=" * 70)
print("YOLO26-M PRETRAINED CHECK")
print("=" * 70)
print("checkpoint :", CKPT.resolve())
print("size_MB    :", CKPT.stat().st_size / 1024**2)
print("params     :", params)
print("params_M   :", params / 1e6)
print("nc         :", net.yaml.get("nc"))
print("scale      :", net.yaml.get("scale"))
print("end2end    :", getattr(net, "end2end", None))
print("head       :", net.model[-1].__class__.__name__)
PY
```

Checkpoint hiện dùng:

```text
~/MedRT-SFOD/yolo26m.pt
```

SHA256:

```text
401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7
```

Kiểm tra hash:

```bash
cd ~/MedRT-SFOD
sha256sum yolo26m.pt
```

---

# 7. Stage 1 — Source training trên Clear Cityscapes

## Mục đích

Fine-tune YOLO26-M pretrained trên labeled Clear Cityscapes với 8 classes để tạo source detector.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

export COCO_Y26M="$HOME/MedRT-SFOD/yolo26m.pt"

mkdir -p logs
rm -rf runs/c2f_source_yolo26m_pretrained

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/train_source_supervised.py \
  --weights "$COCO_Y26M" \
  --data dataset/c2f_yolo/cityscapes/cityscapes.yaml \
  --task detect \
  --epochs 100 \
  --imgsz 1024 \
  --batch 4 \
  --device 0 \
  --out-dir runs/c2f_source_yolo26m_pretrained \
> logs/c2f_source_yolo26m_pretrained.log 2>&1 &

echo "PID=$!"
```

Theo dõi:

```bash
tail -f logs/c2f_source_yolo26m_pretrained.log
```

Checkpoint cần giữ:

```text
runs/c2f_source_yolo26m_pretrained/weights/best.pt
```

Gán biến:

```bash
export SOURCE_CKPT="$HOME/MedRT-SFOD/runs/c2f_source_yolo26m_pretrained/weights/best.pt"
```

---

# 8. Stage 1 — AdaBN trên Foggy Cityscapes train

## Mục đích

Cập nhật BatchNorm statistics của source detector bằng unlabeled Foggy train images trước khi chạy self-training.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

export SOURCE_CKPT="$HOME/MedRT-SFOD/runs/c2f_source_yolo26m_pretrained/weights/best.pt"

mkdir -p logs
rm -rf runs/c2f_stage1_official_yolo26m_pretrained

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/stage0_stage1_adabn_rc_yolo26.py \
  --weights "$SOURCE_CKPT" \
  --data dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml \
  --out_dir runs/c2f_stage1_official_yolo26m_pretrained \
  --imgsz 1024 \
  --batch 16 \
  --workers 4 \
  --epochs_adabn 2 \
  --epochs_rc 0 \
  --device 0 \
> logs/c2f_stage1_official_yolo26m_pretrained.log 2>&1 &

echo "PID=$!"
```

Theo dõi:

```bash
tail -f logs/c2f_stage1_official_yolo26m_pretrained.log
```

Checkpoint cần giữ:

```text
runs/c2f_stage1_official_yolo26m_pretrained/
└── yolo26_stage1_adabnrc_foggy_cityscapes.pt
```

Gán biến:

```bash
export STAGE1_CKPT="$HOME/MedRT-SFOD/runs/c2f_stage1_official_yolo26m_pretrained/yolo26_stage1_adabnrc_foggy_cityscapes.pt"
```

---

# 9. RT-SFOD pseudo-label audit

## Mục đích

Đánh giá chất lượng pseudo-label O2O và DHF của Stage-1 Teacher trước khi chạy full RT-SFOD.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

export STAGE1_CKPT="$HOME/MedRT-SFOD/runs/c2f_stage1_official_yolo26m_pretrained/yolo26_stage1_adabnrc_foggy_cityscapes.pt"

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/audit_rtsfod_yolo26.py \
  --model "$STAGE1_CKPT" \
  --data dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml \
  --imgsz 1024 \
  --iou 0.5 \
  --device 0 \
> logs/c2f_rtsfod_yolo26m_pretrained_audit.log 2>&1 &

echo "PID=$!"
```

Xem kết quả:

```bash
cat logs/c2f_rtsfod_yolo26m_pretrained_audit.log
```

---

# 10. RT-SFOD smoke test

## Mục đích

Kiểm tra pipeline Mean-Teacher + DHF + MARD chạy ổn định trước full training.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

export STAGE1_CKPT="$HOME/MedRT-SFOD/runs/c2f_stage1_official_yolo26m_pretrained/yolo26_stage1_adabnrc_foggy_cityscapes.pt"

rm -rf runs/c2f_rtsfod_yolo26m_pretrained_smoke

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/stage2_rtsfod_yolo26.py \
  --stage1_model "$STAGE1_CKPT" \
  --data dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml \
  --out_dir runs/c2f_rtsfod_yolo26m_pretrained_smoke \
  --imgsz 1024 \
  --batch 8 \
  --workers 2 \
  --epochs 1 \
  --lr 1e-4 \
  --grad_clip 10 \
  --tau_o2o 0.5 \
  --tau_o2m 0.5 \
  --tau_no 0.2 \
  --tau_dup 0.7 \
  --mard_lambda0 0.05 \
  --ema_momentum 0.999 \
  --save_interval 1 \
  --print_freq 10 \
  --device 0 \
> logs/c2f_rtsfod_yolo26m_pretrained_smoke.log 2>&1 &

echo "PID=$!"
```

Theo dõi:

```bash
tail -f logs/c2f_rtsfod_yolo26m_pretrained_smoke.log
```

---

# 11. Dense RT-SFOD — 60 epochs

## Mục đích

Train dense RT-SFOD baseline từ Stage-1 checkpoint trên unlabeled Foggy train images.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

export STAGE1_CKPT="$HOME/MedRT-SFOD/runs/c2f_stage1_official_yolo26m_pretrained/yolo26_stage1_adabnrc_foggy_cityscapes.pt"

rm -rf runs/c2f_rtsfod_yolo26m_pretrained

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/stage2_rtsfod_yolo26.py \
  --stage1_model "$STAGE1_CKPT" \
  --data dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml \
  --out_dir runs/c2f_rtsfod_yolo26m_pretrained \
  --imgsz 1024 \
  --batch 8 \
  --workers 2 \
  --epochs 60 \
  --lr 1e-4 \
  --grad_clip 10 \
  --tau_o2o 0.5 \
  --tau_o2m 0.5 \
  --tau_no 0.2 \
  --tau_dup 0.7 \
  --mard_lambda0 0.05 \
  --ema_momentum 0.999 \
  --save_interval 10 \
  --print_freq 10 \
  --device 0 \
> logs/c2f_rtsfod_yolo26m_pretrained.log 2>&1 &

echo "PID=$!"
```

Theo dõi:

```bash
tail -f logs/c2f_rtsfod_yolo26m_pretrained.log
```

Final dense checkpoint:

```text
runs/c2f_rtsfod_yolo26m_pretrained/checkpoints/
└── yolo26_stage2_rtsfod_epoch_60.pt
```

Gán biến:

```bash
export DENSE_RTSFOD_CKPT="$HOME/MedRT-SFOD/runs/c2f_rtsfod_yolo26m_pretrained/checkpoints/yolo26_stage2_rtsfod_epoch_60.pt"
```

---

# 12. RASP smoke test

## Mục đích

Kiểm tra các thành phần cơ bản của RASP controller và structured pruning.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

PYTHONPATH="$PWD" \
python scripts/YOLO26/test_rasp_smoke.py
```

---

# 13. RASP structural / DepGraph audit

## Mục đích

Xác định các Bottleneck hidden groups có thể prune an toàn và kiểm tra dependency graph trước main RASP training.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

export DENSE_RTSFOD_CKPT="$HOME/MedRT-SFOD/runs/c2f_rtsfod_yolo26m_pretrained/checkpoints/yolo26_stage2_rtsfod_epoch_60.pt"

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/inspect_rasp_prunable_groups.py \
  --model "$DENSE_RTSFOD_CKPT" \
  --imgsz 256 \
  --device 0 \
  --depgraph \
  --out runs/rasp_audit_yolo26m.json \
> logs/rasp_audit_yolo26m.log 2>&1 &

echo "PID=$!"
```

Xem audit:

```bash
cat logs/rasp_audit_yolo26m.log
```

Audit report:

```text
runs/rasp_audit_yolo26m.json
```

---

# 14. RASP pruning-event smoke — 6 epochs

## Mục đích

Chạy qua warm-up và kích hoạt ít nhất một pruning decision trước main 60-epoch RASP training.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

export STAGE1_CKPT="$HOME/MedRT-SFOD/runs/c2f_stage1_official_yolo26m_pretrained/yolo26_stage1_adabnrc_foggy_cityscapes.pt"

rm -rf runs/c2f_rasp_yolo26m_event_smoke

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/stage2_rasp_rtsfod_yolo26.py \
  --stage1_model "$STAGE1_CKPT" \
  --data dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml \
  --out_dir runs/c2f_rasp_yolo26m_event_smoke \
  --imgsz 1024 \
  --batch 4 \
  --workers 2 \
  --epochs 6 \
  --lr 1e-4 \
  --grad_clip 10 \
  --tau_o2o 0.5 \
  --tau_o2m 0.5 \
  --tau_no 0.2 \
  --tau_dup 0.7 \
  --mard_lambda0 0.05 \
  --ema_momentum 0.999 \
  --save_interval 1 \
  --print_freq 20 \
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
> logs/c2f_rasp_yolo26m_event_smoke.log 2>&1 &

echo "PID=$!"
```

Xem pruning status:

```bash
grep "RASP(status" \
  logs/c2f_rasp_yolo26m_event_smoke.log
```

---

# 15. Main RASP-SFOD — 60 epochs

## Mục đích

Train proposed RASP-SFOD từ cùng Stage-1 checkpoint với dense RT-SFOD, đồng thời thực hiện adaptive structured pruning trong quá trình source-free adaptation.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

export STAGE1_CKPT="$HOME/MedRT-SFOD/runs/c2f_stage1_official_yolo26m_pretrained/yolo26_stage1_adabnrc_foggy_cityscapes.pt"

rm -rf runs/c2f_rasp_yolo26m

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/stage2_rasp_rtsfod_yolo26.py \
  --stage1_model "$STAGE1_CKPT" \
  --data dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml \
  --out_dir runs/c2f_rasp_yolo26m \
  --imgsz 1024 \
  --batch 4 \
  --workers 2 \
  --epochs 60 \
  --lr 1e-4 \
  --grad_clip 10 \
  --tau_o2o 0.5 \
  --tau_o2m 0.5 \
  --tau_no 0.2 \
  --tau_dup 0.7 \
  --mard_lambda0 0.05 \
  --ema_momentum 0.999 \
  --save_interval 5 \
  --print_freq 20 \
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
  --rasp_depgraph_audit \
  --rasp_require_depgraph \
  --rasp_audit_imgsz 256 \
  --device 0 \
> logs/c2f_rasp_yolo26m.log 2>&1 &

echo "PID=$!"
```

Theo dõi:

```bash
tail -f logs/c2f_rasp_yolo26m.log
```

Xem pruning history:

```bash
tail -20 \
  runs/c2f_rasp_yolo26m/rasp_history.jsonl
```

Final files:

```text
runs/c2f_rasp_yolo26m/checkpoints/
├── rasp_training_state_epoch_60.pt
├── rasp_training_state_latest.pt
└── yolo26_rasp_latent_epoch_60.pt
```

---

# 16. Resume RASP training

## Mục đích

Tiếp tục RASP training từ `rasp_training_state_latest.pt` nếu quá trình training **chưa hoàn tất** đủ 60 epochs.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

export STAGE1_CKPT="$HOME/MedRT-SFOD/runs/c2f_stage1_official_yolo26m_pretrained/yolo26_stage1_adabnrc_foggy_cityscapes.pt"

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/stage2_rasp_rtsfod_yolo26.py \
  --stage1_model "$STAGE1_CKPT" \
  --data dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml \
  --out_dir runs/c2f_rasp_yolo26m \
  --imgsz 1024 \
  --batch 4 \
  --workers 2 \
  --epochs 60 \
  --lr 1e-4 \
  --grad_clip 10 \
  --tau_o2o 0.5 \
  --tau_o2m 0.5 \
  --tau_no 0.2 \
  --tau_dup 0.7 \
  --mard_lambda0 0.05 \
  --ema_momentum 0.999 \
  --save_interval 5 \
  --print_freq 20 \
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
  --resume_state runs/c2f_rasp_yolo26m/checkpoints/rasp_training_state_latest.pt \
  --device 0 \
> logs/c2f_rasp_yolo26m_resume.log 2>&1 &

echo "PID=$!"
```

---

# 17. Kiểm tra final RASP state

## Mục đích

Xác nhận final checkpoint chứa đầy đủ Student state và RASP state trước physical compact export.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

PYTHONPATH="$PWD" python - <<'PY'
import torch

STATE = (
    "runs/c2f_rasp_yolo26m/checkpoints/"
    "rasp_training_state_epoch_60.pt"
)

s = torch.load(
    STATE,
    map_location="cpu",
    weights_only=False,
)

print("epoch       :", s.get("epoch"))
print("global_step :", s.get("global_step"))
print("student     :", "student_state" in s)
print("teacher     :", "teacher_state" in s)
print("rasp        :", "rasp" in s)

if "rasp" in s:
    print("prune_events:", s["rasp"].get("prune_events"))
    print("current_stats:", s["rasp"].get("current_stats"))
PY
```

Kiểm tra file:

```bash
ls -lh \
  runs/c2f_rasp_yolo26m/checkpoints/rasp_training_state_epoch_60.pt \
  runs/c2f_rasp_yolo26m/checkpoints/yolo26_rasp_latent_epoch_60.pt
```

---

# 18. Physical compact export

## Mục đích

Chuyển masked latent Student sang physical compact model bằng cách thực sự loại bỏ các hidden channels đã được RASP chọn.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

export RASP_OUT="$HOME/MedRT-SFOD/runs/c2f_rasp_yolo26m"
export RASP_STATE="$RASP_OUT/checkpoints/rasp_training_state_epoch_60.pt"
export RASP_LATENT="$RASP_OUT/checkpoints/yolo26_rasp_latent_epoch_60.pt"

rm -f \
  "$RASP_OUT/yolo26m_rasp_compact.pt" \
  "$RASP_OUT/yolo26m_rasp_compact.pt.report.json"

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/export_rasp_compact.py \
  --state "$RASP_STATE" \
  --latent_model "$RASP_LATENT" \
  --out "$RASP_OUT/yolo26m_rasp_compact.pt" \
  --device 0 \
  --verify \
  --verify_imgsz 256 \
  --verify_tol 2e-4 \
  --count_macs \
> logs/c2f_rasp_export_compact.log 2>&1 &

echo "PID=$!"
```

Theo dõi:

```bash
tail -f logs/c2f_rasp_export_compact.log
```

Output cần giữ:

```text
runs/c2f_rasp_yolo26m/
├── yolo26m_rasp_compact.pt
└── yolo26m_rasp_compact.pt.report.json
```

---

# 19. Final Foggy-val evaluation — RASP compact

## Mục đích

Đánh giá compact RASP-SFOD model trên Foggy Cityscapes validation set.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/eval_rasp_student.py \
  --model runs/c2f_rasp_yolo26m/yolo26m_rasp_compact.pt \
  --data dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml \
  --imgsz 1024 \
  --batch 4 \
  --device 0 \
> logs/c2f_rasp_compact_eval.log 2>&1 &

echo "PID=$!"
```

Xem kết quả:

```bash
cat logs/c2f_rasp_compact_eval.log
```

---

# 20. Same-protocol evaluation — Dense RT-SFOD

## Mục đích

Đánh giá dense RT-SFOD bằng cùng image size, batch size và validator với compact RASP model để so sánh accuracy và runtime.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

PYTHONPATH="$PWD" python - <<'PY'
from ultralytics import YOLO

MODEL = (
    "runs/c2f_rtsfod_yolo26m_pretrained/checkpoints/"
    "yolo26_stage2_rtsfod_epoch_60.pt"
)

DATA = (
    "dataset/c2f_yolo/foggy_cityscapes/"
    "foggy_cityscapes.yaml"
)

m = YOLO(MODEL)

r = m.val(
    data=DATA,
    imgsz=1024,
    batch=4,
    device=0,
    conf=0.001,
    iou=0.6,
    plots=False,
    verbose=True,
)

print("=" * 70)
print("DENSE RT-SFOD")
print("=" * 70)
print(f"Precision : {r.box.mp:.6f}")
print(f"Recall    : {r.box.mr:.6f}")
print(f"mAP50     : {r.box.map50:.6f}")
print(f"mAP50-95  : {r.box.map:.6f}")
print("speed     :", r.speed)
PY
```

---

# 21. Đánh giá Source / Stage 1 / Dense / RASP cùng protocol

## Mục đích

Tạo bảng final metrics cho toàn bộ pipeline trên cùng Foggy validation split.

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate

PYTHONPATH="$PWD" python - <<'PY'
import json
from pathlib import Path
from ultralytics import YOLO

DATA = (
    "dataset/c2f_yolo/foggy_cityscapes/"
    "foggy_cityscapes.yaml"
)

MODELS = {
    "source_only": (
        "runs/c2f_source_yolo26m_pretrained/"
        "weights/best.pt"
    ),
    "stage1_adabn": (
        "runs/c2f_stage1_official_yolo26m_pretrained/"
        "yolo26_stage1_adabnrc_foggy_cityscapes.pt"
    ),
    "dense_rtsfod": (
        "runs/c2f_rtsfod_yolo26m_pretrained/checkpoints/"
        "yolo26_stage2_rtsfod_epoch_60.pt"
    ),
    "rasp_compact": (
        "runs/c2f_rasp_yolo26m/"
        "yolo26m_rasp_compact.pt"
    ),
}

results = {}

for name, ckpt in MODELS.items():
    if not Path(ckpt).exists():
        print("[SKIP missing]", name, ckpt)
        continue

    print("\n===", name, "===")

    m = YOLO(ckpt)

    r = m.val(
        data=DATA,
        imgsz=1024,
        batch=4,
        device=0,
        conf=0.001,
        iou=0.6,
        plots=False,
        verbose=False,
    )

    params = sum(p.numel() for p in m.model.parameters())

    results[name] = {
        "checkpoint": ckpt,
        "params": int(params),
        "params_M": params / 1e6,
        "mAP50_95": float(r.box.map),
        "mAP50": float(r.box.map50),
        "mAP75": float(r.box.map75),
        "precision": float(r.box.mp),
        "recall": float(r.box.mr),
        "speed": r.speed,
    }

    print(json.dumps(results[name], indent=2))

Path("runs/final_metrics_c2f.json").write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print("\nSaved -> runs/final_metrics_c2f.json")
PY
```

Final metrics:

```text
runs/final_metrics_c2f.json
```

---

# 22. Xem RASP pruning history

## Mục đích

Theo dõi pruning trajectory, pruning events và final adaptive sparsity của RASP.

```bash
cd ~/MedRT-SFOD

cat runs/c2f_rasp_yolo26m/rasp_history.jsonl
```

Xem các epoch cuối:

```bash
tail -20 \
  runs/c2f_rasp_yolo26m/rasp_history.jsonl
```

Xem các status pruning:

```bash
grep "RASP(status" \
  logs/c2f_rasp_yolo26m.log
```

---

# 23. Các checkpoint và report cần giữ

```text
~/MedRT-SFOD/
├── yolo26m.pt
│
├── runs/
│   ├── c2f_source_yolo26m_pretrained/
│   │   └── weights/
│   │       └── best.pt
│   │
│   ├── c2f_stage1_official_yolo26m_pretrained/
│   │   └── yolo26_stage1_adabnrc_foggy_cityscapes.pt
│   │
│   ├── c2f_rtsfod_yolo26m_pretrained/
│   │   └── checkpoints/
│   │       └── yolo26_stage2_rtsfod_epoch_60.pt
│   │
│   ├── rasp_audit_yolo26m.json
│   │
│   └── c2f_rasp_yolo26m/
│       ├── rasp_history.jsonl
│       ├── yolo26m_rasp_compact.pt
│       ├── yolo26m_rasp_compact.pt.report.json
│       └── checkpoints/
│           ├── rasp_training_state_epoch_60.pt
│           ├── rasp_training_state_latest.pt
│           └── yolo26_rasp_latent_epoch_60.pt
│
└── logs/
```

---

# 24. Kết quả tham chiếu của run hiện tại

## Source-only YOLO26-M

```text
Precision  = 0.7011
Recall     = 0.4062
mAP50      = 0.4280
mAP50-95   = 0.2796
```

## Stage-1 AdaBN

```text
Precision  = 0.7156
Recall     = 0.4287
mAP50      = 0.4712
mAP50-95   = 0.3070
```

## Dense RT-SFOD

```text
Precision  = 0.6988
Recall     = 0.4621
mAP50      = 0.5130
mAP50-95   = 0.3352
Params     = 21.785M
MACs@256   = 5.957G
```

## RASP-SFOD compact

```text
Precision  = 0.7320
Recall     = 0.4580
mAP50      = 0.509016
mAP50-95   = 0.326954
Params     = 21.255M
MACs@256   = 5.683G
Inference  = 12.3 ms/image
```

Physical compaction:

```text
CPU full max_abs_diff = 7.62939453e-05
CPU raw max_abs_diff  = 6.86645508e-05
verify_tol             = 2e-4
```

Compression:

```text
Parameter reduction = 2.44%
MAC reduction       = 4.60%
```

---

# 25. Lệnh chạy rút gọn theo thứ tự

```text
1. cd ~/MedRT-SFOD
2. source .venv/bin/activate
3. export PYTHONPATH="$PWD"

4. Check project
5. Check yolo26m.pt

6. Train source detector
7. Run Stage-1 AdaBN

8. Run RT-SFOD pseudo-label audit
9. Run RT-SFOD smoke
10. Train dense RT-SFOD 60 epochs

11. Run RASP smoke
12. Run RASP structural / DepGraph audit
13. Run RASP 6-epoch pruning-event smoke
14. Train RASP-SFOD 60 epochs

15. Resume only if RASP training has not reached epoch 60

16. Check final RASP state
17. Export physical compact model
18. Verify masked-vs-compact numerical equivalence
19. Count compact Params / MACs

20. Evaluate RASP compact on Foggy val
21. Evaluate dense RT-SFOD with the same protocol
22. Export final metrics
```

---

# 26. Main experiment configuration

## RT-SFOD

```text
imgsz             = 1024
epochs            = 60
lr                = 1e-4
grad_clip         = 10

tau_o2o           = 0.5
tau_o2m           = 0.5
tau_no            = 0.2
tau_dup           = 0.7

mard_lambda0      = 0.05

EMA momentum      = 0.999
EMA update        = once per epoch
```

## RASP

```text
warmup epochs             = 5
pruning cycle             = 3 epochs
reliability threshold     = 0.50
Taylor EMA beta           = 0.90
minimum hidden channels   = 16
minimum keep ratio        = 0.50
pack size                 = 8

GMM posterior             = 0.80
GMM BIC gain              = 0
GMM minimum separation    = 1.0
GMM minimum samples       = 16

cost gamma                = 1.0
max step cost fraction    = 0.05
```

---

# 27. Experiment protocol

```text
Source training:
Clear Cityscapes train labels
Clear Cityscapes val labels

Stage-1:
Foggy train images

Dense RT-SFOD:
Foggy train images
Teacher pseudo-labels

RASP-SFOD:
Foggy train images
Teacher pseudo-labels
Target Taylor importance
GMM / Kneedle pruning signals

Final evaluation:
Foggy Cityscapes val labels
```

Main dense RT-SFOD và main RASP-SFOD đều bắt đầu từ cùng:

```text
runs/c2f_stage1_official_yolo26m_pretrained/
yolo26_stage1_adabnrc_foggy_cityscapes.pt
```

---

# 28. Final output

Sau khi hoàn tất toàn bộ pipeline, các artifact chính là:

```text
Source:
runs/c2f_source_yolo26m_pretrained/weights/best.pt

Stage-1:
runs/c2f_stage1_official_yolo26m_pretrained/
yolo26_stage1_adabnrc_foggy_cityscapes.pt

Dense RT-SFOD:
runs/c2f_rtsfod_yolo26m_pretrained/checkpoints/
yolo26_stage2_rtsfod_epoch_60.pt

RASP final state:
runs/c2f_rasp_yolo26m/checkpoints/
rasp_training_state_epoch_60.pt

RASP dense latent:
runs/c2f_rasp_yolo26m/checkpoints/
yolo26_rasp_latent_epoch_60.pt

RASP physical compact:
runs/c2f_rasp_yolo26m/
yolo26m_rasp_compact.pt

Compact report:
runs/c2f_rasp_yolo26m/
yolo26m_rasp_compact.pt.report.json

Final metrics:
runs/final_metrics_c2f.json
```
