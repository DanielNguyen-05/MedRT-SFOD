# HƯỚNG DẪN CHẠY TOÀN BỘ PROJECT MedRT-SFOD / RASP-SFOD

> **Mục tiêu hiện tại:** Cityscapes → Foggy Cityscapes (C2F), backbone **YOLO26-M**, so sánh **RT-SFOD-Y26** với **RASP-SFOD-Y26**, sau đó physical compact export và đánh giá trên Foggy Cityscapes validation set.
>
> Tài liệu này mô tả **2 cách chạy**:
>
> 1. Máy local có NVIDIA GPU/CUDA.
> 2. Google Colab T4, toàn bộ project nằm trong Google Drive.

---

## 0. Protocol cần giữ cố định

Pipeline chính:

```text
YOLO26-M source detector
        │
        │ supervised source training nếu chưa có source checkpoint
        ▼
Clear Cityscapes train + labels
        │
        ▼
SOURCE CHECKPOINT
        │
        │ Stage 1: AdaBN, target images only
        ▼
Foggy Cityscapes train images
        │
        ▼
STAGE-1 CHECKPOINT
        │
        ├──────────────► Original RT-SFOD-Y26 baseline
        │
        └──────────────► RASP-SFOD-Y26
                               │
                               ▼
                       Physical compact export
                               │
                               ▼
                      Foggy Cityscapes val
                       FINAL EVALUATION ONLY
```

### Dataset usage

| Split | Labels được phép dùng? | Mục đích |
|---|---:|---|
| Clear Cityscapes `train` | Có | supervised source training |
| Clear Cityscapes `val` | Có | source validation / chọn `best.pt` |
| Foggy Cityscapes `train` | **Không** | AdaBN + RT-SFOD + RASP adaptation |
| Foggy Cityscapes `val` | Có, **chỉ sau training** | final target evaluation |
| Foggy `test` | Không cần cho primary C2F protocol | không dùng |

**Không dùng Foggy val labels để:** early stopping, chọn pruning ratio, chọn Kneedle knee, chọn epoch tốt nhất, chọn checkpoint, hoặc quyết định pruning.

---

# 1. Tạo `dataset/c2f_yolo` từ raw Cityscapes + Foggy Cityscapes

Nếu project chưa có `dataset/c2f_yolo/`, phải chạy:

```text
scripts/YOLO26/prepare_c2f_yolo.py
```

Script chuyển `gtFine` polygons sang YOLO bounding boxes cho 8 class:

```text
person, rider, car, truck, bus, train, motorcycle, bicycle
```

và dùng Foggy Cityscapes `beta=0.02` cho C2F.

## 1.1. Raw dataset layout bắt buộc

```text
dataset/
├── gtFine_trainvaltest/
│   └── gtFine/
│       ├── train/
│       └── val/
│
├── leftImg8bit_trainvaltest/
│   └── leftImg8bit/
│       ├── train/
│       └── val/
│
└── leftImg8bit_trainvaltest_foggy/
    └── leftImg8bit_foggy/
        ├── train/
        └── val/
```

Foggy filenames phải có dạng:

```text
*_leftImg8bit_foggy_beta_0.02.png
```

Không cần tạo official `test` cho primary C2F experiment.

## 1.2. Đặt converter vào project

File cần nằm tại:

```text
scripts/YOLO26/prepare_c2f_yolo.py
```

Kiểm tra:

```bash
ls scripts/YOLO26/prepare_c2f_yolo.py
```

## 1.3. Convert trên máy local

Từ root `MedRT-SFOD/`:

```bash
PYTHONPATH="$PWD" python scripts/YOLO26/prepare_c2f_yolo.py \
  --dataset-root dataset \
  --out dataset/c2f_yolo \
  --fog-beta 0.02 \
  --copy-images
```

## 1.4. Output đúng

```text
dataset/c2f_yolo/
├── cityscapes/
│   ├── cityscapes.yaml
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
└── foggy_cityscapes/
    ├── foggy_cityscapes.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

Expected official fine split:

```text
train = 2975
val   = 500
```

Script sẽ cảnh báo nếu count khác.

## 1.5. Verify sau convert

```bash
find dataset/c2f_yolo/cityscapes/images/train -type f -o -type l | wc -l
find dataset/c2f_yolo/cityscapes/images/val   -type f -o -type l | wc -l
find dataset/c2f_yolo/foggy_cityscapes/images/train -type f -o -type l | wc -l
find dataset/c2f_yolo/foggy_cityscapes/images/val   -type f -o -type l | wc -l
```

```bash
cat dataset/c2f_yolo/cityscapes/cityscapes.yaml
echo "----------------"
cat dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml
```

## 1.6. Colab

Nếu raw dataset nằm trên Drive, phương án ưu tiên:

```text
Google Drive raw dataset
        ↓
copy đúng train/val + Foggy beta=0.02 xuống /content
        ↓
run prepare_c2f_yolo.py ở /content/MedRT-SFOD
        ↓
train từ local SSD
```

Nếu `/content` thiếu disk, dùng `c2f_yolo` đã prepare sẵn trên Drive hoặc Colab Mode B ở phần sau thay vì duplicate toàn bộ ảnh.

**Protocol:** `foggy_cityscapes/labels/train` có thể tồn tại sau converter nhưng Stage 1/Stage 2 source-free main run không được đọc ground-truth target labels.

---

# 2. Cấu trúc project cần có

Từ root `MedRT-SFOD/`:

```text
MedRT-SFOD/
├── colab/
│   ├── 00_MedRT_SFOD_T4_End_to_End.ipynb
│   ├── 05_rasp_audit_train.ipynb
│   └── 06_rasp_export_eval.ipynb
│
├── dataset/
│   └── c2f_yolo/
│       ├── cityscapes/
│       │   ├── cityscapes.yaml
│       │   ├── images/train/
│       │   ├── images/val/
│       │   ├── labels/train/
│       │   └── labels/val/
│       └── foggy_cityscapes/
│           ├── foggy_cityscapes.yaml
│           ├── images/train/
│           ├── images/val/
│           ├── labels/train/
│           └── labels/val/
│
├── scripts/YOLO26/
│   ├── train_source_supervised.py
│   ├── stage0_stage1_adabn_rc_yolo26_v2.py
│   ├── stage2_rtsfod_yolo26.py
│   ├── rasp_pruning.py
│   ├── stage2_rasp_rtsfod_yolo26.py
│   ├── inspect_rasp_prunable_groups.py
│   ├── export_rasp_compact.py
│   ├── eval_rasp_student.py
│   └── test_rasp_smoke.py
│
├── ultralytics/
├── requirements-rasp.txt
└── ...
```

Kiểm tra nhanh:

```bash
cd /path/to/MedRT-SFOD

ls scripts/YOLO26/train_source_supervised.py \
   scripts/YOLO26/stage0_stage1_adabn_rc_yolo26_v2.py \
   scripts/YOLO26/stage2_rtsfod_yolo26.py \
   scripts/YOLO26/stage2_rasp_rtsfod_yolo26.py \
   scripts/YOLO26/rasp_pruning.py \
   scripts/YOLO26/inspect_rasp_prunable_groups.py \
   scripts/YOLO26/export_rasp_compact.py \
   scripts/YOLO26/eval_rasp_student.py
```

---

# 3. Kiểm tra dataset trước khi dùng GPU

## 3.1. Kiểm tra YAML

```bash
cat dataset/c2f_yolo/cityscapes/cityscapes.yaml
echo "----------------"
cat dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml
```

Hai YAML cần trỏ đúng tới:

```text
.../dataset/c2f_yolo/cityscapes
.../dataset/c2f_yolo/foggy_cityscapes
```

## 3.2. Kiểm tra số file

```bash
find dataset/c2f_yolo/cityscapes/images/train -type f | wc -l
find dataset/c2f_yolo/cityscapes/images/val   -type f | wc -l
find dataset/c2f_yolo/foggy_cityscapes/images/train -type f | wc -l
find dataset/c2f_yolo/foggy_cityscapes/images/val   -type f | wc -l
```

Và labels:

```bash
find dataset/c2f_yolo/cityscapes/labels/train -name '*.txt' | wc -l
find dataset/c2f_yolo/cityscapes/labels/val   -name '*.txt' | wc -l
find dataset/c2f_yolo/foggy_cityscapes/labels/val -name '*.txt' | wc -l
```

Foggy `labels/train` có thể tồn tại trong dataset, nhưng Stage 1 / Stage 2 **không được đọc chúng**.

---

# 4. Source checkpoint: có 2 trường hợp

## Trường hợp A — đã có YOLO26-M source checkpoint train trên Clear Cityscapes

Nếu đã có một checkpoint đúng protocol, ví dụ:

```text
/path/to/yolo26m_cityscapes_source_best.pt
```

thì **không cần train source lại**.

Gán:

```bash
SOURCE_CKPT=/path/to/yolo26m_cityscapes_source_best.pt
```

Sau đó đi thẳng tới **Stage 1 AdaBN**.

## Trường hợp B — chưa có source checkpoint

Train source detector **một lần**, sau đó freeze checkpoint này và dùng chung cho toàn bộ baseline/proposed experiments.

### Lưu ý quan trọng về code hiện tại

`train_source_supervised.py` hiện tạo model bằng **model YAML**, nghĩa là source training hiện tại bắt đầu từ architecture config, không tự động load `yolo26m.pt` COCO weights.

Vì paper RT-SFOD không công bố đầy đủ source-training recipe/checkpoint cho YOLO26-M, hãy ghi rõ reproduction choice này trong báo cáo. Nếu sau này muốn dùng COCO-pretrained initialization, cần sửa source trainer hoặc cung cấp checkpoint `.pt` làm initialization; không được mặc định rằng script hiện tại đã làm điều đó.

---

# 5. Chạy trên MÁY LOCAL CÓ NVIDIA GPU

## 5.1. Tạo môi trường

Từ root project:

```bash
python -m venv .venv
source .venv/bin/activate
```

Cài project và dependency:

```bash
python -m pip install -U pip
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install thop
```

Kiểm tra CUDA:

```bash
nvidia-smi
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
```

Phải có:

```text
CUDA available: True
```

## 5.2. Smoke test RASP

```bash
PYTHONPATH="$PWD" python scripts/YOLO26/test_rasp_smoke.py
```

Expected:

```text
[OK] RASP smoke test
```

`knee_method=max_distance` trong smoke test không phải lỗi; đó là deterministic fallback nếu `kneed` không phát hiện knee hữu ích trên toy curve.

---

# 6. Force đúng YOLO26-M

Không nên train trực tiếp generic `yolo26.yaml` mà không kiểm tra scale.

Tạo alias:

```bash
cp ultralytics/cfg/models/26/yolo26.yaml \
   ultralytics/cfg/models/26/yolo26m.yaml
```

Kiểm tra parameter count:

```bash
PYTHONPATH="$PWD" python - <<'PY'
from ultralytics import YOLO
m = YOLO('ultralytics/cfg/models/26/yolo26m.yaml', task='detect')
n = sum(p.numel() for p in m.model.parameters())
print(f'params = {n/1e6:.3f}M')
assert 15_000_000 < n < 30_000_000, 'Model không giống YOLO26-M'
PY
```

Trong fork hiện tại, YOLO26-M dự kiến nằm khoảng ~21–22M params. Nếu không nằm trong khoảng kiểm tra trên, **dừng trước khi train**.

---

# 7. Stage -1 — train source YOLO26-M trên Clear Cityscapes

**Chỉ chạy nếu chưa có source checkpoint.**

```bash
nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/train_source_supervised.py \
  --model-cfg ultralytics/cfg/models/26/yolo26m.yaml \
  --data dataset/c2f_yolo/cityscapes/cityscapes.yaml \
  --task detect \
  --epochs 100 \
  --imgsz 1024 \
  --batch 4 \
  --device 0 \
  --out-dir runs/c2f_source_yolo26m \
> logs/c2f_source_yolo26m.log 2>&1 &
```

Nếu OOM:

```text
batch 4 → batch 2
```

Không đổi `imgsz` giữa baseline và RASP nếu muốn giữ comparison nhất quán.

Sau training:

```text
runs/detect/runs/c2f_source_yolo26m/weights/best.pt
runs/detect/runs/c2f_source_yolo26m/weights/last.pt
```

Dùng:

```bash
SOURCE_CKPT=runs/detect/runs/c2f_source_yolo26m/weights/last.pt
```

### Source validation trên Clear val

Được phép dùng Clear val labels vì đây là source-domain supervised stage.

```bash
PYTHONPATH="$PWD" python - <<'PY'
from ultralytics import YOLO
m = YOLO('runs/detect/runs/c2f_source_yolo26m/weights/best.pt')
r = m.val(
    data='dataset/c2f_yolo/cityscapes/cityscapes.yaml',
    imgsz=1024,
    batch=4,
    device=0,
    conf=0.001,
    iou=0.6,
    verbose=False,
)
print('Clear val mAP50    =', r.box.map50)
print('Clear val mAP50-95 =', r.box.map)
PY
```

`best.pt` của Ultralytics có thể được chọn dựa trên **Clear validation**, điều này hợp lệ.

---

# 8. Stage 1 — AdaBN trên unlabeled Foggy Cityscapes train

```bash
nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/stage0_stage1_adabn_rc_yolo26_v2.py \
  --weights "$SOURCE_CKPT" \
  --data dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml \
  --out_dir runs/c2f_stage1_yolo26m \
  --imgsz 1024 \
  --batch 4 \
  --workers 2 \
  --epochs_adabn 2 \
  --epochs_rc 0 \
  --device 0 \
> logs/c2f_stage1_adabn.log 2>&1 &
```

**Không thêm**:

```text
--eval
--oracle_early_stop
```

Main source-free protocol không được dùng target validation labels để select model.

Output dự kiến:

```text
runs/c2f_stage1_yolo26m/yolo26_stage1_adabnrc_foggy_cityscapes.pt
```

Gán:

```bash
STAGE1_CKPT=runs/c2f_stage1_yolo26m/yolo26_stage1_adabnrc_foggy_cityscapes.pt
```

---

# 9. RASP audit — bắt buộc trước khi train 60 epochs

```bash
nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/inspect_rasp_prunable_groups.py \
  --model "$STAGE1_CKPT" \
  --imgsz 256 \
  --device 0 \
  --depgraph \
  --out runs/rasp_audit_yolo26m.json \
> logs/rasp_audit_yolo26m.log 2>&1 &
```

Kiểm tra các giá trị:

```text
eligible_hidden_groups
eligible_hidden_channels
controlled_params
controlled_macs
DepGraph local-safe X/Y
```

Go/no-go:

- `eligible_hidden_groups == 0` → **STOP**.
- DepGraph có group unsafe → review trước khi full train.
- controllable parameter/MAC space rất nhỏ → cân nhắc mở rộng pruning space trước khi tốn 60 epochs.
- v1 hiện chủ động prune hidden channels trong standard Bottleneck để physical export an toàn.

---

# 10. Original RT-SFOD-Y26 baseline — 60 epochs

Baseline và RASP phải dùng **chính xác cùng `STAGE1_CKPT`**.

```bash
nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/stage2_rtsfod_yolo26.py \
  --stage1_model "$STAGE1_CKPT" \
  --data dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml \
  --out_dir runs/c2f_rtsfod_yolo26m \
  --imgsz 1024 \
  --batch 4 \
  --workers 2 \
  --epochs 60 \
  --lr 1e-4 \
  --device 0 \
  --save_interval 10 \
> logs/c2f_rtsfod_yolo26m.log 2>&1 &
```

**Không bật `--eval` trong main run.**

Final baseline checkpoint:

```text
runs/c2f_rtsfod_yolo26m/checkpoints/yolo26_stage2_rtsfod_epoch_60.pt
```

### Resume limitation

Original baseline script hiện không lưu đầy đủ Teacher + optimizer + scheduler state để exact resume. Nếu bị ngắt giữa Stage 2 baseline, phương án sạch nhất là chạy lại Stage 2 baseline từ cùng Stage-1 checkpoint.

---

# 11. RASP-SFOD-Y26 — 60 epochs

```bash
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
  --device 0 \
  --save_interval 5 \
  --rasp_enable \
  --rasp_warmup_epochs 5 \
  --rasp_cycle_epochs 3 \
  --rasp_reliability_threshold 0.50 \
  --rasp_importance_beta 0.90 \
  --rasp_min_hidden 16 \
  --rasp_min_keep_ratio 0.50 \
  --rasp_round_to 8 \
  --rasp_gmm_posterior 0.80 \
  --rasp_gmm_bic_gain 10 \
  --rasp_gmm_min_separation 1.0 \
  --rasp_gmm_min_samples 16 \
  --rasp_cost_gamma 1.0 \
  --rasp_max_step_cost_fraction 0.05 \
> logs/c2f_rasp_yolo26m.log 2>&1 &
```

Main defaults:

```text
Warm-up                  5 epochs
Recovery interval        3 epochs
Reliability gate         0.50
Taylor EMA beta          0.90
Minimum keep ratio       50% / Bottleneck
Pack size                8 channels
GMM low posterior        0.80
GMM BIC gain             10
GMM separation           1.0
Max new cost/event       5% baseline prunable MACs
```

Đây **không phải fixed global sparsity target**. Final pruning ratio được tìm adaptively.

### Files quan trọng

```text
runs/c2f_rasp_yolo26m/checkpoints/
├── yolo26_rasp_latent_epoch_*.pt
├── rasp_training_state_epoch_*.pt
└── rasp_training_state_latest.pt
```

Final:

```text
rasp_training_state_epoch_60.pt
yolo26_rasp_latent_epoch_60.pt
```

### Resume RASP

```bash
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
  --device 0 \
  --save_interval 5 \
  --rasp_enable \
  --resume_state runs/c2f_rasp_yolo26m/checkpoints/rasp_training_state_latest.pt \
> logs/c2f_rasp_yolo26m_resume.log 2>&1 &
```

Resume state khôi phục Teacher, dense latent Student, optimizer, scheduler, masks, Taylor EMA và pruning-cycle state.

---

# 12. Physical compact export

Training-time masked Student vẫn lưu dense tensors; Params trên disk chưa thật sự giảm.

Chỉ sau physical export mới được report compact Params/FLOPs/model size.

```bash
nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/export_rasp_compact.py \
  --state runs/c2f_rasp_yolo26m/checkpoints/rasp_training_state_epoch_60.pt \
  --latent_model runs/c2f_rasp_yolo26m/checkpoints/yolo26_rasp_latent_epoch_60.pt \
  --out runs/c2f_rasp_yolo26m/yolo26m_rasp_compact.pt \
  --device 0 \
  --verify \
  --count_macs \
> logs/c2f_rasp_export_compact.log 2>&1 &
```

Bắt buộc kiểm tra:

```text
masked dense Student ≈ physical compact Student
```

Nếu equivalence check fail, **không report compact efficiency** cho checkpoint đó.

---

# 13. Final evaluation — chỉ lúc này dùng Foggy val labels

## 13.1. RASP compact

```bash
nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/eval_rasp_student.py \
  --model runs/c2f_rasp_yolo26m/yolo26m_rasp_compact.pt \
  --data dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml \
  --imgsz 1024 \
  --batch 4 \
  --device 0 \
> logs/c2f_rasp_compact_eval.log 2>&1 &
```

## 13.2. Đánh giá cả 4 checkpoint trên cùng Foggy val

```bash
PYTHONPATH="$PWD" python - <<'PY'
import json
from pathlib import Path
from ultralytics import YOLO

DATA = 'dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml'
MODELS = {
    'source_only': 'runs/c2f_source_yolo26m/weights/best.pt',
    'stage1_adabn': 'runs/c2f_stage1_yolo26m/yolo26_stage1_adabnrc_foggy_cityscapes.pt',
    'rtsfod_y26': 'runs/c2f_rtsfod_yolo26m/checkpoints/yolo26_stage2_rtsfod_epoch_60.pt',
    'rasp_compact': 'runs/c2f_rasp_yolo26m/yolo26m_rasp_compact.pt',
}

results = {}
for name, ckpt in MODELS.items():
    if not Path(ckpt).exists():
        print('[SKIP missing]', name, ckpt)
        continue
    print('\n===', name, '===')
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
        'checkpoint': ckpt,
        'params': int(params),
        'params_M': params / 1e6,
        'mAP50_95': float(r.box.map),
        'mAP50': float(r.box.map50),
        'mAP75': float(r.box.map75),
        'precision': float(r.box.mp),
        'recall': float(r.box.mr),
    }
    print(json.dumps(results[name], indent=2))

Path('runs/final_metrics_c2f.json').write_text(json.dumps(results, indent=2))
print('\nSaved -> runs/final_metrics_c2f.json')
PY
```

Primary C2F comparison nên dựa trên cùng:

```text
same source checkpoint
same Stage-1 checkpoint
same target train split
same image size
same training length
same final Foggy val split
```

---

# 14. Bảng kết quả cuối cùng nên report

| Model | Target eval | mAP50 | mAP50-95 | Precision | Recall | Params | MACs/FLOPs | Latency/FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Source-only YOLO26-M | Foggy val | | | | | | | |
| Stage-1 AdaBN | Foggy val | | | | | | | |
| RT-SFOD-Y26 | Foggy val | | | | | | | |
| **RASP-SFOD-Y26 compact** | **Foggy val** | | | | | | | |

Latency/FPS chỉ so khi cùng GPU, precision, input shape, batch size và timing protocol.

---

# 15. GOOGLE COLAB T4 — project nằm trong Google Drive

Giả sử Drive:

```text
MyDrive/
└── MedRT-SFOD/
    ├── dataset/
    ├── scripts/
    ├── ultralytics/
    ├── colab/
    └── ...
```

Mở Colab:

```text
Runtime → Change runtime type → T4 GPU
```

## 15.1. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 15.2. Config

```python
from pathlib import Path

DRIVE_PROJECT = Path('/content/drive/MyDrive/MedRT-SFOD')
LOCAL_PROJECT = Path('/content/MedRT-SFOD')
DRIVE_RUNS = DRIVE_PROJECT / 'runs_t4'

DEVICE = '0'
IMGSZ = 1024
BATCH = 4       # T4 OOM -> 2
WORKERS = 2

assert DRIVE_PROJECT.exists(), DRIVE_PROJECT
DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
```

## 15.3. Check GPU và disk trước khi copy

```python
import torch, os
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

os.system('nvidia-smi')
os.system('df -h /content')
os.system('du -sh /content/drive/MyDrive/MedRT-SFOD')
```

Nếu project/dataset gần bằng hoặc lớn hơn dung lượng `Avail` của `/content`, **không copy toàn bộ dataset xuống local**.

---

# 16. Colab Mode A — đủ disk, copy project + dataset xuống `/content`

Đây là mode nhanh hơn khi training vì đọc ảnh từ local SSD.

**Không dùng `rsync -a` nếu bạn đã gặp exit status 11.** Dùng `shutil.copytree()`:

```python
from pathlib import Path
import shutil, os

DRIVE_PROJECT = Path('/content/drive/MyDrive/MedRT-SFOD')
LOCAL_PROJECT = Path('/content/MedRT-SFOD')

if LOCAL_PROJECT.exists():
    shutil.rmtree(LOCAL_PROJECT)

IGNORE = shutil.ignore_patterns(
    '.git', '.venv', '__pycache__',
    'runs', 'runs_t4', '.DS_Store'
)

print('FROM:', DRIVE_PROJECT)
print('TO  :', LOCAL_PROJECT)

shutil.copytree(DRIVE_PROJECT, LOCAL_PROJECT, ignore=IGNORE)
os.chdir(LOCAL_PROJECT)
print('[OK] cwd =', Path.cwd())
```

Sau đó kiểm tra disk lại:

```python
!df -h /content
```

### Rewrite YAML local cho Colab

```python
from pathlib import Path
import yaml

SRC_YAML = LOCAL_PROJECT / 'dataset/c2f_yolo/cityscapes/cityscapes.yaml'
TGT_YAML = LOCAL_PROJECT / 'dataset/c2f_yolo/foggy_cityscapes/foggy_cityscapes.yaml'

def rewrite(yaml_path, root):
    d = yaml.safe_load(yaml_path.read_text())
    d['path'] = str(Path(root).resolve())
    yaml_path.write_text(yaml.safe_dump(d, sort_keys=False))

rewrite(SRC_YAML, LOCAL_PROJECT/'dataset/c2f_yolo/cityscapes')
rewrite(TGT_YAML, LOCAL_PROJECT/'dataset/c2f_yolo/foggy_cityscapes')

print(SRC_YAML.read_text())
print(TGT_YAML.read_text())
```

Drive YAML gốc không bị sửa.

---

# 17. Colab Mode B — disk thấp, code local nhưng dataset ở Drive

Mode này tiết kiệm `/content`, nhưng training có thể chậm hơn vì đọc nhiều ảnh từ mounted Drive.

## 17.1. Copy code, bỏ dataset

```python
from pathlib import Path
import shutil, os

DRIVE_PROJECT = Path('/content/drive/MyDrive/MedRT-SFOD')
LOCAL_PROJECT = Path('/content/MedRT-SFOD')

if LOCAL_PROJECT.exists():
    shutil.rmtree(LOCAL_PROJECT)

IGNORE = shutil.ignore_patterns(
    '.git', '.venv', '__pycache__',
    'runs', 'runs_t4', '.DS_Store',
    'dataset'
)

shutil.copytree(DRIVE_PROJECT, LOCAL_PROJECT, ignore=IGNORE)
os.chdir(LOCAL_PROJECT)
print('[OK] code copied')
```

## 17.2. Tạo runtime YAML trỏ thẳng tới Drive dataset

```python
from pathlib import Path
import yaml

DRIVE_DATA = DRIVE_PROJECT / 'dataset/c2f_yolo'
RUNTIME_CFG = LOCAL_PROJECT / 'runtime_cfg'
RUNTIME_CFG.mkdir(exist_ok=True)

SRC_YAML = RUNTIME_CFG / 'cityscapes_colab.yaml'
TGT_YAML = RUNTIME_CFG / 'foggy_cityscapes_colab.yaml'

names = {
    0: 'person', 1: 'rider', 2: 'car', 3: 'truck',
    4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle'
}

SRC_YAML.write_text(yaml.safe_dump({
    'path': str(DRIVE_DATA/'cityscapes'),
    'train': 'images/train',
    'val': 'images/val',
    'names': names,
}, sort_keys=False))

TGT_YAML.write_text(yaml.safe_dump({
    'path': str(DRIVE_DATA/'foggy_cityscapes'),
    'train': 'images/train',
    'val': 'images/val',
    'names': names,
}, sort_keys=False))

print(SRC_YAML.read_text())
print(TGT_YAML.read_text())
```

Sau đó mọi command dùng `SRC_YAML` / `TGT_YAML` runtime này.

---

# 18. Colab install / preflight

```python
import os, sys, subprocess
os.chdir(LOCAL_PROJECT)

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'scikit-learn', 'kneed', 'torch-pruning', 'thop'
], check=True)

required = [
    'ultralytics/cfg/models/26/yolo26.yaml',
    'scripts/YOLO26/train_source_supervised.py',
    'scripts/YOLO26/stage0_stage1_adabn_rc_yolo26_v2.py',
    'scripts/YOLO26/stage2_rtsfod_yolo26.py',
    'scripts/YOLO26/rasp_pruning.py',
    'scripts/YOLO26/stage2_rasp_rtsfod_yolo26.py',
    'scripts/YOLO26/inspect_rasp_prunable_groups.py',
    'scripts/YOLO26/export_rasp_compact.py',
    'scripts/YOLO26/eval_rasp_student.py',
]

missing = [p for p in required if not (LOCAL_PROJECT/p).exists()]
assert not missing, f'Missing: {missing}'
print('[OK] required files exist')
```

Smoke test:

```python
import subprocess, os

env = os.environ.copy()
env['PYTHONPATH'] = str(LOCAL_PROJECT)
subprocess.run(
    ['python', 'scripts/YOLO26/test_rasp_smoke.py'],
    cwd=LOCAL_PROJECT,
    env=env,
    check=True,
)
```

---

# 19. Colab output policy

**Code/dataset có thể chạy từ `/content`, nhưng checkpoint nên lưu thẳng vào Drive** để runtime disconnect không làm mất kết quả.

Ví dụ:

```python
SOURCE_OUT   = DRIVE_RUNS / 'c2f_source_yolo26m'
STAGE1_OUT   = DRIVE_RUNS / 'c2f_stage1_yolo26m'
BASELINE_OUT = DRIVE_RUNS / 'c2f_rtsfod_yolo26m'
RASP_OUT     = DRIVE_RUNS / 'c2f_rasp_yolo26m'
```

Expected:

```text
MyDrive/MedRT-SFOD/runs_t4/
├── c2f_source_yolo26m/
├── c2f_stage1_yolo26m/
├── rasp_audit_yolo26m.json
├── c2f_rtsfod_yolo26m/
├── c2f_rasp_yolo26m/
└── final_metrics_c2f.json
```

---

# 20. Colab commands tương ứng

Trong notebook nên dùng helper:

```python
import subprocess, os

def run_shell(cmd):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(LOCAL_PROJECT)
    print('>>>', cmd)
    subprocess.run(cmd, shell=True, cwd=LOCAL_PROJECT, env=env, check=True)
```

Sau đó chạy đúng sequence:

```text
1. Source training nếu chưa có SOURCE_CKPT
2. Stage 1 AdaBN
3. RASP audit
4. RT-SFOD baseline
5. RASP-SFOD
6. Physical compact export
7. Final Foggy val evaluation
```

Các CLI arguments giống hệt phần **Local GPU**, chỉ thay:

```text
runs/...  →  /content/drive/MyDrive/MedRT-SFOD/runs_t4/...
```

và `SRC_YAML`, `TGT_YAML` bằng runtime paths đã tạo trong Colab.

---

# 21. Colab session restart / resume

Mỗi session mới:

```text
1. Bật T4 GPU
2. Mount Drive
3. Check disk
4. Copy code/dataset theo Mode A hoặc B
5. Install dependencies
6. Recreate yolo26m.yaml alias
7. Recreate runtime YAML nếu dùng Mode B
8. Skip stage đã có final checkpoint
9. RASP resume từ rasp_training_state_latest.pt nếu chưa xong
```

RASP resume được.

Original RT-SFOD baseline hiện không exact-resume đầy đủ, nên cố chạy baseline 60 epochs trọn một session; nếu ngắt, restart baseline từ Stage 1.

---

# 22. Nếu đã có source checkpoint thì Colab/local skip gì?

Nếu có:

```text
SOURCE_CKPT = <YOLO26-M trained on labeled Clear Cityscapes>
```

thì bỏ toàn bộ **Stage -1 source training**.

Flow còn:

```text
SOURCE_CKPT
   ↓
AdaBN
   ↓
RASP audit
   ↓
RT-SFOD baseline
   ↓
RASP-SFOD
   ↓
compact export
   ↓
final Foggy val evaluation
```

Không được dùng một generic COCO `yolo26m.pt` rồi gọi nó là C2F source checkpoint nếu nó chưa được train/fine-tune trên labeled Cityscapes.

---

# 23. Model-selection rules

## Được phép

```text
Source training:
- Clear train labels
- Clear val labels
- chọn source best.pt theo Clear val

Source-free adaptation:
- Foggy train images
- Teacher pseudo-labels
- DHF confidence
- target Taylor importance
- GMM / Kneedle / RASP internal unsupervised signals

Final reporting:
- Foggy val labels
```

## Không được phép trong main SFOD/RASP run

```text
- Foggy train ground-truth labels
- Foggy val mAP để early stop
- Foggy val mAP để chọn pruning event
- Foggy val mAP để chọn pruning ratio
- Foggy val mAP để chọn best epoch/checkpoint
- --oracle_early_stop
- --eval trong Stage 1 / Stage 2 main experiment
```

---

# 24. Troubleshooting

## CUDA OOM trên T4

Ưu tiên:

```text
BATCH=4 → BATCH=2
```

Giữ `IMGSZ=1024` cho main comparison nếu có thể.

## `rsync` exit status 11 trên Colab

Check:

```python
!df -h /content
!du -sh /content/drive/MyDrive/MedRT-SFOD
```

Xóa local copy dở:

```python
!rm -rf /content/MedRT-SFOD
```

Sau đó dùng `shutil.copytree()` hoặc chuyển sang **Mode B**.

## YAML vẫn trỏ `/Users/...`

Không train cho tới khi YAML runtime trỏ đúng local/Drive path.

## `RASP found 0 eligible Bottleneck groups`

Dừng full training. Kiểm tra exact YOLO26 fork/graph và pruning audit.

## Physical export equivalence fail

Không report compact Params/FLOPs. Kiểm tra masks/state/checkpoint pairing.

## Source model params không giống M

Dừng. Verify `yolo26m.yaml` / architecture trước khi train 100 epochs.

---

# 25. Recommended experiment order cho paper/thesis

```text
A. Source-only YOLO26-M
B. Stage-1 AdaBN
C. RT-SFOD-Y26 baseline
D. RASP-SFOD-Y26 main method
E. Physical compact RASP Student
F. Post-adaptation/static pruning control
G. Fixed-budget/importance-only ablation
H. GMM vs Otsu ablation nếu cần
I. Knee vs fixed budget
J. Reliability gate on/off
K. Quantization/QAT chỉ sau khi pruning-only ổn định
```

Core comparison:

```text
Same source checkpoint
Same Stage-1 checkpoint
Same YOLO26-M
Same target train images
Same RT-SFOD settings
Same training length

RT-SFOD-Y26
     vs
RASP-SFOD-Y26
```

---

# 26. Final checklist trước khi báo kết quả

```text
[ ] CUDA/GPU đúng
[ ] Dataset YAML đúng environment path
[ ] Source checkpoint thực sự là YOLO26-M Cityscapes source model
[ ] Source Clear-val performance đã kiểm tra
[ ] Stage 1 chạy không dùng target labels
[ ] RASP audit pass
[ ] Baseline và RASP dùng cùng Stage-1 checkpoint
[ ] Không bật --eval trong main Stage 1/2
[ ] RASP final state + latent checkpoint tồn tại
[ ] Physical compact equivalence pass
[ ] Final Foggy-val evaluation chỉ chạy sau training
[ ] Params/FLOPs lấy trên physically compact model
[ ] Latency/FPS đo cùng hardware + precision + input + batch
[ ] Không dùng Foggy val để model selection
```

---

# 27. Các checkpoint quan trọng cần giữ lại

```text
1. Source:
   c2f_source_yolo26m/weights/best.pt

2. Stage 1:
   c2f_stage1_yolo26m/yolo26_stage1_adabnrc_foggy_cityscapes.pt

3. RT-SFOD baseline:
   c2f_rtsfod_yolo26m/checkpoints/yolo26_stage2_rtsfod_epoch_60.pt

4. RASP state:
   c2f_rasp_yolo26m/checkpoints/rasp_training_state_epoch_60.pt

5. RASP dense latent:
   c2f_rasp_yolo26m/checkpoints/yolo26_rasp_latent_epoch_60.pt

6. RASP physical compact:
   c2f_rasp_yolo26m/yolo26m_rasp_compact.pt

7. Final metrics:
   final_metrics_c2f.json
```

Không xóa source checkpoint hoặc Stage-1 checkpoint sau khi train xong, vì chúng cần để chứng minh baseline và RASP xuất phát từ cùng initialization.

---

## Tóm tắt lệnh chạy theo thứ tự

```text
PRE-FLIGHT
  ↓
CUDA check
  ↓
RASP smoke
  ↓
YOLO26-M verify
  ↓
SOURCE checkpoint có chưa?
  ├─ Có → skip source train
  └─ Chưa → Clear Cityscapes supervised train → best.pt
  ↓
Stage 1 AdaBN — Foggy train images only
  ↓
RASP audit
  ↓
RT-SFOD-Y26 baseline 60 epochs
  ↓
RASP-SFOD-Y26 60 epochs
  ↓
Physical compact export + equivalence
  ↓
Foggy val final evaluation
  ↓
Report accuracy + compactness + runtime
```
