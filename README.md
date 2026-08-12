## 1. Đặt code vào branch `seg`

Sau khi đưa các file vào project, cấu trúc phải là:

```text
~/MedRT-SFOD/
└── scripts/
    └── YOLO26/
        └── medseg/
            ├── prepare_kvasir.py
            ├── check_yolo26s_seg.py
            ├── train_source_seg.py
            └── eval_source_seg.py
```

Sau đó:

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"

mkdir -p \
  logs/seg/source \
  logs/seg/benchmarks \
  runs/seg/source \
  runs/seg/benchmarks
```

Kiểm tra branch:

```bash
git branch --show-current
```

Phải ra:

```text
seg
```

---

# 2. Bước đầu tiên: prepare Kvasir

Script này **không sửa raw dataset**:

```text
dataset/Kvasir-SEG/
├── images/
└── masks/
```

Nó tạo:

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

Chạy:

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"

python scripts/YOLO26/medseg/prepare_kvasir.py \
  --src dataset/Kvasir-SEG \
  --dst dataset/Kvasir-SEG-YOLO26 \
  --val-fraction 0.20 \
  --seed 29 \
  --overwrite \
2>&1 | tee logs/seg/source/prepare_kvasir.log
```

Mặc định hiện tại là:

```text
80% train
20% val
seed = 29
```

Tức nếu đủ 1000 ảnh:

```text
train = 800
val   = 200
```

Split được ghi hẳn ra file để mọi experiment sau dùng **chính xác cùng split**, không random lại.

Một điểm quan trọng: script giữ riêng:

```text
gt_masks/
```

vì `labels/` phục vụ native YOLO training, còn `gt_masks/` sẽ được dùng để tính **Dice/IoU/Sensitivity/Precision** theo pixel.

---

# 3. Kiểm tra dataset vừa convert

```bash
find dataset/Kvasir-SEG-YOLO26/images/train -type f | wc -l
find dataset/Kvasir-SEG-YOLO26/labels/train -name "*.txt" | wc -l

find dataset/Kvasir-SEG-YOLO26/images/val -type f | wc -l
find dataset/Kvasir-SEG-YOLO26/labels/val -name "*.txt" | wc -l
```

Kỳ vọng với default:

```text
800
800
200
200
```

Xem một polygon label:

```bash
LABEL=$(find dataset/Kvasir-SEG-YOLO26/labels/train \
  -name "*.txt" | head -1)

echo "$LABEL"
head -1 "$LABEL"
```

Dạng:

```text
0 x1 y1 x2 y2 x3 y3 ...
```

Đây là format segmentation mà Ultralytics dùng. ([Ultralytics Docs][2])

---

# 4. Kiểm tra YOLO26-S-Seg

Tiếp theo:

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"

python scripts/YOLO26/medseg/check_yolo26s_seg.py \
  --weights yolo26s-seg.pt \
  --data dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml \
2>&1 | tee logs/seg/source/check_yolo26s_seg.log
```

Nếu chưa có:

```text
yolo26s-seg.pt
```

Ultralytics có thể lấy pretrained segmentation checkpoint khi `YOLO("yolo26s-seg.pt")` được gọi; đây là cách pretrained model được dùng trong tài liệu training chính thức. ([Ultralytics Docs][1])

Điều mình cần nhìn trong output là:

```text
head        : Segment26
end2end     : True
nc          : ...
nm          : ...
npr         : ...
params M    : ...
[PASS] YOLO26-S-Seg ready
```

---

# 5. Smoke test 3 epochs

**Chưa full train ngay.**

Chạy:

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/train_source_seg.py \
  --weights yolo26s-seg.pt \
  --data dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --device 0 \
  --smoke \
> logs/seg/source/kvasir_yolo26s_seg_smoke.log 2>&1 &

echo "PID=$!"
```

Theo dõi:

```bash
tail -f logs/seg/source/kvasir_yolo26s_seg_smoke.log
```

Output nằm đúng convention:

```text
runs/seg/source/
└── kvasir_yolo26s_seg_smoke/
```

Không chạm vào bất kỳ:

```text
runs/c2f_...
logs/c2f_...
```

nào của RASP-SFOD detection.

---

# 6. Nếu smoke PASS mới chạy source 100 epochs

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"

nohup env PYTHONPATH="$PWD" \
python scripts/YOLO26/medseg/train_source_seg.py \
  --weights yolo26s-seg.pt \
  --data dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --device 0 \
> logs/seg/source/kvasir_yolo26s_seg.log 2>&1 &

echo "PID=$!"
```

Checkpoint chính sẽ ở:

```text
runs/seg/source/
└── kvasir_yolo26s_seg/
    └── weights/
        └── best.pt
```

Ultralytics hỗ trợ trực tiếp `.train()` cho YOLO26 segmentation models theo cách này. ([Ultralytics Docs][1])

---

# 7. Evaluator mình cũng đã code sẵn

`eval_source_seg.py` không chỉ lấy native YOLO metrics.

Nó tính cả:

```text
Mask mAP50
Mask mAP50-95

Dice
IoU
Precision
Sensitivity
Specificity
```

Cách đánh giá medical là:

```text
YOLO predicted instances
          ↓
union all predicted polyp masks
          ↓
binary predicted polyp mask
          ↓
compare pixel-wise
          ↓
Kvasir binary GT mask
```

Sau source training:

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"

nohup env PYTHONPATH="$PWD" \
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
> logs/seg/benchmarks/kvasir_yolo26s_seg_source_eval.log 2>&1 &

echo "PID=$!"
```

Kết quả:

```text
runs/seg/benchmarks/
└── kvasir_yolo26s_seg_source_metrics.json
```

