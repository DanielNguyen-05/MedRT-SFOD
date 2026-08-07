# RT-SFOD-Lite (a.k.a. MedRT-SFOD LiteSeg)

Mở rộng của **RT-SFOD** (Real-Time Source-Free Object Detection, ECCV 2026) theo hướng: giảm nhẹ mô hình hơn nữa (Try 1 + Try 2, đã code + test), rồi mở rộng sang instance segmentation (Try 3, mới có kiến trúc) và domain y tế — polyp nội soi (Try 4, mới có pipeline convert dữ liệu + training nguồn).

File này là README riêng cho phần mở rộng. README gốc của repo Ultralytics/RT-SFOD nằm ở `README.md` (không đổi).

> **Đổi so với bản trước:** bản README trước có 1 phần ("Nhánh B" — benchmark/dataset/training cho city+polyp) mô tả các script **không thực sự chạy đúng chức năng đã ghi** (xem mục 7 "Đã sửa gì"). Toàn bộ phần đó đã được viết lại trong bản này.

---

## 1. Proposed Method

### 1.1. Bối cảnh
RT-SFOD gốc giải quyết source-free domain adaptation (không cần dữ liệu nguồn có nhãn) cho object detection real-time, dựa trên YOLOv10/YOLO26 (dual-head NMS-free: O2O + O2M), với 2 module:
- **DHF (Dual-Head Fusion)**: hợp nhất pseudo-label từ 2 head (O2O chính xác nhưng recall thấp, O2M recall cao nhưng nhiễu).
- **MARD (Multi-scale Adaptive Representation Diversification)**: regularize variance + covariance trên feature đa tỉ lệ (P3/P4/P5) để chống collapse biểu diễn do domain-shift.

Cả 2 module đều **chỉ hoạt động lúc train**, không tốn thêm chi phí lúc inference.

### 1.2. Đề xuất mở rộng — 4 giai đoạn (Try)

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| **Try 1** | Backbone nhẹ dựa trên **Partial Convolution / FasterNet** (`C2fFaster`), giữ nguyên neck PAN + Detect head dual-head. | ✅ Code + test (logic/shape) |
| **Try 2** | **CARD**: nén student **trong lúc** self-training (pruning channel + QAT, cả 2 theo schedule ramp+confidence-gate như MARD). Teacher giữ full-precision. | ✅ Code + test (logic/shape) |
| **Try 3** | Mask branch (instance segmentation kiểu YOLO-seg) trên backbone nhẹ. | 🟡 Chỉ mới có **kiến trúc** (`yolo26-lite-seg.yaml`, build+forward đã test) — **chưa** mở rộng DHF sang mask-IoU, **chưa** có self-training loop cho mask |
| **Try 4** | Domain y tế — polyp nội soi (Kvasir-SEG), có tiền lệ SFOD để so sánh (SMPT/SMPT++, FSM). | 🟡 Có converter data + script train nguồn (supervised) đã test logic — **chưa** chạy self-training source-free thật trên polyp, **chưa** tải/test với dataset Kvasir-SEG thật |

Chạy `python3 scripts/YOLO26/medrt_liteseg_cli.py` để xem trạng thái này tự động kiểm tra theo file thật đang có trong repo (không phải danh sách gõ tay).

### 1.3. Kiến trúc Try 1 + Try 2 (đã implement)

```
Ảnh target domain (không nhãn)
        │
   weak aug ─────────────► Teacher (yolo26-lite, FULL PRECISION, EMA mỗi epoch)
        │                        │
        │                  O2O preds + O2M preds
        │                        │
        │                       DHF (không đổi so với RT-SFOD gốc)
        │                        │
        │                  pseudo-labels Ŷ^w
        │                        │  (warp qua hình học của strong-aug)
   strong aug ───────────► Student (yolo26-lite)
        │                        │
        │              ┌─────────┴─────────┐
        │        Detection loss      MARD loss (không đổi)
        │        (box+cls+dfl)       (variance+covariance trên P3/P4/P5)
        │                 │                 │
        │                 └──────┬──────────┘
        │                  optimizer.step()
        │                        │
        │                  CARD.step()
        │                  ├─ ChannelPruner.apply_masks()  (mỗi step)
        │                  ├─ ChannelPruner.update_masks() (mỗi N step, theo target sparsity)
        │                  └─ set_qat_ratio() trên các Conv2d đã patch (mỗi step)
```

---

## 2. Cấu trúc thư mục

```
RT-SFOD/
├── README.md                          # gốc — không đổi
├── README_RTSFOD_LITE.md              # file này
├── run_polyp_training.sh              # 🆕 (viết lại) convert + train polyp, không hardcode path máy cá nhân
├── scripts/YOLO26/
│   ├── stage0_stage1_adabn_rc_yolo26.py     # gốc — AdaBN warm-start
│   ├── stage2_rtsfod_yolo26.py              # gốc — DHF + MARD + Mean-Teacher
│   ├── stage2_card_rtsfod_yolo26.py         # = stage2 gốc + CARD (Try 2)
│   ├── card_compression.py                  # module CARD
│   ├── test_card_lite_smoke.py              # smoke test Try 1+2 (không cần GPU/dataset)
│   ├── architecture_benchmark.py            # 🆕 (viết lại) benchmark params/FPS trên KIẾN TRÚC THẬT
│   ├── architecture_cli.py                  # 🆕 (viết lại) CLI cho benchmark trên
│   ├── domain_benchmark.py                  # 🆕 (viết lại) list ảnh theo convention path đúng
│   ├── medrt_liteseg_cli.py                 # 🆕 (viết lại) status check dựa trên file thật
│   ├── polyp_kvasir_to_yolo.py              # 🆕 converter Kvasir-SEG → chuẩn YOLO (Try 4)
│   └── train_source_supervised.py           # 🆕 train nguồn (city/polyp, detect/segment) dùng YOLO(...).train() thật
└── ultralytics/
    ├── nn/modules/lite_block.py             # PConv / FasterNetBlock / C2fFaster (Try 1)
    ├── nn/tasks.py                          # ✏️ đăng ký C2fFaster
    └── cfg/
        ├── models/26/
        │   ├── yolo26.yaml                 # gốc
        │   ├── yolo26-seg.yaml              # gốc
        │   ├── yolo26-lite.yaml             # Try 1 — backbone nhẹ, detect
        │   └── yolo26-lite-seg.yaml         # 🆕 Try 3 — backbone nhẹ, segment (chỉ kiến trúc)
        └── datasets/
            └── c2f_example.yaml             # template — copy & sửa path thật
```

### File đã **xóa hoàn toàn** khỏi project (không sửa được, viết lại thay thế)
| File cũ | Vấn đề | Thay bằng |
|---|---|---|
| `train_polyp_detection.py` | `loss = outputs.mean()` — **không dùng ground-truth box**, không học gì | `train_source_supervised.py --task detect` |
| `train_city_detection.py` | Load `split="test"` để training (bug copy-paste); target là mean/std màu ảnh, không phải nhãn detection | `train_source_supervised.py --task detect` |
| `train_polyp_segmentation.py` | Chạy đúng loss (điểm duy nhất đúng) nhưng dùng model tự chế, không liên quan `yolo26-lite-seg.yaml` | `train_source_supervised.py --task segment` |
| `city_dataset_utils.py` | `__getitem__` trả "target" là thống kê màu ảnh, không phải nhãn | không cần nữa — dùng chuẩn YOLO dataset + `--data <yaml>` |
| `polyp_dataset_utils.py` | Chỉ đọc `bbox[0]`, bỏ sót polyp thứ 2+ trong ảnh nhiều polyp | `polyp_kvasir_to_yolo.py` (đã fix, có test) |
| `polyp_mask_head.py` | Encoder-decoder rời, không liên quan `yolo26-lite-seg.yaml`/`Segment26` | `yolo26-lite-seg.yaml` + `SegmentationModel` |

---

## 3. Cài đặt

### 3.1. Yêu cầu
- Python ≥ 3.10, GPU CUDA khuyến nghị (paper gốc dùng 1× RTX A6000)
- `torch`, `torchvision`, `opencv-python`, `pyyaml`, `numpy`

```bash
cd RT-SFOD
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # chọn bản cu12x khớp driver
pip install opencv-python pyyaml numpy
```

### 3.2. Chạy smoke test trước (không cần data/GPU)
```bash
python3 scripts/YOLO26/test_card_lite_smoke.py          # Try 1 + Try 2
python3 scripts/YOLO26/architecture_cli.py --device cpu # params/FPS trên kiến trúc thật (Try 1 + Try 3)
python3 scripts/YOLO26/medrt_liteseg_cli.py              # trạng thái roadmap dựa trên file thật
```

### 3.3. ⚠️ Vá `ultralytics/data/` (bắt buộc trước khi train thật — không phải do project này gây ra)
Repo GitHub gốc thiếu hẳn thư mục `ultralytics/data/`. `from ultralytics import YOLO` sẽ lỗi nếu chưa vá:
```bash
pip install ultralytics --target /tmp/ul_ref
cp -r /tmp/ul_ref/ultralytics/data ultralytics/data
```

---

## 4. Chuẩn bị dữ liệu — data lưu ở đâu?

### 4.1. Convention duy nhất (áp dụng cho MỌI domain, city và polyp)
Đặt data **ngoài** thư mục `RT-SFOD/`, ngang hàng với nó:
```
<workdir>/
├── RT-SFOD/                   # project
└── datasets/                   # data — KHÔNG nằm trong RT-SFOD/
    ├── cityscapes/             # domain NGUỒN (Cityscapes có nhãn)
    │   └── images,labels/{train,val}/
    ├── foggy_cityscapes/       # domain TARGET (C2F) — self-training không dùng nhãn thật
    │   └── images,labels/{train,val}/     # labels/val chỉ để tính mAP theo dõi
    ├── polyp_detect/           # ⟵ output của polyp_kvasir_to_yolo.py --task detect
    │   └── images,labels/{train,val}/, dataset_detect.yaml
    └── polyp_seg/              # ⟵ output của polyp_kvasir_to_yolo.py --task segment
        └── images,labels/{train,val}/, dataset_seg.yaml
```
`domain_benchmark.py` và `run_polyp_training.sh` đều dùng `<repo>/../datasets` làm default (override được qua `--datasets-root` hoặc biến môi trường `DATASETS_ROOT`).

### 4.2. Cityscapes/Foggy Cityscapes/Sim10k/KITTI/BDD100k
Cần tự convert từ format gốc (Cityscapes polygon JSON, KITTI txt, v.v.) sang YOLO txt — chưa có converter tự động trong repo cho các dataset này (khác license/cách tải, cần đăng ký tài khoản riêng từng nơi).

### 4.3. Polyp (Kvasir-SEG) — ĐÃ có converter
```bash
# 1. Tự tải Kvasir-SEG từ trang chủ dataset (cần vào https://datasets.simula.no/kvasir-seg/), giải nén, có sẵn:
#    <raw>/images/*.jpg  <raw>/masks/*.jpg  <raw>/kavsir_bboxes.json

# 2a. Convert cho detection
python3 scripts/YOLO26/polyp_kvasir_to_yolo.py \
    --src /path/to/raw/Kvasir-SEG --dst ../datasets/polyp_detect --task detect

# 2b. Convert cho segmentation (polygon từ mask, hỗ trợ nhiều polyp/ảnh — đã test)
python3 scripts/YOLO26/polyp_kvasir_to_yolo.py \
    --src /path/to/raw/Kvasir-SEG --dst ../datasets/polyp_seg --task segment
```
Converter đã được test với dữ liệu giả (ảnh + mask 2 vùng tách biệt + json 2 box) — xác nhận trích đúng **cả 2** box/polygon, không chỉ cái đầu tiên như bug cũ.

---

## 5. Cách chạy

### 5.1. Train nguồn (source-supervised) — city hoặc polyp, detect hoặc segment
```bash
# Cityscapes, detection
python3 scripts/YOLO26/train_source_supervised.py \
    --model-cfg ultralytics/cfg/models/26/yolo26-lite.yaml \
    --data ultralytics/cfg/datasets/c2f_example.yaml \
    --task detect --epochs 100 --imgsz 1024 --out-dir runs/city_source

# Polyp, detection (sau khi convert --task detect ở mục 4.3)
python3 scripts/YOLO26/train_source_supervised.py \
    --model-cfg ultralytics/cfg/models/26/yolo26-lite.yaml \
    --data ../datasets/polyp_detect/dataset_detect.yaml \
    --task detect --epochs 100 --imgsz 640 --out-dir runs/polyp_detect_source

# Polyp, segmentation (sau khi convert --task segment ở mục 4.3)
python3 scripts/YOLO26/train_source_supervised.py \
    --model-cfg ultralytics/cfg/models/26/yolo26-lite-seg.yaml \
    --data ../datasets/polyp_seg/dataset_seg.yaml \
    --task segment --epochs 100 --imgsz 640 --out-dir runs/polyp_seg_source
```
Hoặc gộp bước convert + train polyp detection bằng 1 lệnh: `./run_polyp_training.sh /path/to/raw/Kvasir-SEG detect`.

> Script này thay hoàn toàn 3 script training cũ — dùng đúng `YOLO(cfg).train(data=...)` của Ultralytics (loss/target-assignment đúng, không tự chế) thay vì loop tay.

### 5.2. Stage 0+1 — AdaBN warm-start (chỉ áp dụng cho detection driving-scene hiện tại; polyp/segmentation chưa nối vào Stage 0-2)
```bash
python3 scripts/YOLO26/stage0_stage1_adabn_rc_yolo26.py \
    --weights runs/city_source/weights/best.pt \
    --data ultralytics/cfg/datasets/c2f_example.yaml \
    --out_dir runs/lite_c2f/stage1 --epochs_adabn 2 --early_stop
```

### 5.3. Stage 2 — DHF + MARD, có/không CARD
```bash
python3 scripts/YOLO26/stage2_card_rtsfod_yolo26.py \
    --weights runs/lite_c2f/stage1/adapted_best.pt \
    --data ultralytics/cfg/datasets/c2f_example.yaml \
    --out_dir runs/lite_c2f/stage2_card --epochs 60 \
    --card_enable --card_prune_target 0.3 --card_quant_bits 8 --card_warmup_epochs 10
```

### 5.4. Bảng ablation nên chạy trước khi viết paper
| Cấu hình | Lệnh khác biệt |
|---|---|
| (a) yolo26 gốc + DHF/MARD | `yolo26.yaml`, `stage2_rtsfod_yolo26.py` |
| (b) yolo26-lite + DHF/MARD, không CARD | `yolo26-lite.yaml`, `stage2_rtsfod_yolo26.py` |
| (c) yolo26-lite + DHF/MARD + CARD | `yolo26-lite.yaml`, `stage2_card_rtsfod_yolo26.py --card_enable` |
| (d) như (c) nhưng tắt gate | thêm `--card_gate_threshold 0` |

---

## 6. Benchmark kiến trúc (Table 1-style)
```bash
python3 scripts/YOLO26/architecture_cli.py --arch all --device cpu   # đổi --device cuda nếu có GPU
```
Benchmark này đo **trực tiếp trên `yolo26.yaml`/`yolo26-lite.yaml`/`yolo26-lite-seg.yaml` thật** (qua `DetectionModel`/`SegmentationModel` của Ultralytics), không phải CNN tự chế như bản trước — số liệu CPU hiện tại chỉ để demo pipeline chạy được, **số liệu paper-worthy cần đo trên cùng GPU với Table 1 gốc (RTX A6000, FP32 PyTorch)**.

---

## 7. Đã sửa gì so với bản trước (đợt code review)

Bản trước có 1 nhóm script ("Nhánh B") tồn tại song song với pipeline RT-SFOD thật, không đụng gì tới `yolo26-lite.yaml`/DHF/MARD/CARD, và có **3 lỗi logic thật** (không phải chỉ thiếu tính năng):
1. `train_polyp_detection.py`: loss không dùng ground-truth → không học detection dù script chạy không lỗi.
2. `train_city_detection.py`: load nhầm split `"test"` để train; "nhãn" thực chất là thống kê màu ảnh.
3. `polyp_dataset_utils.py`: chỉ đọc box/polygon đầu tiên mỗi ảnh, bỏ sót polyp thứ 2+.

Toàn bộ được viết lại (mục 2, bảng "File đã xóa hoàn toàn"), test bằng dữ liệu giả để xác nhận bug đa-box đã hết, và benchmark/status CLI giờ đo trên kiến trúc thật + kiểm tra file thật thay vì báo cáo tĩnh.

---

## 8. Việc chưa làm / hạn chế đã biết

- Chưa chạy trên GPU thật + dataset thật (Cityscapes/Foggy Cityscapes/Kvasir-SEG) — mọi thứ mới verify về **logic/shape** (smoke test, converter test với data giả, benchmark chạy được trên CPU).
- Try 3: DHF chưa mở rộng sang mask-IoU; chưa có self-training loop (Mean-Teacher + MARD) cho segmentation — hiện chỉ có kiến trúc `yolo26-lite-seg.yaml`.
- Try 4: chưa nối polyp vào Stage 0-2 (AdaBN + DHF/MARD/CARD source-free) — hiện chỉ có converter dữ liệu + train nguồn supervised.
- `export_pruned_state_dict()` trong `card_compression.py` chủ động `raise NotImplementedError` (pruning hiện là mask-based, chưa export ra checkpoint nhỏ thật).

---

## 9. Trích dẫn
```bibtex
@inproceedings{sairam2026rtsfod,
  title={Real-Time Source-Free Object Detection},
  author={Sairam VCR and Gopal, Varun and Jain, Poornima and Balasubramanian, Vineeth N and Khan, Muhammad Haris},
  booktitle={ECCV},
  year={2026}
}
```
