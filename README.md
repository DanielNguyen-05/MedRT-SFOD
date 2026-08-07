# MedRT-SFOD

**Efficient source-free instance segmentation for medical imaging via compression-aware self-training.**

MedRT-SFOD nhắm tới bối cảnh thiết bị y tế biên (nội soi cầm tay, máy siêu âm/X-quang di động): dữ liệu không thể chia sẻ giữa bệnh viện (→ bắt buộc source-free adaptation), thiết bị tính toán yếu (→ cần model nhẹ hơn detector driving-scene thông thường), và bác sĩ cần **vùng bệnh (mask)** chứ không chỉ khung chữ nhật.

Cơ chế self-training lõi (Dual-Head Fusion, Multi-scale Adaptive Representation Diversification) được xây dựa trên kỹ thuật của RT-SFOD (Real-Time Source-Free Object Detection, ECCV 2026) — xem mục 10 để trích dẫn. MedRT-SFOD dùng lại 2 cơ chế đó làm nền, rồi thêm 3 lớp riêng: backbone nhẹ hơn, nén trong-lúc-train (CARD), và mở rộng sang mask + domain y tế.

---

## 1. Kiến trúc đề xuất

```
Ảnh target domain y tế (không nhãn)
        │
   weak aug ─────────────► Teacher (backbone nhẹ C2fFaster, FULL PRECISION, EMA mỗi epoch)
        │                        │
        │                  O2O preds + O2M preds  (dual-head, NMS-free)
        │                        │
        │                    DHF  (fusion pseudo-label 2 head)
        │                        │
        │                  pseudo-labels Ŷ^w  (box, và — roadmap — mask)
        │                        │  (warp qua hình học của strong-aug)
   strong aug ───────────► Student (backbone nhẹ C2fFaster)
        │                        │
        │              ┌─────────┴─────────┐
        │        Detection loss         MARD loss
        │        (box+cls+dfl,          (variance+covariance
        │         — roadmap: +mask)      trên feature đa tỉ lệ P3/P4/P5)
        │                 │                 │
        │                 └──────┬──────────┘
        │                  optimizer.step()
        │                        │
        │                  CARD.step()  ← nén student trong lúc train
        │                  ├─ pruning channel theo target sparsity (ramp+gate)
        │                  └─ quantization-aware training (STE fake-quant)
```

Backbone nhẹ (`C2fFaster`, dựa trên Partial Convolution/FasterNet) và CARD (pruning + QAT in-loop) là phần MedRT-SFOD tự thêm vào cơ chế DHF/MARD gốc — cả 2 chỉ hoạt động lúc train, không tốn chi phí lúc inference.

---

## 2. Đã code được gì / chưa code gì

| Thành phần | Trạng thái | Đã verify bằng |
|---|---|---|
| Backbone nhẹ `C2fFaster` (PConv/FasterNet) | ✅ Code xong | `test_card_lite_smoke.py` — build+forward đúng interface DHF/MARD cần |
| `yolo26-lite.yaml` (detect, backbone nhẹ) | ✅ Code xong | 2.408M params vs 2.572M bản gốc |
| CARD — pruning channel in-loop | ✅ Code xong | test schedule ramp+gate, mask zero đúng target sparsity, forward vẫn chạy sau khi prune |
| CARD — QAT (STE fake-quant) | ✅ Code xong | test gradient chảy qua STE, output ratio=0 vs ratio=1 khác biệt rõ (đã sửa 1 bug binding thật khi test) |
| `yolo26-lite-seg.yaml` (segment, backbone nhẹ) | 🟡 Chỉ có kiến trúc | build+forward test — **chưa** nối DHF/MARD/CARD vào loop train cho mask |
| DHF mở rộng mask-IoU | ❌ Chưa code | — |
| Self-training loop cho segmentation (Mean-Teacher + MARD trên mask) | ❌ Chưa code | — |
| Converter Kvasir-SEG → chuẩn YOLO (`polyp_kvasir_to_yolo.py`) | ✅ Code xong | test với ảnh giả 2-polyp/ảnh — trích đúng cả 2 box và cả 2 polygon (bug cũ trong bản nháp đầu chỉ lấy box/polygon đầu tiên) |
| Train nguồn supervised (`train_source_supervised.py`) cho city/polyp, detect/segment | ✅ Code xong | gọi đúng `YOLO(cfg).train(...)` chuẩn của Ultralytics, chưa chạy trên dataset thật ở đây (không có GPU/dataset trong môi trường code này) |
| Stage 0/1 AdaBN + Stage 2 DHF/MARD/CARD trên **polyp** (source-free thật) | ❌ Chưa code | Hiện Stage 0-2 chỉ được test trên driving-scene (Cityscapes/Foggy). Polyp mới có converter data + train nguồn, **chưa nối vào self-training source-free** |
| Chạy thật trên GPU + dataset thật (mAP/FPS thật) | ❌ Chưa làm | Không có GPU/dataset trong môi trường phát triển code này — mọi số liệu hiện tại chỉ verify logic/shape |
| Export pruned model thành checkpoint nhỏ thật (không chỉ mask=0) | ❌ Chưa code | `export_pruned_state_dict()` chủ động `raise NotImplementedError`, lý do kỹ thuật ghi trong docstring |

Chạy lệnh sau để tự kiểm tra trạng thái này dựa trên file thật đang có (không phải danh sách gõ tay):
```bash
python3 scripts/YOLO26/medrt_liteseg_cli.py
```

---

## 3. Cấu trúc thư mục (phần MedRT-SFOD tự thêm — nằm trên nền Ultralytics/RT-SFOD)

```
MedRT-SFOD/
├── README.md                              # file này
├── run_polyp_training.sh                  # convert + train polyp bằng 1 lệnh
├── scripts/YOLO26/
│   ├── stage0_stage1_adabn_rc_yolo26.py    # AdaBN warm-start (driving-scene, đã dùng được với backbone nhẹ)
│   ├── stage2_rtsfod_yolo26.py             # DHF + MARD + Mean-Teacher (driving-scene)
│   ├── stage2_card_rtsfod_yolo26.py        # = trên + CARD
│   ├── card_compression.py                 # module CARD (pruning + QAT)
│   ├── test_card_lite_smoke.py             # smoke test backbone nhẹ + CARD
│   ├── architecture_benchmark.py           # benchmark params/FPS trên kiến trúc thật
│   ├── architecture_cli.py                 # CLI cho benchmark trên
│   ├── domain_benchmark.py                 # list ảnh theo domain, kiểm tra path data
│   ├── medrt_liteseg_cli.py                # kiểm tra trạng thái roadmap dựa trên file thật
│   ├── polyp_kvasir_to_yolo.py             # convert Kvasir-SEG → chuẩn YOLO
│   └── train_source_supervised.py          # train nguồn (city/polyp, detect/segment)
└── ultralytics/
    ├── nn/modules/lite_block.py            # PConv / FasterNetBlock / C2fFaster
    ├── nn/tasks.py                         # (đã sửa) đăng ký C2fFaster vào YAML parser
    └── cfg/models/26/
        ├── yolo26-lite.yaml                # backbone nhẹ, detect
        └── yolo26-lite-seg.yaml            # backbone nhẹ, segment (chỉ kiến trúc)
```

---

## 4. Cài đặt

```bash
cd MedRT-SFOD
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # chọn bản cu12x khớp driver GPU
pip install opencv-python pyyaml numpy
```

⚠️ **Vá `ultralytics/data/` trước khi train thật** — thư mục này thiếu trong bản release công khai gốc (không phải do MedRT-SFOD), khiến `from ultralytics import YOLO` lỗi:
```bash
pip install ultralytics --target /tmp/ul_ref
cp -r /tmp/ul_ref/ultralytics/data ultralytics/data
```

Kiểm tra môi trường trước khi đụng tới dataset/GPU:
```bash
python3 scripts/YOLO26/test_card_lite_smoke.py
python3 scripts/YOLO26/architecture_cli.py --device cpu
```

---

## 5. Chuẩn bị dữ liệu

Đặt data **ngoài** thư mục `MedRT-SFOD/`, ngang hàng với nó:
```
<workdir>/
├── MedRT-SFOD/
└── datasets/
    ├── cityscapes/            # domain nguồn có nhãn (nếu làm driving-scene)
    ├── foggy_cityscapes/       # domain target self-training
    ├── polyp_detect/           # ⟵ output polyp_kvasir_to_yolo.py --task detect
    └── polyp_seg/              # ⟵ output polyp_kvasir_to_yolo.py --task segment
```

**Polyp (Kvasir-SEG)** — đã có converter, tự tải dataset gốc từ https://datasets.simula.no/kvasir-seg/, rồi:
```bash
python3 scripts/YOLO26/polyp_kvasir_to_yolo.py \
    --src /path/to/raw/Kvasir-SEG --dst ../datasets/polyp_detect --task detect

python3 scripts/YOLO26/polyp_kvasir_to_yolo.py \
    --src /path/to/raw/Kvasir-SEG --dst ../datasets/polyp_seg --task segment
```

**Cityscapes/Foggy Cityscapes** (nếu muốn benchmark trên driving-scene trước) — chưa có converter tự động, cần tự convert annotation gốc (polygon JSON) sang YOLO txt.

---

## 6. Cách train kiến trúc mới (MedRT-SFOD)

### Bước 1 — Train nguồn (source-supervised)
Đây là bước bắt buộc đầu tiên: train detector/segmenter bình thường (có nhãn) trên domain **nguồn**, tạo checkpoint để Bước 2 thích nghi sang domain **target** không nhãn.

```bash
# Polyp detection (sau khi convert ở mục 5)
python3 scripts/YOLO26/train_source_supervised.py \
    --model-cfg ultralytics/cfg/models/26/yolo26-lite.yaml \
    --data ../datasets/polyp_detect/dataset_detect.yaml \
    --task detect --epochs 100 --imgsz 640 --out-dir runs/polyp_detect_source

# Polyp segmentation
python3 scripts/YOLO26/train_source_supervised.py \
    --model-cfg ultralytics/cfg/models/26/yolo26-lite-seg.yaml \
    --data ../datasets/polyp_seg/dataset_seg.yaml \
    --task segment --epochs 100 --imgsz 640 --out-dir runs/polyp_seg_source
```
Hoặc gộp convert+train detection polyp bằng 1 lệnh: `./run_polyp_training.sh /path/to/raw/Kvasir-SEG detect`.

Checkpoint kết quả nằm ở `runs/<out-dir>/weights/best.pt`.

### Bước 2 — AdaBN warm-start (Stage 0+1)
**Hiện chỉ verify trên driving-scene (Cityscapes→Foggy Cityscapes)**, chưa thử với polyp — nhưng script không phụ thuộc domain cụ thể nên về lý thuyết dùng được, cần bạn tự thử và báo lại kết quả:
```bash
python3 scripts/YOLO26/stage0_stage1_adabn_rc_yolo26.py \
    --weights runs/polyp_detect_source/weights/best.pt \
    --data ../datasets/<target_domain>/dataset.yaml \
    --out_dir runs/lite_stage1 --epochs_adabn 2 --early_stop
```

### Bước 3 — Self-training source-free (Stage 2: DHF + MARD + CARD)
**Cũng hiện chỉ verify trên driving-scene** — dùng cho detection, chưa có bản cho segmentation (xem mục 2, "❌ Chưa code"):
```bash
python3 scripts/YOLO26/stage2_card_rtsfod_yolo26.py \
    --weights runs/lite_stage1/adapted_best.pt \
    --data ../datasets/<target_domain>/dataset.yaml \
    --out_dir runs/lite_stage2_card --epochs 60 \
    --card_enable --card_prune_target 0.3 --card_quant_bits 8 --card_warmup_epochs 10
```

Log mỗi step có thêm `card_sparsity`/`card_quant_ratio` để theo dõi schedule nén.

### Ablation nên chạy để có số liệu paper
| Cấu hình | Khác biệt |
|---|---|
| (a) backbone gốc (`yolo26.yaml`) + DHF/MARD | baseline so sánh |
| (b) backbone nhẹ (`yolo26-lite.yaml`) + DHF/MARD, không CARD | đo riêng tác động backbone |
| (c) backbone nhẹ + DHF/MARD + CARD | phương pháp đầy đủ |
| (d) như (c) nhưng `--card_gate_threshold 0` | kiểm chứng gate confidence thật sự cần thiết |

---

## 7. Benchmark kiến trúc
```bash
python3 scripts/YOLO26/architecture_cli.py --arch all --device cpu   # đổi --device cuda nếu có GPU
```
Đo trực tiếp trên kiến trúc thật (`yolo26.yaml`/`yolo26-lite.yaml`/`yolo26-lite-seg.yaml`), không phải model tự chế. Số liệu CPU hiện tại chỉ để demo pipeline chạy được — số liệu dùng cho paper cần đo trên GPU cố định (paper RT-SFOD gốc dùng 1× RTX A6000, FP32 PyTorch).

---

## 8. Roadmap tiếp theo

1. Chạy thật Bước 1-3 ở mục 6 trên polyp (Kvasir-SEG) với GPU thật, đo mAP/FPS thật.
2. Mở rộng DHF sang mask-IoU (hiện chỉ dùng box-IoU) để dùng được cho `yolo26-lite-seg.yaml`.
3. Viết self-training loop cho segmentation (Mean-Teacher + MARD trên mask, tương tự `stage2_card_rtsfod_yolo26.py` nhưng cho `Segment26` head).
4. Export pruning thật (resize tensor, không chỉ zero-mask) để đo tốc độ inference thật sau nén.
5. So sánh với baseline SFOD y tế đã có (SMPT/SMPT++, FSM) trên cùng benchmark polyp.

---

## 9. Trích dẫn

MedRT-SFOD kế thừa cơ chế DHF/MARD từ RT-SFOD:
```bibtex
@inproceedings{sairam2026rtsfod,
  title={Real-Time Source-Free Object Detection},
  author={Sairam VCR and Gopal, Varun and Jain, Poornima and Balasubramanian, Vineeth N and Khan, Muhammad Haris},
  booktitle={ECCV},
  year={2026}
}
```
Và nền tảng codebase từ Ultralytics (AGPL-3.0) — xem `LICENSE` trong repo.
