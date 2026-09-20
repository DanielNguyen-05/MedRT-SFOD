# BraTS2023: train / val / test

Đây là baseline **YOLO26-S-Seg supervised**, dùng mã Ultralytics trong repo.
Chưa phải thí nghiệm đầy đủ AdaBN → Mean Teacher → Mask-DHF → MARD → RASP.
Chưa thể kết luận mô hình hoạt động tốt trên BraTS trước khi chạy huấn luyện đầy đủ.

## Dữ liệu đã kiểm kê

| Nhóm | Train (ca/bệnh nhân) | Val (ca/bệnh nhân) | Test (ca/bệnh nhân) |
|---|---:|---:|---:|
| GLI | 877 / 793 | 186 / 170 | 188 / 170 |
| MEN | 703 / 660 | 146 / 142 | 151 / 142 |
| PED | 69 / 69 | 15 / 15 | 15 / 15 |
| Tổng | 1.649 / 1.522 | 347 / 327 | 354 / 327 |

Seed 29, tỷ lệ 70/15/15 **theo bệnh nhân, riêng từng nhóm**. Các lần chụp
`BraTS-XXX-NNNNN-xxx` có cùng tiền tố bệnh nhân được đặt trong cùng split.
Giả định định danh này được lưu trong manifest; chưa có metadata để phát hiện
một bệnh nhân được cấp nhiều tiền tố khác nhau.

47 ca `BraTS-MEN-TRAIN-FIX-V4` thay thế ca gốc tương ứng trong `BraTS-MEN-Train`.
Việc ưu tiên bản sửa phù hợp với [changelog của ban tổ chức](https://www.synapse.org/Synapse%3Asyn51156910/discussion/threadId%3D10314).
219 GLI + 141 MEN + 45 PED thuộc ValidationData không có `seg.nii.gz` trong bản
local, được kiểm kê riêng và không đưa vào train/val/test có nhãn.
Test ở đây là **holdout nội bộ**, không phải test chính thức của challenge.

## Cách biểu diễn và đánh giá

- Ảnh axial 2D, RGB = T1c / T2-FLAIR / T2w; T1n chưa sử dụng.
- Chuẩn hóa từng modality, từng volume bằng percentile 0,5–99,5 trên voxel khác 0;
  không dùng mask hay thống kê từ tập khác. Lưu PNG 8-bit.
- Mask nhị phân `seg > 0`: hợp tất cả vùng được chú giải, tên lớp `whole_lesion`.
  Đây không phải phân đoạn riêng ET/TC/WT hay benchmark đa lớp chính thức.
- Giữ mọi lát cắt, kể cả lát không có tổn thương, trong lần chạy đầy đủ.
- YOLO học từ polygon contour ngoài; lỗ và contour suy biến không thể được biểu
  diễn chính xác. Số contour suy biến được ghi lại. Đánh giá Dice/IoU dùng
  **mask gốc**, không dùng polygon đã xấp xỉ.
- Tắt HSV/BGR augmentation vì ba kênh là ba MRI modality.
- `best.pt` được chọn bằng fitness box + mask mAP50–95 trên val của Ultralytics.
- Đánh giá val và test với checkpoint đã chọn. Confidence cho Dice/IoU mặc định
  0,25; không chọn threshold dựa trên test. mAP dùng confidence 0,001.
- Cộng TP/FP/FN qua toàn bộ lát của từng ca rồi tính Dice/IoU thể tích; trung bình
  các lần chụp trong từng bệnh nhân trước khi lấy trung bình bệnh nhân. Báo cáo
  riêng GLI/MEN/PED và bootstrap CI 95% theo bệnh nhân (1.000 lần lấy mẫu).
- Cả GT/pred rỗng: Dice/IoU = 1; chỉ một mask rỗng: Dice/IoU = 0. Có thêm số lát
  âm tính bị dự đoán dương tính. Không báo HD95 hoặc điểm lesion-wise chính thức.

Không dùng kết quả gộp để che khuất chất lượng trên PED ít mẫu. Muốn đánh giá
MedRT-SFSeg, cần một thí nghiệm chuyển miền riêng với source/target xác định,
target train không dùng nhãn, và target val/test không tham gia adaptation.

## Cài đặt

```bash
source .venv/bin/activate
python -m pip install -e . -r requirements-brats.txt
```

## Chạy đầy đủ trên GPU

Notebook: [07_BraTS2023_Train_Val_Test.ipynb](../colab/07_BraTS2023_Train_Val_Test.ipynb).
Hoặc từ thư mục gốc repo:

```bash
python scripts/YOLO26/medseg/prepare_brats2023.py \
  --src dataset/BraTS2023 --dst dataset/BraTS2023-YOLO26

python scripts/YOLO26/medseg/run_brats2023.py \
  --data dataset/BraTS2023-YOLO26/dataset_seg.yaml \
  --weights yolo26s-seg.pt --epochs 100 --imgsz 256 \
  --batch 8 --workers 4 --device 0 --name yolo26s_wt
```

Checkpoint pretrained sẽ được tải nếu chưa có. Có thể truyền đường dẫn checkpoint
tương thích của bạn. Dùng `--domains GLI` (hoặc MEN/PED) với thư mục đích mới để
chạy một nhóm; split của nhóm đó giữ nguyên. Baseline mặc định huấn luyện gộp 3 nhóm.
100 epochs là ngân sách ban đầu; early stopping patience 20. Chưa đo thời gian GPU.

Máy local Apple M3 có 24 GiB RAM và hỗ trợ MPS khi tiến trình được phép truy cập
GPU ngoài sandbox. Có thể thay `--device 0` bằng `--device mps --workers 0`, đặt
`PYTORCH_ENABLE_MPS_FALLBACK=1` trước lệnh Python. Script tắt AMP trên MPS.
Kiểm tra `torch.backends.mps.is_available()` trong đúng môi trường sẽ chạy.
Huấn luyện đầy đủ trên 2.350 ca nhiều hơn rất nhiều so với smoke 90 lát; chưa
có ước lượng thời gian đáng tin cậy cho toàn bộ cấu hình 256/batch 8/100 epochs.

Script không ghi đè thư mục kết quả đã tồn tại. Khi training bị ngắt, giữ nguyên
đường dẫn dataset và dùng cùng tham số, thêm `--resume`; cần `weights/last.pt`
và `protocol.json` của lần chạy đó. Resume phục hồi cấu hình huấn luyện từ
checkpoint. Colab mất runtime có thể yêu cầu chuẩn bị lại dataset tại cùng đường dẫn.

Đánh giá checkpoint đã huấn luyện (dùng tên kết quả mới):

```bash
python scripts/YOLO26/medseg/run_brats2023.py \
  --data dataset/BraTS2023-YOLO26/dataset_seg.yaml \
  --weights runs/seg/brats2023/yolo26s_wt/weights/best.pt \
  --eval-only --device 0 --name yolo26s_wt_evaluation
```

Đầu ra chính:

- Dataset: `manifest.json`, `dataset_seg.yaml`, `images/`, `labels/`, `gt_masks/`.
- Training: `runs/seg/brats2023/<name>/weights/{best,last}.pt`, `results.csv`.
- Đánh giá: `runs/seg/brats2023/<name>/evaluation.json`, gồm mAP, Dice, IoU,
  precision, recall, kết quả từng ca/từng nhóm và CI theo bệnh nhân.

## Chạy thử nhỏ trên CPU

```bash
python scripts/YOLO26/medseg/prepare_brats2023.py \
  --dst dataset/BraTS2023-YOLO26-smoke \
  --max-cases-per-domain-split 1 --slice-stride 16

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/YOLO26/medseg/run_brats2023.py \
  --data dataset/BraTS2023-YOLO26-smoke/dataset_seg.yaml \
  --weights ultralytics/cfg/models/26/yolo26s-seg.yaml \
  --epochs 1 --imgsz 128 --batch 2 --workers 0 --device cpu \
  --name cpu_smoke
```

Chỉ 9 ca, lấy mẫu lát cắt không dựa vào mask; huấn luyện từ đầu 1 epoch.
Kết quả được gắn `smoke_subset: true`, **không dùng đánh giá chất lượng mô hình**.
Việc chạy thử đã đọc test của các ca nhỏ này; nếu cần một test hoàn toàn chưa
quan sát cho nghiên cứu chính thức, thiết lập protocol mới trước khi phát triển
mô hình và không dùng lại các ca đã kiểm tra để ra quyết định.

Lần CPU đã hoàn tất tại `runs/seg/brats2023/cpu_smoke_verified/evaluation.json`:
30 lát train / 30 val / 30 test, 1 epoch, ảnh 128, batch 2, khởi tạo ngẫu nhiên.
Dice test = 0, mask mAP50 và mAP50–95 = 0; chưa dự đoán được tổn thương trên test.
Dice val trung bình 0,3333 chỉ do một ca không còn voxel tổn thương trong các lát
được lấy mẫu và dự đoán cũng rỗng (quy ước Dice = 1); không phải phát hiện đúng u.
Kết quả này chứng minh luồng chạy hoàn tất, chưa chứng minh khả năng học hoặc
chất lượng của YOLO26/MedRT-SFSeg trên toàn bộ BraTS2023.

Đã xác nhận lại cả train/val/test trên GPU MPS tại
`runs/seg/brats2023/mps_smoke_verified_v2/evaluation.json`, cùng cấu hình smoke;
Dice test và mask mAP đều bằng 0. Môi trường local đã sửa torchvision từ 0.1.6
sang 0.28.0 tương thích torch 2.13.0. MPS có cảnh báo một toán tử chưa bảo đảm
deterministic hoàn toàn dù đã đặt seed; không mặc định kết quả lặp lại bit-for-bit.
Huấn luyện đầy đủ 100 epochs chưa được thực hiện. Notebook Colab đã kiểm tra cú
pháp các cell Python, chưa chạy trên dịch vụ Colab.

Kiểm tra dữ liệu không chuyển đổi ảnh và chạy kiểm thử:

```bash
python scripts/YOLO26/medseg/prepare_brats2023.py \
  --audit-only --dst runs/seg/brats2023/new_dataset_audit
python -m unittest discover -s tests -p 'test_brats2023.py' -v
```
