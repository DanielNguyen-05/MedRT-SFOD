# MedRT-SFSeg trên BraTS2024-GLI

Pipeline này chạy **model MedRT-SFSeg của repository**, với backbone/head YOLO26-S-Seg,
AdaBN, Mean Teacher, Mask-DHF, DURR và SegMARD-v2. Tell2Adapt được dùng để xác định bốn
hướng chuyển modality; pipeline không sử dụng BiomedParse, CAPR hoặc VPR của Tell2Adapt.

## Dữ liệu giữ lại

Trong `dataset/BraTS2024`:

| Đường dẫn | Nội dung | Vai trò |
|---|---|---|
| `training_data1_v2` | 1.350 ca GLI có bốn MRI và mask | Pool có nhãn để tạo các partition |
| `training_data_additional` | 271 ca GLI có bốn MRI và mask | Cùng pool; metadata ghi `Train-additional` |
| `validation_data` | 188 ca GLI, bốn MRI, không có mask | Chỉ suy luận ngoài thí nghiệm nội bộ |
| `BraTS-PTG supplementary demographic information and metadata.xlsx` | Metadata và nguồn dữ liệu | Đối chiếu cohort, split phát hành và provenance |

Các gói MEN-RT, pathology, fastlane và mock MLCubes đã được bỏ. Ba ca GLI-fastlane là
bản sao byte-for-byte của train. Đếm theo **ca chụp**, không phải tất cả là bệnh nhân độc lập:
1.621 ca có nhãn tương ứng 731 tiền tố ID bệnh nhân, nhiều bệnh nhân có nhiều lần chụp.

Raw labels: `0=background`, `1=NETC`, `2=SNFH`, `3=ET`, `4=RC`.
YOLO class IDs: `0=NETC`, `1=SNFH`, `2=ET`, `3=RC`. Không gộp `seg > 0`, không đánh đồng
NETC với vùng TC tổng hợp (ET + NETC). Tell2Adapt gọi cột này TC nhưng prompt ghi
“non-enhancing tumor core”; xem [paper](https://arxiv.org/abs/2603.05012) và
[mô tả BraTS post-treatment](https://arxiv.org/html/2405.18368v1).

## Protocol thí nghiệm

Mỗi hướng khởi tạo độc lập từ cùng loại pretrained weights:

- `t1n_to_t2w`
- `t2w_to_t1n`
- `t1c_to_t2f`
- `t2f_to_t1c`

**Bốn nhóm bệnh nhân tách biệt**, seed 29. Các hướng dùng chung danh sách ID:

| Role | Tỷ lệ bệnh nhân | Bệnh nhân | Ca chụp | Dùng nhãn thật? |
|---|---:|---:|---:|---|
| `source_train` | 45% | 328 | 719 | Có, huấn luyện source |
| `source_val` | 10% | 73 | 174 | Có, chọn source checkpoint |
| `target_train` | 30% | 219 | 483 | **Không**, chỉ ảnh cho AdaBN/calibration/adaptation |
| `target_test` | Phần còn lại, khoảng 15% | 111 | 245 | Chỉ sau khi chốt Student cuối |

Các tỷ lệ tính trên bệnh nhân; số ca chụp không nhất thiết đúng các tỷ lệ đó.
Mọi lần chụp và modality của cùng bệnh nhân ở cùng role. Không dùng source modality
của bệnh nhân target-train/test để pretrain. 188 ca external validation cũng không
trùng tiền tố bệnh nhân với pool nội bộ và không tham gia adaptation mặc định.

```text
Source modality / source_train + GT
    → YOLO26-S-Seg supervised training
    → best.pt theo source_val
Target modality / target_train, chỉ ảnh
    → AdaBN (2 epoch)
    → audit reliability Q25 theo lớp, không nhãn
    → Mean Teacher + Mask-DHF + DURR theo lớp + SegMARD-v2 (60 epoch)
    → Student ở epoch cuối, không chọn bằng target GT
Target modality / target_test
    → cùng tập test cho source-only, AdaBN, final Student
```

Source mặc định 100 epoch, AdamW lr 0.001, patience 20. `best.pt` được chọn bằng fitness
box + mask mAP50-95 của **source val**, không phải Dice test. Adaptation dùng SGD lr 1e-4,
momentum 0.937, weight decay 5e-4, cosine decay, gradient clip 10 và parameter EMA 0.999
mỗi epoch. Teacher BN giữ statistics AdaBN theo logic hiện có của model. Student cuối
là model triển khai; EMA Teacher cũng được lưu để phân tích nhưng không dùng target score
để chọn giữa Student và Teacher. Không thêm DURR/SegMARD vào đồ thị suy luận.

Đây là **protocol riêng được công bố rõ**, không phải tái lập nguyên trạng bảng Tell2Adapt:
paper ghi train/test 80/20 theo lát cắt, không mô tả val riêng hoặc danh sách split BraTS.
Ở đây chia theo bệnh nhân và còn tách source/target training patients. Backbone, normalization,
ngân sách train, tập ca và quy ước metric cũng khác; không so trực tiếp số điểm như cùng benchmark.

## Model được mở rộng như thế nào

`durr_multiclass.py` tái sử dụng thuật toán DURR hiện có, decode Teacher một lần rồi xử lý
riêng từng lớp. Matching O2O/O2M, signed boundary routes, rescue và safe-background đều
theo lớp. Không dùng witness lớp khác để sửa một vùng. Ba loss phụ áp dụng lên semantic
channel tương ứng, lấy trung bình bốn lớp; trường hợp một lớp tương đương binary DURR cũ.

Native pseudo batch giữ class ID và tạo semantic map từ pseudo masks. Khi pseudo instances
khác lớp chồng nhau, instance confidence cao hơn quyết định nhãn semantic. SegMARD-v2 giữ
cách lấy FG/HBG/EBG từ hard pseudo geometry ở P3/P4/P5, `hard_bg_ratio=0.5`, không dùng
reliability weighting v3 hoặc RASP. Vùng HBG/EBG là vùng tương đối với pseudo instances
theo implementation gốc; đây không phải ground truth mô học của toàn ảnh.

Reliability threshold tính lại cho từng target và từng lớp bằng Q25 của valid novel O2M
candidates sau NMS; mặc định lấy tối đa 4.096 ảnh target train theo chỉ số đều, xác định trước.
Đặt `calibration_images: 0` để audit toàn bộ target train. Nếu lớp không có candidate,
threshold > 1 vô hiệu hóa extras lớp đó; O2O anchors vẫn được giữ. Không dùng hằng số CVC.
Nếu một epoch không có tín hiệu tối ưu nào, runner dừng và báo lỗi, không báo adaptation thành công.

## Chuẩn bị ảnh

- Mỗi modality chuẩn hóa độc lập theo percentile 0.5–99.5 của voxel khác 0 trong volume;
  background bằng 0, ghi PNG uint8 một kênh và lặp thành ba kênh khi model đọc.
- Không ghép modality; augmentation target chỉ intensity/gamma/noise/blur, giữ ba kênh giống nhau.
- Geometry weak/strong giống nhau, có flip ngang chung. Không dùng target mask để chọn lát cắt.
- Full run dùng mọi axial slice, kể cả lát không có tổn thương.
- Mỗi connected contour của từng raw label thành polygon YOLO riêng. Đây là xấp xỉ supervision:
  external contour lấp các lỗ bên trong; contour suy biến bị bỏ và đếm trong manifest.
- Evaluation giữ mask voxel gốc, không chấm bằng polygon đã xấp xỉ.
- Converter **không mở hoặc xuất mask `target_train`**. Source YAML chỉ trỏ source_train/source_val.
- Source-val/test GT được đặt trong `evaluation_masks`, tách khỏi thư mục ảnh target adaptation.
- Manifest kiểm tra đầy đủ tên ảnh/label, hình học lúc convert và patient overlap. Conversion bị
  ngắt có thể chạy lại `--stage prepare` để tiếp tục các ca chưa hoàn tất.

Việc chuẩn bị bốn modality đầy đủ tạo khoảng 1,32 triệu PNG trên dataset hiện tại; nên dùng
ổ SSD local cho prepared data. Tránh convert trực tiếp từng ảnh nhỏ lên Google Drive.

## Cài đặt và chạy

Tại repository root, sử dụng environment có PyTorch/torchvision tương thích:

```bash
python -m pip install -e . -r requirements-brats.txt
python scripts/YOLO26/medseg/brats2024_experiments.py --stage plan
```

Chạy trọn một hướng trên GPU CUDA:

```bash
python scripts/YOLO26/medseg/brats2024_experiments.py --stage all --directions t1n_to_t2w --device 0
```

Chạy lần lượt cả bốn hướng:

```bash
python scripts/YOLO26/medseg/brats2024_experiments.py --stage all --device 0
```

Tách giai đoạn khi cần:

```bash
python scripts/YOLO26/medseg/brats2024_experiments.py --stage prepare
python scripts/YOLO26/medseg/brats2024_experiments.py --stage source --directions t1n_to_t2w --device 0
python scripts/YOLO26/medseg/brats2024_experiments.py --stage adapt --directions t1n_to_t2w --device 0
python scripts/YOLO26/medseg/brats2024_experiments.py --stage evaluate --directions t1n_to_t2w --device 0
```

Tiếp tục run bị ngắt với đúng config ban đầu:

```bash
python scripts/YOLO26/medseg/brats2024_experiments.py --stage all --directions t1n_to_t2w --device 0 --resume
```

Source resume dùng native `last.pt`. Adaptation lưu Student/Teacher state, optimizer, scheduler,
epoch, RNG và Q25 calibration mỗi epoch vào `resume.pt`; phục hồi ở ranh giới epoch đã lưu.
AdaBN bị ngắt chạy lại từ source checkpoint. Evaluation bị ngắt chạy lại trên các checkpoint
đã chốt. Các stage hoàn tất được bỏ qua sau khi kiểm tra hash artifact; cấu hình/split/code thay đổi
bị từ chối để tránh trộn kết quả. Giữ raw/prepared/project paths ổn định khi resume source training.

Thay đổi thiết lập nghiên cứu qua [config](../configs/experiments/brats2024_sfseg.yaml).
Có CLI overrides `--source-epochs`, `--adapt-epochs`, `--imgsz`, `--batch`, `--workers`,
`--seed`, `--weights`, `--raw-root`, `--prepared-root`, `--project`. Khi đổi split, dùng cả
prepared root và project mới; đổi hyperparameters dùng project mới. CUDA không có sẽ báo lỗi,
không tự chuyển sang CPU. Máy Mac có thể dùng `--device mps --workers 0` nếu PyTorch hỗ trợ.

Smoke test CPU (mỗi role một ca, stride 32, YOLO26-N ngẫu nhiên, một epoch mỗi giai đoạn):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/YOLO26/medseg/brats2024_experiments.py --stage all --smoke
python -m unittest discover -s tests -p 'test_brats2024.py' -v
```

Smoke dùng threshold hallucination thấp riêng để exercise backward khi source chưa học được
pseudo instances. Đây là kiểm tra kỹ thuật; scores, calibration và trọng số smoke không dùng
cho kết luận khoa học. Unit test riêng chạy native loss + class-wise DURR + SegMARD backward
trên **YOLO26-S-Seg bốn lớp** và kiểm tra gradient từng semantic channel.

Notebook: [07_BraTS2024_SFSeg.ipynb](../colab/07_BraTS2024_SFSeg.ipynb).

## Kết quả và metric

```text
runs/seg/brats2024_sfseg/full/
├── plan.json
├── summary.json / summary.csv
└── t1n_to_t2w/                         # tương tự cho ba hướng khác
    ├── source/weights/best.pt
    ├── source/selection.json
    ├── adabn/adabn.pt
    ├── adapt/calibration.json
    ├── adapt/history.json
    ├── adapt/resume.pt
    ├── adapt/student_final.pt
    ├── adapt/teacher_final.pt
    └── evaluation/
        ├── source_val.json
        ├── source_only_target_test.json
        ├── adabn_target_test.json
        └── medrt_sfseg_target_test.json
```

Inference chỉ dùng native O2O predictions. Retina masks trả về ảnh trước resize/letterbox;
instances chồng khác lớp được resolve bằng confidence cố định 0.25 rồi ghép volume 3D.
Các class NETC/SNFH/ET/RC được chấm riêng: Dice, IoU, precision/recall, ASD, HD95.
ASD là trung bình gộp khoảng cách mặt biên hai chiều; HD95 là percentile 95 của cùng tập
khoảng cách. Dùng surface 6-connectivity và physical spacing từ NIfTI.

Quy ước empty: cả hai rỗng → Dice=1, ASD=HD95=0; chỉ một rỗng → Dice=0, ASD/HD95=null.
Luôn báo số ca không xác định surface, số GT-present, Dice riêng GT-present để không che lỗi
bỏ sót bằng trung bình boundary trên các ca còn lại. Báo mean/std theo ca và mean sau khi
trung bình các lần chụp trong từng bệnh nhân. Không phải official lesion-wise BraTS score.
Không chọn threshold, epoch hoặc biến thể bằng target test.

Suy luận 188 ca không nhãn sau khi có full Student, lưu NIfTI về đúng affine:

```bash
python scripts/YOLO26/medseg/brats2024_experiments.py --stage predict --directions t1n_to_t2w --device 0
```

Kết quả ở `<direction>/external_predictions`. Không có Dice/ASD cho tập này do thiếu GT.
Script đánh giá source-only/AdaBN/Student cùng test là đánh giá adaptation của MedRT-SFSeg;
chưa bao gồm target-supervised oracle hoặc tái lập code Tell2Adapt.

## Trạng thái kiểm tra

Đã audit 1.809 ca GLI chính: header ảnh/mask khớp; đọc hết 1.621 masks, kiểm tra 12 MRI payloads.
Xem `runs/audits/brats2024_inventory.json` và `brats2024_cleanup.json` cho kiểm kê/dọn dữ liệu.
Huấn luyện đầy đủ không tự động bắt đầu khi chạy `plan`; cần chủ động chạy `source/adapt/all`.
