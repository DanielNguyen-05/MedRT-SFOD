# MedRT-SFOD / MedRT-SFSeg
## Kiến trúc tổng thể cho Source-Free Real-Time Medical Image Segmentation

> Tài liệu này mô tả **kiến trúc, luồng dữ liệu, các module và cơ chế huấn luyện/inference** của hệ thống MedRT-SFOD hiện tại.  

---

# 1. Mục tiêu hệ thống

MedRT-SFOD là framework nghiên cứu hướng tới bài toán:

> **Real-Time Source-Free Medical Image Segmentation dưới domain shift**

Trong nhánh hiện tại, bài toán cụ thể là **polyp segmentation** với:

- **Source domain:** Kvasir-SEG
- **Target domain:** CVC-ClinicDB
- **Target supervision trong adaptation:** không sử dụng target labels hoặc target GT masks
- **Backbone / base model:** YOLO26-S-Seg
- **Task:** instance segmentation, đồng thời đánh giá thêm binary union-mask segmentation

Mục tiêu cuối cùng gồm hai phần:

1. **Source-Free Domain Adaptation**
   - cải thiện chất lượng segmentation trên target domain mà không cần truy cập source data trong adaptation;
   - chỉ sử dụng source-trained checkpoint và unlabeled target images.

2. **Real-Time Model Compression**
   - giữ lại phần lớn accuracy sau adaptation;
   - cắt tỉa có cấu trúc;
   - tạo model compact vật lý;
   - giảm Params, MACs, latency;
   - tăng FPS.

---

# 2. Kiến trúc tổng thể

```text
┌──────────────────────────────────────────────────────────────┐
│                     SOURCE DOMAIN                            │
│                     Kvasir-SEG                               │
│                                                              │
│  Labeled images + masks                                      │
│            │                                                 │
│            ▼                                                 │
│  YOLO26-S-Seg supervised training                            │
│            │                                                 │
│            ▼                                                 │
│  Frozen Source Segmentation Model                            │
└────────────┬─────────────────────────────────────────────────┘
             │
             │ source data removed
             ▼
┌──────────────────────────────────────────────────────────────┐
│                     TARGET DOMAIN                            │
│                    CVC-ClinicDB                              │
│                                                              │
│               Target images only                             │
│                       │                                      │
│                       ▼                                      │
│                Stage 1: AdaBN                                │
│                       │                                      │
│                       ▼                                      │
│              AdaBN initialized model                         │
│                       │                                      │
│                       ▼                                      │
│         Stage 2: Dense MedRT-SFSeg                           │
│                                                              │
│   Mean Teacher                                               │
│      +                                                       │
│   O2O / O2M dual-head pseudo predictions                     │
│      +                                                       │
│   Mask-DHF                                                   │
│      +                                                       │
│   native segmentation loss                                   │
│      +                                                       │
│   MARD                                                       │
│                       │                                      │
│                       ▼                                      │
│             Dense adapted segmentation model                 │
│                       │                                      │
│                       ▼                                      │
│             Stage 3: RASP-SFSeg                              │
│                                                              │
│   Taylor importance                                          │
│      +                                                       │
│   per-group GMM                                              │
│      +                                                       │
│   cost-aware ranking                                         │
│      +                                                       │
│   Kneedle                                                    │
│      +                                                       │
│   reliability gate                                           │
│      +                                                       │
│   structured hidden-channel masks                            │
│                       │                                      │
│                       ▼                                      │
│              Physical compaction                             │
│                       │                                      │
│                       ▼                                      │
│               FINAL COMPACT MODEL                            │
└──────────────────────────────────────────────────────────────┘
```

---

# 3. Base Network: YOLO26-S-Seg

## 3.1. Model class

```text
SegmentationModel
```

Final head:

```text
Segment26
```

Main head configuration:

```text
end2end = True
nc      = 1
nm      = 32
npr     = 128
```

Trong bài toán hiện tại:

```text
class 0 = polyp
```

---

# 4. Dual-Head Segmentation Architecture

YOLO26-S-Seg sử dụng kiến trúc end-to-end dual-head:

```text
                    P3 / P4 / P5
                        │
                        ▼
                  ┌───────────┐
                  │ Segment26 │
                  └─────┬─────┘
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
        O2O branch            O2M branch
      One-to-One             One-to-Many
```

Mỗi branch gồm:

```text
box head
class head
mask coefficient head
```

Cả hai branch sử dụng shared prototype representation:

```text
Proto / Proto26
```

---

# 5. Segment26 Output Structure

Một prediction row sau post-processing có dạng:

```text
[x1, y1, x2, y2, confidence, class_id, mask_coefficients...]
```

Vì:

```text
nm = 32
```

nên mỗi prediction row có:

```text
6 + 32 = 38 values
```

Output điển hình:

```text
O2O:
[B, 300, 38]

O2M:
[B, 300, 38]

Proto:
[B, 32, Hp, Wp]
```

Ví dụ với input 480 × 640:

```text
Proto ≈ [B, 32, 120, 160]
```

---

# 6. Instance Mask Reconstruction

Instance mask được tạo từ:

```text
mask coefficients
        ×
shared Proto features
```

Công thức:

\[
L_i(x,y)
=
\sum_{c=1}^{C_m}
\alpha_{i,c} P_c(x,y)
\]

trong đó:

- \( \alpha_i \): mask coefficient vector của instance \(i\)
- \( P \): Proto tensor
- \(C_m = 32\)

Soft mask:

\[
M_i^{soft}
=
\sigma(L_i)
\]

Binary mask:

\[
M_i
=
\mathbf{1}
[
M_i^{soft} \ge 0.5
]
\]

---

# 7. Source Training Stage

Source-supervised training sử dụng:

```text
Kvasir-SEG
```

Split:

```text
800 train
200 validation
seed = 29
```

Source model được train bằng native YOLO26 segmentation criterion.

Sau source training, checkpoint được freeze.

Source data không còn được sử dụng trong Stage 1, Stage 2 hoặc Stage 3.

---

# 8. Source-Free Boundary

Từ thời điểm adaptation bắt đầu:

```text
ALLOWED
───────
Target images

NOT ALLOWED
───────────
Target labels
Target masks
Source images
Source labels
Source masks
```

Pipeline adaptation hiện tại theo protocol:

```text
transductive SFDA
```

Toàn bộ target images có thể được dùng ở adaptation dưới dạng unlabeled images.

Ground truth chỉ được dùng sau adaptation để evaluation.

---

# 9. Stage 1 — Adaptive Batch Normalization

Stage 1 sử dụng **AdaBN**.

Mục tiêu:

```text
Source BN statistics
        ↓
target image statistics
        ↓
target-adapted normalization
```

Model control:

```text
whole model: eval mode
all parameters: frozen
BN modules: train mode
```

Không có:

```text
loss
backward
optimizer
```

Chỉ running statistics của BatchNorm thay đổi.

Output Stage 1 trở thành initialization chung cho Dense MedRT-SFSeg và RASP-SFSeg.

---

# 10. Stage 2 — Dense MedRT-SFSeg

Dense Stage 2 gồm:

```text
AdaBN initialization
        ↓
Mean Teacher
        ↓
Teacher weak view
        ↓
O2O + O2M predictions
        ↓
Mask-DHF
        ↓
pseudo boxes + pseudo masks
        ↓
Student strong view
        ↓
native segmentation loss
        +
MARD
        ↓
Student update
        ↓
epoch-level EMA Teacher update
```

---

# 11. Mean Teacher Architecture

Hai network được khởi tạo từ cùng AdaBN checkpoint:

```text
Teacher
Student
```

## Teacher

```text
dense
eval mode
no gradient
receives weak target view
```

## Student

```text
trainable
receives strong target view
optimized on pseudo supervision
```

Teacher update:

\[
\theta_T
\leftarrow
m\theta_T
+
(1-m)\theta_S
\]

với:

```text
EMA momentum = 0.999
update frequency = once per epoch
```

---

# 12. Weak / Strong Target Views

Current implementation giữ cùng geometry giữa weak và strong view.

Strong augmentation tập trung vào:

```text
photometric augmentation
```

Điều này giúp:

```text
Teacher pseudo mask geometry
        ≈
Student target geometry
```

và tránh phải warp segmentation masks qua aggressive geometric transformations.

---

# 13. O2O Branch

O2O branch có vai trò:

```text
high precision
lower coverage
```

Các prediction vượt:

```text
tau_o2o = 0.5
```

được dùng làm reliable anchors.

Pseudo supervision từ O2O có:

```text
box
class
confidence
mask coefficient
reconstructed pseudo mask
```

---

# 14. O2M Branch

O2M branch có:

```text
higher coverage
more redundant predictions
more noise
```

Các prediction O2M vượt:

```text
tau_o2m = 0.5
```

trở thành candidates.

O2M candidates sau đó được kiểm tra bởi DHF / Mask-DHF.

---

# 15. Original Dual-Head Fusion Logic

DHF gốc xử lý:

```text
O2O anchors
+
novel O2M candidates
```

Box novelty condition:

\[
IoU_{box}
(
p_{o2m},
A_{o2o}
)
\le
\tau_{no}
\]

với:

```text
tau_no = 0.2
```

Sau đó classwise NMS trên O2M extras:

```text
tau_dup = 0.7
```

Original fusion:

```text
O2O anchors
+
accepted O2M extras
```

---

# 16. Mask-DHF

MedRT-SFSeg mở rộng DHF thành **Mask-aware Dual-Head Fusion**.

Vấn đề:

```text
box confidence cao
không đảm bảo
pseudo mask ổn định
```

Do đó O2M extra không chỉ được đánh giá ở box level mà còn ở mask level.

---

# 17. Mask Stability

Với soft mask probability \(P\):

\[
M_{low}
=
\mathbf{1}[P \ge 0.40]
\]

\[
M_{high}
=
\mathbf{1}[P \ge 0.60]
\]

Mask stability:

\[
q_{stab}
=
IoU
(
M_{low},
M_{high}
)
\]

Ý nghĩa:

```text
mask ít thay đổi khi threshold perturb
→ prediction ổn định hơn
```

---

# 18. Mask Reliability

Mask reliability score:

\[
r_{mask}
=
\sqrt{
s_{box}
\cdot
q_{stab}
}
\]

trong đó:

- \(s_{box}\): box confidence
- \(q_{stab}\): threshold mask stability

Current threshold:

```text
tau_mask = 0.744898
```

Threshold này được xác định theo label-free target prediction statistics.

Không sử dụng target GT để chọn threshold.

---

# 19. Production Mask-DHF Pipeline

```text
Teacher O2O
   │
   ├── confidence filter
   ├── valid mask
   └── anchors
         │
         │
Teacher O2M
   │
   ├── confidence filter
   ├── valid mask
   ├── box novelty vs O2O
   ├── classwise NMS
   ├── mask stability
   ├── reliability threshold
   └── accepted extras
         │
         ▼
O2O anchors + reliable O2M extras
         │
         ▼
final pseudo box/mask supervision
```

---

# 20. Pseudo Segmentation Batch

Student được supervise bởi pseudo targets gồm:

```text
batch_idx
class
bounding boxes
instance masks
semantic masks
```

Instance masks được reconstruct từ Teacher mask coefficients + Proto.

Semantic pseudo masks được tạo để tương thích native segmentation criterion của local YOLO26 implementation.

---

# 21. Native End-to-End Segmentation Loss

YOLO26 end-to-end segmentation criterion sử dụng hai assignment branches:

```text
one2many
one2one
```

Main loss components:

\[
\mathcal{L}_{seg}
=
\mathcal{L}_{box}
+
\mathcal{L}_{inst}
+
\mathcal{L}_{cls}
+
\mathcal{L}_{dfl}
+
\mathcal{L}_{sem}
\]

Trong implementation, loss vector có 5 thành phần:

```text
box
seg
cls
dfl
semseg
```

---

# 22. O2O/O2M Loss Weighting

End-to-end segmentation loss dùng hai native criteria:

```text
one2many:
TAL top-k = 10

one2one:
TAL top-k = 7
secondary top-k = 1
```

Combined loss có dynamic weighting.

Initial weighting:

```text
O2M = 0.8
O2O = 0.2
```

O2M weight giảm dần trong quá trình training.

---

# 23. MARD

MARD là **Multi-scale Adaptive Representation Diversification**.

MARD hoạt động trên feature maps đưa vào Segment26:

```text
P3
P4
P5
```

Feature hook lấy input của segmentation head trước prediction branches.

---

# 24. MARD Token Sampling

Pseudo boxes được dùng để xác định foreground region.

Mỗi pseudo box được assign vào feature level dựa trên scale.

Current sampling:

```text
foreground points = 8
background points = 128
top-k boxes = 15
```

Feature tokens từ:

```text
foreground
+
background
```

được dùng cho representation regularization.

---

# 25. MARD Loss

MARD gồm hai mục tiêu chính:

```text
variance preservation
covariance decorrelation
```

Tổng MARD loss:

\[
\mathcal{L}_{MARD}
=
\alpha
\mathcal{L}_{var}
+
\beta
\mathcal{L}_{cov}
\]

Current configuration:

```text
alpha = 1.0
beta  = 0.1
eta   = 12
```

---

# 26. MARD Weight Schedule

Student total loss:

\[
\mathcal{L}_{total}
=
\mathcal{L}_{SFSeg}
+
\lambda_{MARD}
\mathcal{L}_{MARD}
\]

Current configuration:

```text
lambda0       = 0.05
lambda_max    = 0.20
warm-up       = 5 epochs
confidence gate = 0.5
```

MARD weight phụ thuộc:

```text
training progress
+
pseudo-label reliability
```

---

# 27. Dense Stage 2 Optimization

Current optimization:

```text
epochs          = 60
optimizer       = SGD
learning rate   = 1e-4
momentum        = 0.937
weight decay    = 5e-4
Nesterov        = True
gradient clip   = 10
scheduler       = cosine
batch           = 4
imgsz           = 640
seed            = 29
```

---

# 28. Dense MedRT-SFSeg Output

Dense Stage 2 tạo model:

```text
AdaBN
+
Mean Teacher adaptation
+
Mask-DHF
+
MARD
```

Không thay đổi model width.

Do đó Dense model vẫn giữ architecture YOLO26-S-Seg đầy đủ trước compression.

---

# 29. Stage 3 — RASP-SFSeg

RASP-SFSeg là:

> **Reliability-Aware Structured Pruning during Source-Free Segmentation Adaptation**

RASP không áp dụng post-training pruning đơn thuần.

Pruning xảy ra **trong quá trình target adaptation**, để model có thời gian recover giữa các pruning events.

---

# 30. Dense Teacher / Prunable Student Design

Trong RASP Stage:

```text
Teacher
→ luôn dense

Student
→ dense latent parameters
→ forward có structured hidden-channel masks
```

Dense latent Student được giữ để:

```text
EMA Teacher compatibility
checkpoint consistency
recoverability
```

Physical width chỉ thực sự giảm sau Stage 3.

---

# 31. Safe Pruning Space

RASP-SFSeg v1 chỉ prune:

```text
dependency-safe Bottleneck hidden channels
```

Cụ thể:

```text
cv1.out
   ↓
hidden channels
   ↓
cv2.in
```

External Bottleneck interface không thay đổi trong masked training.

---

# 32. Modules Không Bị Prune Trực Tiếp

RASP-SFSeg v1 không cắt trực tiếp:

```text
Segment26
Proto / Proto26
O2O box head
O2O class head
O2O mask head
O2M box head
O2M class head
O2M mask head
```

Mục tiêu là bảo vệ segmentation-specific dependency structure.

---

# 33. Verified Prunable Groups

Structural audit hiện tại tìm thấy:

```text
13 safe Bottleneck hidden groups
```

Tổng hidden channels:

```text
880
```

DepGraph audit:

```text
13 / 13 local-safe
```

Maximum theoretical removable channels trong safety constraints:

```text
432
```

---

# 34. Target-Domain Taylor Importance

Channel importance được tính bằng first-order Taylor sensitivity.

Với activation \(a_c\) và gradient \(g_c\):

\[
I_c
=
|a_c \cdot g_c|
\]

Importance được lấy từ chính target adaptation loss:

```text
segmentation loss
+
MARD loss
```

Do đó pruning importance là:

```text
target-aware
task-aware
adaptation-aware
```

---

# 35. EMA Importance

Taylor importance không sử dụng single-batch score.

RASP duy trì EMA:

\[
\bar{I}_c^{(t)}
=
\beta
\bar{I}_c^{(t-1)}
+
(1-\beta)
I_c^{(t)}
\]

Current:

```text
beta = 0.90
```

---

# 36. Per-Group GMM

Mỗi Bottleneck hidden group được phân tích độc lập.

RASP fit Gaussian Mixture Model trên importance distribution.

Mục tiêu:

```text
low-importance mode
vs
high-importance mode
```

Current criteria:

```text
low posterior threshold = 0.80
minimum separation      = 1.0
minimum samples         = 16
```

Các channel thuộc low-importance component trở thành candidates.

---

# 37. Channel Packs

Pruning không loại từng channel ngẫu nhiên.

Candidates được gom thành hardware-friendly channel packs.

Current:

```text
pack size = 8 channels
```

Điều này giúp physical compact widths có cấu trúc tốt hơn.

---

# 38. Cost-Aware Ranking

Mỗi candidate pack được đánh giá theo:

```text
importance loss
relative to
compute saving
```

RASP ưu tiên candidate có:

```text
low information importance
+
high MAC saving
```

Do đó pruning không nhất thiết đồng đều giữa các layer.

---

# 39. Kneedle Adaptive Budget

RASP không dùng fixed sparsity target.

Các candidate packs được xếp theo cost-aware score.

Sau đó cumulative trade-off curve được tạo:

```text
information removed
vs
compute saved
```

Kneedle tìm operating point tự động.

Kết quả:

```text
pruning amount
=
adaptive
not fixed
```

---

# 40. Reliability Gate

Pruning chỉ được phép nếu target pseudo supervision đủ đáng tin cậy.

Current reliability threshold:

```text
0.50
```

Epoch pseudo confidence được dùng làm adaptation reliability signal.

Nếu reliability không đạt:

```text
pruning event bị chặn
```

---

# 41. Recovery Interval

Sau mỗi pruning event, RASP không prune liên tục.

Current recovery interval:

```text
3 epochs
```

Trong recovery period:

```text
Mask-DHF adaptation vẫn chạy
MARD vẫn chạy
Student vẫn optimize
Teacher vẫn EMA
```

Mục tiêu:

```text
recover pruning-induced accuracy loss
```

---

# 42. Monotonic Masks

RASP masks là monotonic:

```text
kept → có thể bị prune

pruned → không tự bật lại
```

Do đó structural compression tăng dần theo thời gian.

---

# 43. Current RASP Final State

RASP-SFSeg hiện tại đạt:

```text
pruning events = 7
```

Events:

```text
epoch 6
epoch 9
epoch 12
epoch 15
epoch 18
epoch 29
epoch 32
```

Final hidden sparsity:

```text
14.545%
```

Saved fraction trong controllable pruning space:

```text
prunable MACs   ≈ 14.06%
prunable params ≈ 8.11%
```

Sau epoch 32:

```text
no safe knee
```

và pruning tự dừng.

---

# 44. Physical Compaction

Training-time Student vẫn chứa dense tensors.

Do đó masked Student:

```text
không phải compact model thật
```

Physical compaction thực hiện slicing:

```text
Bottleneck cv1 output channels
+
corresponding BatchNorm channels
+
matching cv2 input channels
```

Các output dimensions còn lại của Bottleneck không đổi.

---

# 45. Physical Compaction Mapping

Với keep index \(K\):

```text
old cv1:
Cin → H

new cv1:
Cin → |K|
```

Weights:

\[
W_{cv1}^{new}
=
W_{cv1}^{old}[K]
\]

BN:

```text
weight[K]
bias[K]
running_mean[K]
running_var[K]
```

cv2:

```text
old:
H → Cout

new:
|K| → Cout
```

Weights:

\[
W_{cv2}^{new}
=
W_{cv2}^{old}[:,K]
\]

---

# 46. Numerical Equivalence Check

Physical compact model phải tương đương masked dense Student.

Current verification:

```text
device    = CPU
imgsz     = 256
tolerance = 2e-4
```

Observed:

```text
max absolute diff
= 3.433e-05
```

Postprocess max diff:

```text
3.052e-05
```

Do đó:

```text
physical compaction PASS
```

---

# 47. Current Compact Architecture Size

Physical exporter count:

```text
Dense latent params
= 11,434,463

Compact params
= 11,295,967
```

Reduction:

```text
1.211%
```

MACs @ 256:

```text
Dense
= 3.154084992 G

Compact
= 3.111527552 G
```

Reduction:

```text
1.349%
```

---

# 48. Inference Architecture

Training-time modules không cần ở inference:

```text
Teacher
Mask-DHF
MARD
RASP controller
Taylor hooks
GMM
Kneedle
reliability gate
```

Final deployed inference graph chỉ còn:

```text
compact YOLO26-S-Seg Student
```

---

# 49. O2O-Only Deployment

YOLO26 end-to-end design cho phép inference dùng O2O prediction branch.

Deployment objective:

```text
O2O-only
NMS-free
```

O2M branch chủ yếu phục vụ training pseudo-label coverage.

Final inference hướng tới loại bỏ training-only redundancy.

---

# 50. Evaluation Representation

Hệ thống được đánh giá theo hai góc nhìn.

## Native instance segmentation

```text
Mask mAP50
Mask mAP50-95
```

## Binary medical segmentation

Tất cả predicted instance masks được union:

```text
instance mask 1
∪
instance mask 2
∪
...
```

thành binary polyp mask.

Sau đó báo:

```text
Dice
IoU
Precision
Sensitivity / Recall
Specificity
```

---

# 51. Current Main Results

## Source-Only on CVC-ClinicDB

```text
Mask mAP50        0.781710
Mask mAP50-95     0.524527
Dice              0.727507
IoU               0.655414
Precision         0.727876
Sensitivity       0.789525
Specificity       0.974837
```

---

# 52. AdaBN Results

```text
Mask mAP50        0.790490
Mask mAP50-95     0.543404
Dice              0.743922
IoU               0.672770
Precision         0.740146
Sensitivity       0.808458
Specificity       0.974264
```

---

# 53. Dense MedRT-SFSeg Results

```text
Mask mAP50        0.817132
Mask mAP50-95     0.558640
Dice              0.803613
IoU               0.726137
Precision         0.801234
Sensitivity       0.874483
Specificity       0.969119
```

---

# 54. Final RASP Compact Results

```text
Mask mAP50        0.818410
Mask mAP50-95     0.563996
Dice              0.792258
IoU               0.715804
Precision         0.791209
Sensitivity       0.865828
Specificity       0.967874
```

Evaluator inference parameter count:

```text
10.227 M
```

Physical exporter parameter count:

```text
11.296 M
```

Hai con số này đến từ hai counting representations khác nhau và không nên trộn trong cùng một compression comparison.

---

# 55. Accuracy Transition

```text
Source Only
Dice = 0.7275
     │
     │ AdaBN
     ▼
AdaBN
Dice = 0.7439
     │
     │ Mask-DHF + MARD + Mean Teacher
     ▼
Dense MedRT-SFSeg
Dice = 0.8036
     │
     │ RASP-SFSeg
     ▼
Compact MedRT-SFSeg
Dice = 0.7923
```

Compact model vẫn giữ phần lớn adaptation gain.

---

# 56. Main Functional Roles

| Module | Vai trò |
|---|---|
| YOLO26-S-Seg | base real-time segmentation network |
| AdaBN | target-domain normalization adaptation |
| Mean Teacher | stable self-training teacher |
| O2O | high-precision pseudo anchors |
| O2M | high-coverage pseudo candidates |
| DHF | box-level dual-head fusion |
| Mask-DHF | mask-aware reliability filtering |
| Proto26 | shared mask prototype generation |
| Native Seg Loss | train Student bằng pseudo box + mask |
| MARD | preserve / diversify multi-scale target features |
| Taylor Importance | target-aware pruning importance |
| GMM | detect low-importance channel population |
| Cost-aware Ranking | prioritize compute-efficient pruning |
| Kneedle | adaptive pruning budget |
| Reliability Gate | prevent pruning under unreliable pseudo supervision |
| Recovery Interval | recover after structural removal |
| Physical Compaction | convert masks into real smaller tensors |
| O2O-only inference | final NMS-free deployment path |

---

# 57. Training-Only vs Deployment Components

## Training / adaptation only

```text
Teacher
O2M pseudo-label branch usage
Mask-DHF
MARD
Taylor hooks
importance EMA
GMM
Kneedle
RASP masks/controller
```

## Deployment

```text
Physically compact Student
+
Segment26
+
Proto
+
O2O inference branch
```

---

# 58. Scientific Attribution of Current Architecture

MedRT-SFSeg hiện tại có ba lớp ý tưởng.

## Adopted foundation

```text
Mean Teacher
AdaBN
Dual-Head Fusion concept
MARD
```

DHF và MARD là kỹ thuật có nguồn gốc từ RT-SFOD.

## Segmentation-specific extension

```text
YOLO26 dual-head instance segmentation
+
pseudo mask reconstruction
+
Mask-DHF
+
threshold-stability mask reliability
+
native instance + semantic pseudo supervision
```

## Compression extension

```text
RASP-SFSeg
```

gồm:

```text
target Taylor importance
per-group GMM
cost-aware packs
adaptive Kneedle frontier
pseudo-reliability gating
recovery cycles
physical structured compaction
```

---

# 59. Architecture Identity

Kiến trúc cuối cùng có thể tóm tắt thành:

> **A dual-head Mean-Teacher source-free segmentation framework in which O2O predictions provide reliable anchors, O2M predictions expand target coverage, Mask-DHF removes mask-unstable extras, MARD regularizes multi-scale target representations, and RASP adaptively removes low-importance Bottleneck hidden channels during target adaptation before exporting a physically compact NMS-free segmentation Student.**

---

# 60. End-to-End Architecture Summary

```text
Labeled Source
   │
   ▼
YOLO26-S-Seg supervised source training
   │
   ▼
Frozen source model
   │
   └──────────── source data discarded
                       │
                       ▼
                 Target images only
                       │
                       ▼
                    AdaBN
                       │
                       ▼
            ┌─────────────────────┐
            │    Mean Teacher     │
            └──────────┬──────────┘
                       │
              Teacher weak view
                       │
              ┌────────┴────────┐
              ▼                 ▼
             O2O               O2M
              │                 │
              │          confidence filter
              │                 │
              │          box novelty filter
              │                 │
              │             NMS extras
              │                 │
              │          mask stability
              │                 │
              │        reliability filter
              │                 │
              └────────┬────────┘
                       ▼
                    Mask-DHF
                       │
             pseudo boxes + masks
                       │
                       ▼
              Student strong view
                       │
              native SFSeg loss
                       │
                      +│
                       ▼
                     MARD
                       │
                       ▼
               Student optimization
                       │
                       ▼
              epoch EMA → Teacher
                       │
                       ▼
               Dense MedRT-SFSeg
                       │
                       ▼
                  RASP-SFSeg
                       │
          Taylor |activation × grad|
                       │
                       ▼
                importance EMA
                       │
                       ▼
                per-group GMM
                       │
                       ▼
              low-importance packs
                       │
                       ▼
              cost-aware ranking
                       │
                       ▼
                   Kneedle
                       │
                       ▼
             pseudo reliability gate
                       │
                       ▼
             monotonic channel masks
                       │
                       ▼
               recovery adaptation
                       │
                       ▼
              final masked Student
                       │
                       ▼
              physical compaction
                       │
                       ▼
       Compact YOLO26-S-Seg Student
                       │
                       ▼
             O2O-only deployment
```

---

# 61. Current Final Model Definition

Current final experimental model is:

```text
YOLO26-S-Seg
+
AdaBN initialization
+
Mean Teacher SFDA
+
Mask-DHF
+
MARD
+
RASP-SFSeg
+
physical Bottleneck hidden-channel compaction
```

Final target evaluation:

```text
Dice          = 0.792258
IoU           = 0.715804
Mask mAP50    = 0.818410
Mask mAP50-95 = 0.563996
```

Physical compact report:

```text
Params reduction = 1.211%
MAC reduction    = 1.349% @ 256
```

Physical equivalence:

```text
PASS
```

---

# 62. Current Scope of MedRT-SFOD

Current version intentionally does **not** include:

```text
direct Proto pruning
direct Segment26 head pruning
direct mask-head pruning
quantization-aware training
INT8 compression
mask-guided MARD
geometric strong-view mask warping
target-label model selection
target-label pruning threshold selection
```

Các phần này có thể trở thành future extensions hoặc ablations riêng.

---

# 63. One-Line Pipeline

```text
Source YOLO26-S-Seg
→ AdaBN
→ Mean Teacher
→ O2O/O2M
→ Mask-DHF
→ pseudo segmentation
→ MARD
→ Dense MedRT-SFSeg
→ RASP-SFSeg
→ physical compaction
→ O2O-only real-time deployment
```
