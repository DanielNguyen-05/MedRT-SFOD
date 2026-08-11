Architecture chính là:

```text
YOLO26-M
   +
Mean Teacher
   +
DHF
   +
MARD
   +
RASP structured pruning
```

Trong đó RASP chỉ thay đổi Student trong quá trình adaptation và cuối cùng compact một số **hidden channels bên trong Bottleneck**; backbone/neck/head bên ngoài vẫn giữ topology YOLO26-M. Điều này đúng với thiết kế trong code RASP: Teacher dense, Student giữ dense latent parameters, RASP gate hidden channels, sau đó physical export mới thực sự cắt channel. 

---

# 1. Nhìn toàn bộ architecture trước

Có thể hình dung toàn bộ project như sau:

```text
                    SOURCE TRAINING
                         │
                         ▼
                COCO YOLO26-M pretrained
                         │
                         ▼
                 Clear Cityscapes
                  supervised train
                         │
                         ▼
              Source YOLO26-M (8 class)
                         │
                         ▼
                 Stage-1 AdaBN
              Foggy images, no labels
                         │
                         ▼
                Stage-1 YOLO26-M
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
   Dense RT-SFOD                   RASP-SFOD
          │                             │
          │                             │
   ┌──────┴──────┐               ┌──────┴──────┐
   │   Teacher   │               │   Teacher   │
   │ YOLO26-M    │               │ YOLO26-M    │
   │   dense     │               │   dense     │
   └──────┬──────┘               └──────┬──────┘
          │ weak                       │ weak
          │ DHF pseudo labels          │ DHF pseudo labels
          ▼                            ▼
   ┌─────────────┐              ┌────────────────┐
   │   Student   │              │     Student    │
   │ YOLO26-M    │              │ YOLO26-M latent│
   │   dense     │              │ + RASP gates   │
   └──────┬──────┘              └────────┬───────┘
          │                              │
    Detection Loss                 Detection Loss
          +                              +
        MARD                           MARD
                                         +
                               Taylor importance
                                         │
                                         ▼
                                    GMM selection
                                         │
                                         ▼
                                   Cost ranking
                                         │
                                         ▼
                                     Kneedle
                                         │
                                         ▼
                                 structured masks
                                         │
                                         ▼
                                  physical export
                                         │
                                         ▼
                          YOLO26-M RASP compact
```

---

# 2. Base detector thực tế: YOLO26-M

File architecture gốc là `yolo26.yaml`.

Nó khai báo:

```yaml
nc: 80
end2end: True
reg_max: 1
```

và scale `m` là:

```yaml
m: [0.50, 1.00, 512]
```

tức:

```text
depth multiplier = 0.50
width multiplier = 1.00
max channels     = 512
```

YOLO26-M COCO gốc khoảng **21.896M parameters**. 

Sau khi train Cityscapes 8 classes, Detect head đổi từ:

```text
80 classes
↓
8 classes
```

nên model main của chúng tôi còn:

```text
21,785,224 parameters
≈ 21.785M
```

Log source run xác nhận đây là YOLO26-M và pretrained weights được transfer vào model. 

---

# 3. YOLO26-M gồm 3 phần

Về mặt detector, ta có:

```text
INPUT
  │
  ▼
BACKBONE
  │
  ├── P3
  ├── P4
  └── P5
  │
  ▼
PAN / NECK
  │
  ├── P3/8
  ├── P4/16
  └── P5/32
  │
  ▼
END-TO-END DUAL DETECT HEAD
  ├── One-to-Many
  └── One-to-One
```

YOLO config thực tế đưa ba feature maps ở layer `16`, `19`, `22` vào Detect. 

---

# 4. Backbone YOLO26-M

Với input main experiment:

```text
1024 × 1024 × 3
```

flow có thể hình dung gần như:

```text
Input
1024×1024×3
       │
       ▼
Conv 3×3 stride 2
64 channels
512×512
       │
       ▼
Conv 3×3 stride 2
128 channels
256×256
       │
       ▼
C3k2
256 channels
       │
       ▼
Conv stride 2
256 channels
128×128              ← backbone P3
       │
       ▼
C3k2
512 channels
       │
       ▼
Conv stride 2
512 channels
64×64                ← backbone P4
       │
       ▼
C3k2
512 channels
       │
       ▼
Conv stride 2
512 channels
32×32                ← backbone P5
       │
       ▼
C3k2
       │
       ▼
SPPF
       │
       ▼
C2PSA
```

Architecture YAML khai báo chuỗi `Conv → C3k2 → ... → SPPF → C2PSA`; scale M dùng width `1.0` và max channel `512`, nên các stage khai báo 1024 được clamp theo scale parser.  Parser của fork cũng thực hiện scaling bằng `min(c2, max_channels) * width`. 

### C3k2 làm gì?

Có thể hiểu C3k2 là một **CSP-style feature aggregation block**. Quan trọng đối với RASP là bên trong các C3k2 này tồn tại các standard `Bottleneck`.

Một Bottleneck mà RASP quan tâm có dạng:

```text
                 ┌──────────── shortcut ────────────┐
                 │                                  │
input C ──► cv1 ──► hidden H ──► cv2 ──► output C ─┤
                 │                                  │
                 └──────────────────────────────────┘
```

RASP không prune toàn bộ C3k2. Nó đi vào trong và tìm các standard `Bottleneck` có:

```text
cv1.out_channels == cv2.in_channels
groups = 1
```



### SPPF

SPPF nằm cuối backbone, trước attention:

```text
C3k2
 ↓
SPPF
 ↓
C2PSA
```

Mục đích ở mức architecture là tăng receptive field / tổng hợp multi-scale context trước khi đưa feature sang neck.

### C2PSA

C2PSA là attention block ở high-level feature stage.

Điểm quan trọng với project của chúng tôi:

```text
C2PSA không bị RASP prune
```

RASP chủ động exclude:

```text
c2psa
attn
psa
one2one
one2many
dfl
```



---

# 5. Neck/PAN

Sau backbone, YOLO26 tạo feature pyramid theo cả **top-down** và **bottom-up**.

Top-down:

```text
P5
 │
Upsample ×2
 │
Concat với backbone P4
 │
C3k2
 │
 ▼
P4 feature
 │
Upsample ×2
 │
Concat với backbone P3
 │
C3k2
 │
 ▼
P3/8
```

Sau đó bottom-up:

```text
P3
 │
Conv stride 2
 │
Concat P4
 │
C3k2
 ▼
P4/16
 │
Conv stride 2
 │
Concat P5
 │
C3k2
 ▼
P5/32
```

Đây chính xác là layer 11–22 của YAML. 

Cuối cùng Detect nhận:

```text
P3 = layer 16
P4 = layer 19
P5 = layer 22
```

Với YOLO26-M 8-class hiện tại, ba input vào Detect thực tế là:

```text
P3 : 256 channels, stride 8
P4 : 512 channels, stride 16
P5 : 512 channels, stride 32
```

Ở input `1024×1024`:

```text
P3 ≈ 256 × 128 × 128
P4 ≈ 512 ×  64 ×  64
P5 ≈ 512 ×  32 ×  32
```

---

# 6. Detect head của YOLO26 khác YOLO truyền thống ở điểm quan trọng

Chúng tôi đang dùng:

```yaml
end2end: True
```

Nên Detect head có **hai assignment branches**:

```text
               P3 / P4 / P5
                    │
             ┌──────┴──────┐
             │             │
             ▼             ▼
        One-to-Many    One-to-One
           O2M            O2O
             │             │
             └──────┬──────┘
                    │
                 training
```

Code Detect tạo box regression `cv2` và classification `cv3`. Khi `end2end=True`, nó deepcopy chúng thành `one2one_cv2` và `one2one_cv3`. 

Trong forward:

```python
preds = one2many(x)

x_detach = x.detach()
one2one = one2one(x_detach)

preds = {
    "one2many": ...,
    "one2one": ...
}
```

Training trả cả hai branches.

Inference thì:

```text
chỉ O2O
↓
decode
↓
postprocess
```



Đây là lý do RT-SFOD có thể làm DHF: Teacher có đồng thời **O2O + O2M** trong training/adaptation.

---

# 7. Head box và classification bên trong

Mỗi scale có hai loại prediction path.

Box regression:

```text
feature
  ↓
Conv
  ↓
Conv
  ↓
Conv2d → bbox parameters
```

Classification path của non-legacy head dùng depthwise separable structure:

```text
feature
 ↓
DWConv
 ↓
1×1 Conv
 ↓
DWConv
 ↓
1×1 Conv
 ↓
Conv2d → class logits
```

Code này nằm trực tiếp trong `Detect`. 

Trong config YOLO26 của chúng tôi:

```text
reg_max = 1
```

nên DFL module của head trở thành Identity thay vì distribution projection nhiều bins. 

---

# 8. Sau source training: architecture không đổi

Source stage chỉ biến:

```text
COCO YOLO26-M
80 classes
```

thành:

```text
Cityscapes YOLO26-M
8 classes
```

Backbone/PAN topology vẫn giữ nguyên.

Tức:

```text
YOLO26-M COCO
      │
      ▼
fine-tune
      │
      ▼
YOLO26-M Cityscapes 8 classes
```

---

# 9. Stage-1 AdaBN cũng không thay architecture

Stage-1 dùng chính YOLO26-M source model.

Ta không thêm layer mới.

Ý tưởng đơn giản là:

```text
Source BN statistics
       │
Foggy images
       ▼
update BN running mean / variance
       │
       ▼
Target-adapted BN statistics
```

Model weights không được train như Stage-2; Stage-1 chủ yếu chạy target images qua mạng để cập nhật BN statistics.

Vì vậy:

```text
before AdaBN: YOLO26-M
after AdaBN : YOLO26-M
```

Architecture không thay đổi.

---

# 10. Stage-2 đưa YOLO26-M vào Mean Teacher

Đây mới là architecture tổng của RT-SFOD:

```text
                   Stage-1 checkpoint
                          │
                  ┌───────┴───────┐
                  │               │
                  ▼               ▼
              TEACHER          STUDENT
              YOLO26-M         YOLO26-M
               frozen          trainable
                  │               │
            weak image       strong image
                  │               │
                  ▼               ▼
           O2O + O2M          O2O + O2M
                  │               │
                 DHF          Detection loss
                  │               │
          pseudo labels            │
                  │               │
                  └────────► MARD ─┘
                                  │
                                  ▼
                                loss
                                  │
                                  ▼
                              Student
                                  │
                                  │ epoch end
                                  ▼
                             EMA update
                                  │
                                  ▼
                              Teacher
```

Teacher và Student đều được tạo từ **cùng Stage-1 checkpoint**. Teacher `eval()` và không require gradient; Student trainable. 

---

# 11. Weak view và strong view

Teacher nhìn ảnh **weakly augmented**.

Student nhìn **strongly augmented** version của cùng ảnh.

Weak:

```text
resize
+
shared horizontal flip
```

Strong ngoài resize/shared flip còn có thể gồm:

```text
affine scale/translation
perspective
HSV
brightness/contrast
gamma
RGB channel shuffle
Gaussian blur
Gaussian noise
salt-and-pepper corruption
```

Các transformation hình học được lưu lại để pseudo boxes Teacher tạo trên weak view có thể map chính xác sang strong view. 

---

# 12. DHF — Dual-Head Fusion

Đây là phần đầu tiên RT-SFOD thêm lên YOLO26.

Teacher weak image:

```text
Teacher
   │
   ├── O2O predictions
   │
   └── O2M predictions
```

Chúng tôi dùng:

```text
tau_o2o = 0.5
tau_o2m = 0.5
tau_no  = 0.2
tau_dup = 0.7
```

Flow chính xác:

```text
O2O predictions
confidence ≥ 0.5
       │
       ▼
high-precision anchors
       │
       │
       │          O2M predictions
       │          confidence ≥ 0.5
       │                 │
       │                 ▼
       │       compare IoU với O2O
       │                 │
       │          max IoU ≤ 0.2
       │                 │
       │                 ▼
       │          non-overlap extras
       │                 │
       │       classwise NMS @ 0.7
       │                 │
       └─────────┬───────┘
                 ▼
          fused pseudo labels
```

Code thực hiện đúng sequence này. 

Tư tưởng là:

```text
O2O
→ precision cao
→ làm anchor

O2M
→ có thể tìm thêm object mà O2O bỏ sót

DHF
→ chỉ lấy O2M bổ sung nếu không redundant với O2O
```

---

# 13. Pseudo labels được map sang Student

Teacher tạo:

```text
[x1, y1, x2, y2, confidence, class]
```

trên weak image.

Sau đó project áp dụng lại:

```text
affine matrix
perspective matrix
```

để chuyển box sang coordinate của strong image. Box quá nhỏ sau transformation bị loại.

Sau đó Student học chính các pseudo-label này. 

---

# 14. Student detection loss

Student output phải có:

```text
one2one
one2many
```

Pseudo boxes được đổi:

```text
xyxy
→ xywh
→ normalized
```

rồi đưa vào **native YOLO criterion**.

Code không viết một detection loss khác từ đầu; nó gọi trực tiếp criterion của YOLO model. 

Ta có thể viết:

[
L_{\text{det}}
==============

L_{\text{box}}
+
L_{\text{cls}}
+
L_{\text{dfl}}
]

với weighting do criterion YOLO xử lý.

---

# 15. MARD — regularization trên P3/P4/P5

DHF giải quyết **pseudo-label quality**.

MARD giải quyết **feature quality**.

Trước Detect head, code gắn một forward pre-hook để lấy chính:

```text
P3
P4
P5
```

đang được đưa vào Detect. 

Flow:

```text
Student strong image
        │
        ▼
      Backbone
        │
        ▼
       PAN
        │
  ┌─────┼─────┐
  ▼     ▼     ▼
 P3    P4    P5
  │     │     │
  └─────┼─────┘
        ▼
       MARD
```

---

# 16. MARD chọn feature như thế nào?

Pseudo boxes confidence thấp hơn `0.5` bị bỏ.

Tối đa:

```text
15 boxes/image
```

Box được phân về feature scale tùy kích thước:

```text
small  → P3
medium → P4
large  → P5
```

Sau đó mỗi box sample:

```text
8 foreground points
```

và mỗi level sample:

```text
128 background points
```

Các hyperparameter này được khai báo trực tiếp trong Stage-2. 

---

# 17. MARD có hai loss

### Variance term

Cho feature tokens (z), mỗi channel phải có đủ variation:

[
L_{var}
=======

\frac{1}{C}
\sum_c
\max(0,\gamma-\sigma_c)
]

với:

```text
γ = 1.0
```

Code:

```python
std = sqrt(var + eps)
relu(gamma - std)
```



Nó ngăn:

```text
nhiều channel collapse
→ feature trở nên giống/hằng
```

### Covariance term

Các channel không nên quá correlated.

Code normalize feature rồi tính covariance matrix và penalize off-diagonal terms. 

Nôm na:

```text
channel 1 ≈ channel 2 ≈ channel 3
```

là không tốt.

MARD muốn:

```text
các channel mang thông tin đa dạng hơn
```

---

# 18. Tổng MARD loss

Ở mỗi level:

[
L_{MARD}^{(l)}
==============

\alpha L_{var}^{(l)}
+
\beta L_{cov}^{(l)}
]

với:

```text
α = 1.0
β = 0.1
```

Sau đó:

[
L_{MARD}
========

L_{P3}+L_{P4}+L_{P5}
]

Code thực sự cộng cả ba levels. 

Tổng Student loss:

[
\boxed{
L_{total}
=========

L_{det}
+
\lambda(t,q)L_{MARD}
}
]

Trong đó (\lambda) phụ thuộc:

```text
training warmup
+
pseudo-label confidence
```



---

# 19. Mean Teacher update

Sau khi Student train hết **một epoch**:

[
\theta_T
\leftarrow
m\theta_T+(1-m)\theta_S
]

với:

```text
m = 0.999
```

Code:

```python
teacher_param =
    momentum * teacher_param
    + (1 - momentum) * student_param
```



Vậy feedback loop là:

```text
Teacher
  ↓
better pseudo labels
  ↓
Student learns
  ↓
EMA
  ↓
better Teacher
  ↓
...
```

---

# 20. Dense RT-SFOD architecture cuối cùng

Dense baseline chính xác là:

```text
Dense Teacher YOLO26-M
        │
        │ weak Foggy
        ▼
 O2O + O2M predictions
        │
        ▼
       DHF
        │
        ▼
 pseudo-labels
        │
        ▼
 geometry mapping
        │
        ▼
Dense Student YOLO26-M
 strong Foggy image
        │
        ├──── native YOLO detection loss
        │
        └──── P3/P4/P5 → MARD
                       │
                       ▼
                   total loss
                       │
                       ▼
                  update Student
                       │
                    epoch
                       ▼
                      EMA
                       │
                       ▼
                  update Teacher
```

---

# 21. RASP-SFOD giữ toàn bộ RT-SFOD đó

RASP **không thay DHF**.

RASP **không thay MARD**.

RASP **không thay Teacher**.

RASP **không thay Detect**.

RASP chỉ thêm một nhánh điều khiển Student:

```text
                          Student
                             │
                      forward + backward
                             │
                             ▼
                     Taylor importance
                             │
                             ▼
                         RASP logic
                             │
                             ▼
                     channel masks
```

Code RASP Stage-2 ghi rõ RT-SFOD Mean Teacher, DHF, MARD, augmentation, native losses và epoch EMA đều giữ nguyên; learning mechanism thêm vào là student-only adaptive structured pruning. 

---

# 22. RASP prune ở đâu?

Đây là điểm cực kỳ quan trọng.

**Không phải toàn bộ convolution.**

RASP chỉ tìm standard Bottleneck:

```text
input C
   │
   ▼
 cv1
   │
   ▼
hidden H    ← RASP prune ở đây
   │
   ▼
 cv2
   │
   ▼
output C
```

Suppose:

```text
C = 256
H = 128
```

và RASP quyết định bỏ 16 hidden channels.

Training-time:

```text
cv1 still outputs 128
↓
16 channels × gate=0
↓
cv2 still physically sees 128-shaped input
```

Deployment-time:

```text
cv1 output:
128 → 112

cv2 input:
128 → 112
```

Nhưng:

```text
block input  C = 256  unchanged
block output C = 256  unchanged
```

Vì thế toàn bộ graph bên ngoài block vẫn tương thích. Đây chính là lý do code chọn hidden Bottleneck làm physical pruning unit. 

---

# 23. Training-time RASP Student vẫn là dense latent model

Điểm này dễ nhầm.

Trong 60 epochs:

```text
stored weights:
dense

forward computation:
masked
```

Ví dụ:

```text
128 hidden weights vẫn tồn tại trong checkpoint
```

nhưng forward:

```text
channel 4  → 0
channel 9  → 0
channel 27 → 0
...
```

Forward hook đặt gate ngay sau `cv1`:

```python
gate = mask.view(...)
return output * gate
```



Lý do là Teacher và Student vẫn có cùng dense tensor shapes, nên EMA parameter-wise hoạt động trực tiếp.

---

# 24. Taylor importance

Khi Student backward, RASP đo cho mỗi hidden channel:

[
I_c
===

E\left[
|A_c \cdot \frac{\partial L}{\partial A_c}|
\right]
]

Trong code:

```python
imp = (activation * gradient).abs().mean(...)
```



Ý nghĩa:

```text
activation lớn
+
gradient lớn
→ channel quan trọng
```

Ngược lại:

```text
activation × gradient nhỏ
→ remove channel đó ít ảnh hưởng loss hơn
```

Importance được average trong epoch rồi EMA:

[
I_t
===

0.9I_{t-1}
+
0.1I_{new}
]



---

# 25. GMM quyết định channel nào thật sự thuộc nhóm thấp

RASP không đơn giản:

```text
sort importance
→ lấy 20% thấp nhất
```

Mỗi Bottleneck tự fit:

```text
GMM 1-component
vs
GMM 2-component
```

trên:

```text
log(Taylor importance)
```

Nếu 2-component hợp lý, nó xác định:

```text
low-importance cluster
high-importance cluster
```

Một channel chỉ trở thành candidate khi posterior thuộc low cluster đủ cao.

Main config:

```text
posterior ≥ 0.80
separation ≥ 1.0
BIC gain ≥ 0
```

Code GMM selection và posterior filtering nằm ở đây. 

Do đó RASP là:

```text
adaptive per-block
```

chứ không phải fixed sparsity mỗi layer.

---

# 26. Hardware-friendly packs

RASP không prune từng channel đơn lẻ ở bước apply.

Main setting:

```text
round_to = 8
```

Do đó:

```text
8 channels = 1 pack
```

Ví dụ:

```text
candidate channels:
[3, 17, 18, 22, 31, 35, 40, 44]
            ↓
          1 pack
```

Mục tiêu là width compact thân thiện hơn cho hardware.

Code gom candidate thành complete packs theo `round_to`. 

---

# 27. Cost-aware ranking

RASP không chỉ hỏi:

> channel nào ít quan trọng?

Nó còn hỏi:

> bỏ channel nào tiết kiệm compute nhiều nhất?

Mỗi hidden channel có estimated saving từ:

```text
cv1 output channel
+
cv2 matching input channel
```

Bao gồm:

```text
parameter saving
MAC saving
```



Score về bản chất là:

[
score
\propto
\frac{\text{importance}}
{\text{low-GMM confidence}\times\text{compute benefit}^{\gamma}}
]

Lower score được ưu tiên.

Với:

```text
γ = 1.0
```

nên RASP thích:

```text
low importance
+
high probability redundant
+
high MAC saving
```



---

# 28. Kneedle tự tìm pruning budget

Sau khi packs được global ranking:

```text
pack 1
pack 2
pack 3
...
```

RASP tạo cumulative curve:

```text
X = compute removed
Y = importance lost
```

Ví dụ:

```text
importance lost
^
|                         *
|                     *
|                  *
|               *
|           *
|      *
|   *
| *
+------------------------------> MACs removed
               ^
              knee
```

Ý tưởng:

```text
trước knee:
remove thêm compute nhưng mất ít importance

sau knee:
remove thêm bắt đầu mất importance nhanh
```

Code dùng Kneedle, có max-distance fallback. 

Vậy RASP **không cần fixed global target kiểu 30% hay 50%**.

---

# 29. Reliability gate

RASP chỉ được phép prune nếu DHF pseudo-labels đủ đáng tin.

Project dùng:

```text
mean DHF confidence ≥ 0.50
```

Và phải:

```text
warmup > 5 epochs
```

cũng như đủ recovery interval:

```text
3 epochs giữa pruning events
```

Controller kiểm tra ba điều kiện:

```text
warmup done
AND
recovery interval done
AND
reliability high enough
```



Vì vậy logic là:

```text
Teacher đang không chắc
→ đừng cắt capacity

Teacher đủ đáng tin
→ có thể prune
```

---

# 30. Step cap

Ngay cả khi Kneedle nói có thể prune rất nhiều, RASP không cắt hết ngay một lần.

Main setting:

```text
max new cost/event
=
5% baseline prunable MACs
```

Nên:

```text
prune
↓
train / recover
↓
re-estimate importance
↓
prune tiếp
```

thay vì:

```text
prune rất mạnh một phát
```

Controller áp dụng step cap này sau knee selection. 

---

# 31. RASP topology đầy đủ

Toàn bộ proposed architecture có thể vẽ thế này:

```text
                            Foggy image
                                 │
                       ┌─────────┴─────────┐
                       │                   │
                       ▼                   ▼
                   weak view          strong view
                       │                   │
                       ▼                   ▼
              ┌────────────────┐   ┌────────────────┐
              │ Dense Teacher  │   │ RASP Student   │
              │   YOLO26-M     │   │   YOLO26-M     │
              └───────┬────────┘   └───────┬────────┘
                      │                    │
               ┌──────┴──────┐             │
               ▼             ▼             │
              O2O           O2M            │
               │             │             │
               └──────┬──────┘             │
                      ▼                    │
                     DHF                   │
                      │                    │
               pseudo labels              │
                      │                    │
                      ▼                    │
              geometric mapping           │
                      │                    │
                      └──────────┬─────────┘
                                 ▼
                         Student YOLO loss
                                 +
                               MARD
                                 │
                                 ▼
                            total loss
                                 │
                              backward
                                 │
                ┌────────────────┴───────────────┐
                │                                │
                ▼                                ▼
         optimize Student                Taylor |A × grad|
                                                 │
                                                 ▼
                                         importance EMA
                                                 │
                                                 ▼
                                           per-block GMM
                                                 │
                                                 ▼
                                         low-importance
                                           candidates
                                                 │
                                                 ▼
                                         pack channels ×8
                                                 │
                                                 ▼
                                         cost-aware ranking
                                                 │
                                                 ▼
                                             Kneedle
                                                 │
                                                 ▼
                                      reliability + step gate
                                                 │
                                                 ▼
                                        monotonic masks
                                                 │
                                                 ▼
                                         next training epoch

epoch end:
Student dense latent weights
           │
           ▼
      EMA 0.999
           │
           ▼
    Dense Teacher
```

---

# 32. Physical compact export

Sau 60 epochs mới biến:

```text
masked dense Student
```

thành:

```text
physically smaller Student
```

Giả sử mask:

```text
H = 128
keep = 112
```

Exporter làm:

```text
cv1:
weight[keep]
output channels 128 → 112

BN after cv1:
weight[keep]
bias[keep]
running_mean[keep]
running_var[keep]

cv2:
weight[:, keep]
input channels 128 → 112
```

Code thực hiện chính xác việc slice `cv1`, BN và `cv2`. 

External block width vẫn không đổi:

```text
C → H' → C
```

thay vì:

```text
C → H → C
```

---

# 33. Vì vậy compact model vẫn giữ nguyên toàn bộ outer YOLO architecture

Sau compact:

```text
Backbone topology      unchanged
PAN topology           unchanged
P3/P4/P5 width         unchanged
Detect input width     unchanged
O2O head               unchanged
O2M head               unchanged
number of classes      unchanged
MARD interface         unchanged
```

Chỉ một số:

```text
internal Bottleneck hidden widths
```

nhỏ đi.

Đó là lý do physical export khá an toàn.

---

# 34. Pruning space thực tế của model

Audit của main YOLO26-M tìm thấy:

```text
eligible Bottleneck groups = 15
eligible hidden channels   = 1472
DepGraph safe              = 15 / 15
```



Tức RASP không kiểm soát toàn bộ 21.785M parameters.

Nó chỉ kiểm soát một subset bên trong 15 Bottleneck.

Đây cũng giải thích tại sao kết quả cuối là:

```text
Dense:
21.785M params

Compact:
21.255M params

reduction:
2.44%
```

chứ không phải 20–30%.

---

# 35. Architecture final

Model cuối cùng:

```text
YOLO26-M
8 classes
end-to-end
dual assignment head
P3/P4/P5 detector

+

RT-SFOD:
Mean Teacher
DHF
MARD

+

RASP:
Student-only
hidden Bottleneck structured pruning
target Taylor importance
per-block GMM
cost-aware ranking
Kneedle budget
DHF reliability gate
recovery cycles

+

physical hidden-channel compaction
```

Final architecture có:

```text
21.255M parameters
5.683G MACs @ 256
```

so với dense:

```text
21.785M parameters
5.957G MACs @ 256
```

và final Foggy result:

```text
Dense RT-SFOD:
mAP50    51.30
mAP50-95 33.52

RASP compact:
mAP50    50.90
mAP50-95 32.70
```

---

# 36. Những architecture trong repo nhưng KHÔNG phải main experiment này

Để không bị lẫn khi nhìn folder code:

| Thành phần         | Main experiment hiện tại? | Ghi chú                       |
| ------------------ | ------------------------- | ----------------------------- |
| **YOLO26-M**       | ✅                         | Base detector                 |
| C3k2 backbone      | ✅                         | Base YOLO26-M                 |
| P3/P4/P5 PAN       | ✅                         | Multi-scale neck              |
| O2O + O2M Detect   | ✅                         | Cần cho DHF                   |
| Mean Teacher       | ✅                         | RT-SFOD                       |
| DHF                | ✅                         | Pseudo-label fusion           |
| MARD               | ✅                         | Feature regularization        |
| **RASP**           | ✅                         | Proposed compression          |
| YOLO26-Lite        | ❌                         | Không dùng trong main result  |
| C2fFaster / PConv  | ❌                         | Thuộc nhánh Lite              |
| Segment26          | ❌                         | Không phải C2F detection main |
| Mask-DHF           | ❌                         | Segmentation branch           |
| CARD               | ❌                         | Không phải RASP main          |
| QAT / quantization | ❌                         | Không dùng trong RASP v1      |

RASP code thậm chí ghi rõ module này **không chứa quantization**. 

---

## Nếu tóm gọn architecture của luận văn trong một hình

```text
                 TARGET FOGGY IMAGE
                        │
            ┌───────────┴───────────┐
            │                       │
         Weak Aug                Strong Aug
            │                       │
            ▼                       ▼
   ┌─────────────────┐     ┌────────────────────┐
   │ Dense YOLO26-M  │     │   YOLO26-M Student│
   │     Teacher     │     │ + RASP hidden gates│
   └────────┬────────┘     └──────────┬─────────┘
            │                         │
        O2O + O2M                     │
            │                         │
            ▼                         │
           DHF                        │
            │                         │
       Pseudo Labels ───────────────► YOLO Loss
                                      │
                           P3/P4/P5 ─►MARD
                                      │
                                      ▼
                                  Total Loss
                                      │
                                   backward
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                   Student update          Taylor Importance
                                                   │
                                                   ▼
                                          GMM → Cost Rank
                                                   │
                                                   ▼
                                               Kneedle
                                                   │
                                                   ▼
                                              RASP Masks

                  Student dense latent parameters
                                │
                         EMA once/epoch
                                │
                                ▼
                         Dense Teacher

After 60 epochs:
RASP masks
    │
    ▼
physical Bottleneck compaction
    │
    ▼
YOLO26-M RASP COMPACT
```

**Tóm lại:** architecture đã dùng là **YOLO26-M end-to-end dual-head detector đặt trong RT-SFOD Mean-Teacher framework, với DHF tạo pseudo-label, MARD regularize P3/P4/P5, và RASP chỉ prune có cấu trúc các hidden channels bên trong dependency-safe Bottleneck của Student, sau đó physical compact để tạo detector nhỏ hơn.**
