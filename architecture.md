# RASP-SFOD

**Reliability-Aware Structured Pruning for Source-Free Object Detection**

Mục tiêu của phương pháp là:

> **Nén Student ngay trong quá trình source-free domain adaptation, nhưng chỉ prune khi target-domain evidence cho thấy channel đó ít quan trọng và Teacher đủ đáng tin cậy.**

Điểm quan trọng là RASP-SFOD **không phải “train xong rồi prune”**. Việc pruning trở thành **một phần của quá trình adaptation**.

---

# 1. Bài toán mà phương pháp đang giải quyết

Bối cảnh là Source-Free Object Detection.

Ta có:

```text
Source domain:
Clear Cityscapes
→ có image + label
```

sau khi source training xong, ta chỉ còn:

```text
Source-trained YOLO26-M checkpoint
```

Khi chuyển sang target domain:

```text
Target domain:
Foggy Cityscapes
→ chỉ được dùng image
→ KHÔNG được dùng target label trong adaptation
```

Mục tiêu thông thường của RT-SFOD là:

[
\text{Source model}
\rightarrow
\text{adapt to target}
\rightarrow
\text{better target accuracy}
]

Còn proposed method của bạn muốn đồng thời đạt:

[
\boxed{
\text{Target adaptation}
+
\text{Model compression}
}
]

tức là cuối cùng có một model:

```text
nhỏ hơn
↓
ít FLOPs hơn
↓
nhanh hơn
↓
nhưng vẫn giữ target-domain accuracy tốt
```

Khó khăn nằm ở chỗ:

> Nếu prune model quá sớm hoặc prune sai channel trong source-free adaptation, pseudo-label vốn đã noisy sẽ làm Student suy giảm rất nhanh.

Do đó RASP không hỏi đơn giản:

> “Channel nào có weight nhỏ?”

mà hỏi:

> **“Trên target domain hiện tại, channel nào thực sự ít quan trọng, có redundancy, tiết kiệm compute đáng kể, và tại thời điểm này Teacher có đủ đáng tin để ta prune hay chưa?”**

Đó chính là tư tưởng cốt lõi.

---

# 2. Tổng pipeline

Pipeline hoàn chỉnh là:

```text
Clear Cityscapes train + labels
        ↓
Supervised source training
        ↓
YOLO26-M source checkpoint
        ↓
Foggy Cityscapes train images only
        ↓
AdaBN
        ↓
Target-warmed YOLO26-M checkpoint
        ↓
RASP graph audit
        ↓
Mean Teacher adaptation
Teacher (dense) → pseudo-labels → Student
                         ↓
                RT-SFOD losses
                         ↓
                target-aware Taylor
                         ↓
                GMM redundancy
                         ↓
                cost-aware ranking
                         ↓
                Kneedle adaptive budget
                         ↓
                reliability gate
                         ↓
              progressive structured prune
                         ↓
                   recovery training
                         ↓
               repeat during adaptation
                         ↓
             final masked Student
                         ↓
             physical graph compaction
                         ↓
        compact RASP-SFOD-Y26 model
```

Có thể chia method thành 3 tầng:

```text
RT-SFOD adaptation backbone
        +
RASP pruning controller
        +
final structural compaction
```

---

# 3. Phần nào giữ nguyên từ RT-SFOD?

RASP **không thay thế RT-SFOD**.

RT-SFOD vẫn là adaptation framework chính.

Bạn giữ:

```text
Mean Teacher
+
Dual-Head Fusion (DHF)
+
MARD
```

RASP được gắn vào bên Student.

Tức là:

[
\boxed{
\text{RASP-SFOD}
================

\text{RT-SFOD}
+
\text{adaptive structured pruning}
}
]

Điều này rất quan trọng về mặt paper.

Bạn không muốn nói:

> “We propose a completely new SFOD framework.”

mà chính xác hơn là:

> “We introduce a reliability-aware structured pruning mechanism into source-free Mean-Teacher adaptation.”

---

# 4. Mean Teacher trong method

Sau AdaBN, tạo:

```text
Teacher
Student
```

ban đầu cùng từ checkpoint đã được target-warm-up.

Trong mỗi target batch:

```text
target image
     ↓
 ┌──────────────┐
 │              │
weak view    strong view
 │              │
Teacher       Student
 │              │
pseudo labels   predictions
 └──────→ loss ←┘
```

Teacher không học bằng optimizer.

Teacher cập nhật bằng EMA:

[
\theta_T
\leftarrow
\alpha\theta_T
+
(1-\alpha)\theta_S
]

với:

[
\alpha \approx 0.999
]

Teacher đóng vai trò:

```text
pseudo-label generator
+
reliability reference
```

Student mới là network thực sự được gradient update.

---

# 5. Vì sao Teacher phải dense?

Đây là một thiết kế rất quan trọng của RASP.

Teacher:

```text
DENSE
```

Student:

```text
progressively PRUNED
```

Không prune Teacher.

Lý do đầu tiên là ổn định pseudo-label.

Nếu cả Teacher và Student cùng bị prune:

```text
Student bị mất capacity
        ↓
Teacher EMA cũng mất capacity
        ↓
pseudo-label quality giảm
        ↓
Student học từ pseudo-label yếu hơn
        ↓
feedback loop tiêu cực
```

Trong RASP:

```text
Teacher = stable dense reference

Student = compression target
```

Do đó Teacher vẫn giữ khả năng biểu diễn đầy đủ để hướng dẫn Student.

---

# 6. Một vấn đề kỹ thuật: Teacher và Student vẫn phải shape-compatible

Mean Teacher cần EMA giữa hai model có tensor shape giống nhau.

Vì vậy bạn **không physically delete channel trong quá trình training**.

Thay vào đó:

```text
Student latent weights
vẫn giữ original shape
```

nhưng forward sử dụng:

```text
channel masks / gates
```

Ví dụ:

[
z'_g = m_g z_g
]

với:

[
m_g \in {0,1}
]

Nếu:

```text
m_g = 1
```

channel active.

Nếu:

```text
m_g = 0
```

channel bị prune trong forward.

Nhưng tensor weight vẫn tồn tại.

Do đó:

```text
Teacher shape = Student latent shape
```

EMA vẫn hoạt động.

Đây là khác biệt rất quan trọng giữa:

```text
training-time masking
```

và:

```text
deployment-time physical pruning
```

---

# 7. RT-SFOD pseudo-labels: DHF

YOLO26 là dual-head end-to-end detector.

Có:

```text
O2O = one-to-one head
O2M = one-to-many head
```

RT-SFOD không chỉ lấy toàn bộ prediction từ một head.

DHF dùng:

```text
O2O
→ precision cao
→ làm anchor pseudo-label
```

và bổ sung một số O2M prediction không redundant.

Conceptually:

```text
O2O high-confidence boxes
        ↓
main pseudo labels

O2M high-confidence candidates
        ↓
check overlap with O2O
        ↓
low-overlap candidate
        ↓
keep
```

Current settings:

[
\tau_{O2O}=0.5
]

[
\tau_{O2M}=0.5
]

O2M candidate được thêm nếu:

[
\max IoU(O2M,O2O)\leq0.2
]

sau đó class-wise NMS:

[
IoU_{NMS}=0.7
]

DHF cho RASP hai thứ:

```text
1. pseudo-label supervision
2. reliability signal
```

---

# 8. MARD vẫn giữ nguyên

MARD:

**Multi-scale Adaptive Representation Diversification**

được áp dụng lên PAN features:

```text
P3
P4
P5
```

Mục đích là hạn chế feature representation collapse dưới domain shift.

Loss tổng của Student có thể viết đơn giản:

[
L_{\text{SFOD}}
===============

L_{\text{det}}
+
\lambda_M L_{\text{MARD}}
]

Đây chính là loss mà RASP dùng để tính **target-domain sensitivity**.

Và đây là điểm hay:

> Pruning criterion không được tính từ source data nữa. Nó được lấy trực tiếp từ loss mà Student đang tối ưu trên target domain.

---

# 9. RASP audit nằm ở đâu?

Sau AdaBN, trước 60-epoch adaptation:

```text
AdaBN checkpoint
      ↓
RASP audit
```

Audit không train model.

Nó phân tích graph để tìm:

```text
safe pruning groups
```

Ví dụ đơn giản:

```text
C
↓
Conv1
↓
H channels
↓
Conv2
↓
C
```

Ta có thể prune hidden dimension:

[
H
]

mà vẫn giữ external dimension:

[
C
]

không đổi.

Audit còn xác định dependency:

```text
Conv output
↓
BN
↓
next Conv input
```

Nếu prune output channel (k) của Conv A thì khi compact:

```text
BN channel k
```

và:

```text
input channel k của Conv B
```

cũng phải bỏ.

Nếu không, graph sẽ lỗi shape.

---

# 10. Đơn vị pruning của RASP là group, không phải weight

Đây là **structured pruning**.

Không phải:

```text
delete individual scalar weights
```

mà là:

```text
channel / dependency group
```

ký hiệu:

[
g
]

Một group có thể đại diện cho:

```text
one output channel
+
associated BN channel
+
corresponding downstream input channels
```

Vì vậy model cuối cùng có thể thực sự:

```text
fewer channels
↓
fewer parameters
↓
fewer MACs/FLOPs
↓
faster inference
```

Khác với unstructured sparsity chỉ tạo nhiều số 0.

---

# 11. Target-aware Taylor importance

Đây là thành phần quan trọng nhất để trả lời:

> “Channel nào quan trọng đối với TARGET domain?”

Với activation của pruning group:

[
z_g
]

Taylor importance:

[
I_g
===

EMA
\left[
\left|
z_g
\frac{\partial L_{\text{SFOD}}}{\partial z_g}
\right|
\right]
]

Hiểu trực quan:

Nếu thay đổi channel (g) một chút mà loss thay đổi mạnh:

```text
gradient lớn
×
activation lớn
```

thì:

[
I_g \text{ lớn}
]

→ channel quan trọng.

Nếu:

[
I_g \text{ nhỏ}
]

→ bỏ channel đó có khả năng gây ít damage.

Đặc biệt, loss ở đây là:

[
L_{\text{det}}
+
\lambda_M L_{\text{MARD}}
]

nên importance phản ánh:

```text
pseudo-label detection usefulness
+
target feature representation usefulness
```

chứ không đơn thuần magnitude của weight.

---

# 12. Vì sao cần EMA cho Taylor?

Importance của một batch rất noisy.

Ví dụ:

```text
batch A:
car nhiều

batch B:
person nhiều

batch C:
gần như không có object
```

Nếu prune dựa vào một batch:

[
I_g^{(t)}
]

thì dễ quyết định sai.

Do đó:

[
\bar I_g^{(t)}
==============

\beta \bar I_g^{(t-1)}
+
(1-\beta)I_g^{(t)}
]

RASP sử dụng accumulated/EMA importance.

Điều này làm pruning decision ổn định hơn.

---

# 13. Nhưng importance thấp chưa chắc là redundancy

Đây là lý do có GMM.

Giả sử một stage có importance:

```text
0.01
0.012
0.015
0.018
0.31
0.37
0.40
0.45
```

nhìn rất rõ có:

```text
low-importance cluster
high-importance cluster
```

Nhưng stage khác:

```text
0.12
0.13
0.14
0.15
0.16
0.17
```

không có separation rõ.

Nếu cứ prune bottom 30%:

```text
cả hai stage đều mất 30%
```

→ không hợp lý.

RASP dùng **Gaussian Mixture Model** trên:

[
x_g=\log(I_g+\epsilon)
]

---

# 14. GMM redundancy discovery

Với mỗi candidate block/stage, fit:

```text
GMM K=1
```

và:

```text
GMM K=2
```

Sau đó compare BIC.

Nếu:

```text
K=1 tốt hơn
```

→ không có bằng chứng rõ ràng rằng tồn tại redundant low-importance population.

RASP có thể:

```text
protect block
```

Nếu:

```text
K=2 tốt hơn
```

→ có hai populations.

Component có mean thấp:

```text
low-importance component
```

được xem như redundancy candidate.

Ta có:

[
P(
\text{unimportant}
\mid I_g
)
]

Nếu posterior đủ cao:

```text
group g
→ eligible pruning candidate
```

Điều này khác threshold cứng.

---

# 15. Vai trò của GMM

Taylor trả lời:

> “Importance bao nhiêu?”

GMM trả lời:

> “Importance thấp này có thật sự tạo thành một redundancy population không?”

Hai bước phối hợp:

```text
Taylor
→ sensitivity

GMM
→ redundancy structure
```

Nhờ vậy RASP không ép mọi layer phải prune cùng một tỷ lệ.

Có thể xảy ra:

```text
Stage A → prune nhiều
Stage B → prune ít
Stage C → không prune
Stage D → prune vừa
```

Đây là tính **adaptive** của method.

---

# 16. Sau GMM vẫn chưa prune ngay

Giả sử có hai candidate:

```text
A:
importance = 0.02
saving = 0.1 GFLOPs

B:
importance = 0.03
saving = 1.2 GFLOPs
```

A có importance thấp hơn.

Nhưng B tiết kiệm compute nhiều hơn rất nhiều.

Do đó RASP dùng cost-aware ranking.

---

# 17. Cost-aware pruning score

Bạn dùng:

[
Score_g
=======

\frac{I_g}
{(\Delta C_g)^\gamma}
]

Trong đó:

[
I_g
]

= target importance.

[
\Delta C_g
]

= compute saving nếu prune group (g).

[
\gamma
]

= mức độ ưu tiên efficiency.

Score thấp nghĩa là:

> **mất ít information trên mỗi đơn vị compute tiết kiệm được.**

Vì vậy:

```text
lower score
→ better pruning candidate
```

Đây rất quan trọng vì mục tiêu paper không chỉ là sparsity.

Bạn muốn:

[
\boxed{
\text{accuracy–efficiency trade-off}
}
]

---

# 18. Compute cost được lấy từ đâu?

Ở Phase 1:

```text
MACs / FLOPs
```

có thể dùng làm (\Delta C_g).

Audit cung cấp structural cost estimate.

Ví dụ:

```text
remove 8 channels ở early high-resolution layer
```

có thể tiết kiệm nhiều hơn:

```text
remove 8 channels ở late low-resolution layer
```

Cho nên channel count alone không đủ.

Ở final experiment, bạn vẫn phải report:

```text
Params
FLOPs/MACs
model size
actual latency/FPS
```

vì FLOPs không đồng nghĩa hoàn toàn với wall-clock speed.

---

# 19. RASP không dùng fixed 30% pruning ratio

Đây là một điểm khác của proposed method.

Các pruning method thông thường có thể đặt:

[
r = 0.3
]

và prune 30%.

RASP không muốn đặt:

```text
20%
30%
40%
```

bằng tay cho main method.

Thay vào đó nó xây dựng một frontier:

```text
candidate 1
→ little compression / little information loss

candidate 2
→ more compression / more information loss

candidate 3
→ more compression / more information loss

...
```

---

# 20. Compression–information frontier

Sau cost-aware ranking:

```text
g1, g2, g3, ..., gn
```

RASP progressively giả định prune:

```text
{g1}
{g1,g2}
{g1,g2,g3}
...
```

và tính:

[
x_k=
\frac{\text{cumulative compute removed}}
{\text{candidate compute}}
]

[
y_k=
\frac{\text{cumulative importance lost}}
{\text{candidate importance}}
]

Ta có curve:

[
(x_k,y_k)
]

---

# 21. Kneedle chọn adaptive budget

RASP dùng knee/elbow point.

Ý tưởng:

Ban đầu:

```text
compression tăng nhanh
information loss tăng chậm
```

→ đáng prune.

Nhưng sau một điểm:

```text
muốn thêm một ít compression
→ phải sacrifice rất nhiều importance
```

Điểm chuyển đó là:

```text
knee
```

RASP dừng ở đó.

Do vậy pruning budget được quyết định từ:

```text
target-domain importance
+
available redundancy
+
compute saving
```

chứ không phải:

```text
target mAP
```

hay:

```text
user-set fixed ratio
```

Điều này đặc biệt quan trọng trong SFOD vì:

> Foggy validation GT không được phép dùng để chọn pruning ratio.

---

# 22. Tại sao vẫn cần reliability gate?

Ngay cả khi Taylor + GMM + Kneedle nói:

```text
“nên prune”
```

thì RASP vẫn hỏi:

> “Teacher hiện tại có đáng tin không?”

Ví dụ đầu adaptation:

```text
domain gap còn lớn
Teacher pseudo-label yếu
```

Nếu lúc đó prune:

```text
Student capacity ↓
+
supervision noisy
```

→ nguy hiểm.

Do đó RASP có **reliability-aware gate**.

---

# 23. Reliability signal

Reliability có thể dựa trên Teacher/DHF statistics như:

```text
average accepted pseudo-label confidence
pseudo-label count
DHF quality/stability
zero-pseudo rate
```

Ví dụ đơn giản:

[
R_t =
\text{mean confidence of accepted pseudo labels}
]

Nếu:

[
R_t < \tau_R
]

thì:

```text
NO NEW PRUNING
```

Student vẫn train và giữ masks hiện tại.

Nếu:

[
R_t \ge \tau_R
]

thì mới cho controller activate pruning.

Current natural threshold khoảng:

[
\tau_R \approx 0.5
]

phù hợp với DHF confidence threshold.

---

# 24. Vì sao gọi là Reliability-Aware?

Đây chính là chữ **RA** trong RASP.

Pruning decision không chỉ dựa trên structural criterion.

Nó còn phụ thuộc:

[
\text{Teacher reliability at time }t
]

Nên:

```text
same Student
same importance distribution
```

nhưng:

```text
Teacher unreliable
→ postpone pruning

Teacher reliable
→ allow pruning
```

Điều này phù hợp đặc biệt với source-free self-training.

---

# 25. Structured Pruning là chữ SP

**SP = Structured Pruning**

Method không tạo sparse random weights.

Nó loại:

```text
channels
dependency-consistent channel groups
```

để cuối cùng có thể compact graph thật.

Do đó tên:

[
\boxed{
\text{RASP}
===========

\text{Reliability-Aware Structured Pruning}
}
]

---

# 26. Progressive pruning

RASP không prune toàn bộ một lần.

Thay vào đó:

```text
warmup
↓
estimate importance
↓
discover redundancy
↓
prune a set
↓
continue adaptation
↓
recover
↓
re-estimate importance
↓
next pruning decision
```

Có thể biểu diễn:

[
S_0
\rightarrow
S_1
\rightarrow
S_2
\rightarrow
...
\rightarrow
S_K
]

trong đó:

[
S_{k+1}\subseteq S_k
]

về active channels.

Masks là monotonic:

```text
once pruned
→ remain pruned
```

trong main design.

---

# 27. Tại sao cần recovery phase?

Sau khi prune:

```text
representation bị perturb
```

Student có thể giảm accuracy tạm thời.

Nhưng thay vì:

```text
prune
↓
separate supervised fine-tuning
```

RASP tận dụng chính:

```text
RT-SFOD adaptation
+
MARD
```

để recover.

Tức là:

[
\boxed{
\text{adaptation itself becomes pruning recovery}
}
]

Đây là một trong những hypothesis quan trọng nhất của proposed method.

---

# 28. Đây cũng là lý do experiment “in-loop vs post-pruning” rất quan trọng

Bạn nên có:

```text
A. Dense RT-SFOD
```

```text
B. Dense RT-SFOD
   ↓
   train complete
   ↓
   prune afterward
```

và:

```text
C. RASP-SFOD
   ↓
   prune during adaptation
   ↓
   recovery happens naturally
```

B và C phải matched compactness càng gần càng tốt.

Nếu:

[
mAP_C > mAP_B
]

ở cùng Params/FLOPs,

thì có evidence rằng:

> **Joint adaptation and pruning is better than post-hoc pruning.**

Đây là experiment rất có giá trị cho paper.

---

# 29. Một iteration RASP có thể hiểu như thế này

Giả sử hiện tại là epoch (t).

Teacher nhận weak Foggy image:

[
x_t^w
]

và tạo DHF pseudo labels:

[
\hat y_t
]

Student nhận strong view:

[
x_t^s
]

Student tối ưu:

[
L_{\text{SFOD}}
===============

L_{\text{det}}
(x_t^s,\hat y_t)
+
\lambda_M L_{\text{MARD}}
]

Từ backward, RASP cập nhật:

[
I_g
===

EMA
\left[
\left|
z_g
\frac{\partial L_{\text{SFOD}}}
{\partial z_g}
\right|
\right]
]

Khi tới pruning decision point:

```text
importance
↓
log transform
↓
GMM K=1 vs K=2
↓
redundancy candidates
↓
dependency/cost info
↓
Score = I / cost^γ
↓
rank
↓
compression-information curve
↓
Kneedle
↓
candidate pruning set
```

Sau đó reliability gate:

```text
Teacher reliable?
```

Nếu không:

```text
pruning proposal rejected/postponed
```

Nếu có:

```text
activate masks
```

rồi tiếp tục training.

---

# 30. Một ví dụ rất trực quan

Giả sử audit phát hiện 100 pruning groups.

Sau warmup:

```text
Taylor importance computed for 100 groups
```

GMM tìm ra:

```text
35 groups
```

thuộc low-importance components.

Cost-aware ranking sắp:

```text
g17
g42
g8
g90
...
```

Kneedle cho rằng tốt nhất chỉ prune:

```text
top 14 candidate groups
```

vì từ group thứ 15 trở đi information cost tăng mạnh.

Nhưng epoch đó Teacher reliability:

[
R_t=0.44
]

threshold:

[
0.5
]

RASP:

```text
DO NOT PRUNE
```

Vài epoch sau:

[
R_t=0.71
]

Taylor/GMM được recompute.

Lần này Kneedle chọn:

```text
12 groups
```

RASP mới activate masks.

Sau đó Student tiếp tục RT-SFOD để recover.

---

# 31. Tại sao recompute importance sau pruning?

Bởi vì importance là **context-dependent**.

Trước pruning:

```text
channel A và B có thể redundant nhau
```

sau khi prune A:

```text
B có thể trở nên rất quan trọng
```

Nếu dùng importance ban đầu mãi:

```text
A low
B low
→ prune both
```

có thể phá model.

Cho nên:

```text
prune
↓
recover
↓
recompute
```

là thiết kế đúng hơn.

---

# 32. O2O detached có ảnh hưởng gì tới Taylor?

YOLO26 end-to-end head có O2O branch dùng detached features.

Do đó gradient O2O không propagate giống O2M về backbone.

Vì vậy backbone Taylor importance trong RASP chủ yếu phản ánh:

```text
O2M detection supervision
+
MARD gradients
```

Điều này không phải bug.

Đó là hệ quả của architecture.

Và đây cũng là lý do MARD khá hữu ích đối với RASP:

> Nó cung cấp thêm target-domain representation signal tới backbone/PAN.

---

# 33. Tại sao không prune Detect head?

Prediction head có rất nhiều dependency nhạy cảm:

```text
class channels
bbox regression dimensions
one-to-one branch
one-to-many branch
```

Prune prediction channel trực tiếp có thể thay đổi semantic output dimension.

Ví dụ:

```text
8 classes
```

không thể tùy tiện xóa class output channel chỉ vì Taylor thấp.

Do đó initial safe search space ưu tiên:

```text
backbone
+
neck/PAN
```

và bảo vệ:

```text
Detect output structures
```

---

# 34. Masked model chưa phải compact model

Điểm này cần cực kỳ rõ khi viết paper.

Sau 60 epochs:

```text
Student vẫn có original tensor shapes
```

dù một số channels có:

```text
mask = 0
```

Nếu chỉ tính:

```python
sum(p.numel())
```

thì Params vẫn gần như không đổi.

Do đó bạn **không được claim actual parameter reduction** dựa trên masked Student.

---

# 35. Physical compaction ở cuối

Sau adaptation:

```text
final masks
↓
keep indices
↓
rewrite graph
```

Ví dụ:

```text
Conv A:
out_channels 256
```

mask giữ:

```text
192 channels
```

thì compact model:

```text
Conv A
256 → 192
```

BN:

```text
256 → 192
```

next Conv input:

```text
256 → 192
```

tất cả dependency phải slice cùng index.

Sau đó kiểm tra:

[
f_{\text{masked-dense}}(x)
\approx
f_{\text{compact}}(x)
]

Sai số phải rất nhỏ.

Smoke test trước đây của framework đã đạt kiểu:

```text
masked_vs_compact_max_abs_diff = 0
```

cho supported pattern.

---

# 36. Chỉ sau compaction mới report compression thật

Final compact artifact mới được dùng để tính:

[
\text{Params}
]

[
\text{FLOPs / MACs}
]

[
\text{model size}
]

[
\text{latency}
]

[
\text{FPS}
]

và cuối cùng target-domain accuracy:

[
mAP@0.5
]

[
mAP@0.5:0.95
]

---

# 37. RASP không dùng target labels để quyết định pruning

Đây là requirement cực kỳ quan trọng.

Không được dùng Foggy val GT để chọn:

```text
pruning ratio
Kneedle point
GMM threshold
epoch
reliability threshold
best compactness
```

Adaptation chỉ dùng:

```text
Foggy train images
+
Teacher pseudo labels
+
internal statistics
```

Foggy val labels chỉ dùng:

```text
AFTER model is frozen
→ final benchmark evaluation
```

Như vậy mới giữ đúng source-free protocol.

---

# 38. Những thành phần adaptive của RASP

Có thể xem RASP trả lời 5 câu hỏi:

| Question                       | Component                            |
| ------------------------------ | ------------------------------------ |
| **Prune ở đâu?**               | graph audit + safe dependency groups |
| **Channel nào ít quan trọng?** | target-aware Taylor                  |
| **Có redundancy thật không?**  | GMM                                  |
| **Prune cái nào có lợi nhất?** | cost-aware ranking                   |
| **Prune bao nhiêu?**           | Kneedle                              |
| **Prune lúc nào?**             | Teacher reliability gate             |

Đây là cách rất hay để trình bày method trong presentation/paper.

---

# 39. Điểm khác biệt so với conventional pruning

Conventional pipeline:

```text
train
↓
prune fixed 30%
↓
fine-tune
```

RASP:

```text
source-free adaptation
        ↕
importance estimation
        ↕
redundancy discovery
        ↕
budget discovery
        ↕
reliability control
        ↕
progressive pruning
```

Tức là compression **co-evolves with adaptation**.

---

# 40. Điểm khác biệt so với magnitude pruning

Magnitude pruning:

[
I_g=|W_g|
]

RASP:

[
I_g
===

EMA
\left[
\left|
z_g
\frac{\partial L_{SFOD}}{\partial z_g}
\right|
\right]
]

Magnitude hỏi:

> weight nhỏ không?

RASP hỏi:

> channel này có đóng góp vào target adaptation hiện tại không?

Đây là khác biệt conceptually rất lớn.

---

# 41. Điểm khác biệt so với fixed Taylor pruning

Ngay cả Taylor pruning thường vẫn làm:

```text
rank all channels
↓
prune fixed 30%
```

RASP thêm:

```text
GMM
→ xác định redundancy

cost-aware
→ efficiency-aware selection

Kneedle
→ automatic budget

reliability gate
→ timing control
```

Nên Taylor chỉ là **một signal trong controller**, không phải toàn bộ proposed method.

---

# 42. Ý nghĩa của “target-adaptive”

RASP adaptive theo cả:

```text
domain
```

và:

```text
training state
```

Ví dụ một group:

```text
important ở Clear Cityscapes
```

nhưng:

```text
redundant trên Foggy
```

thì RASP có thể prune.

Ngược lại:

```text
weight magnitude nhỏ
```

nhưng Foggy gradients cho thấy nó rất quan trọng:

[
I_g \text{ lớn}
]

→ giữ lại.

Đó là target-adaptive pruning.

---

# 43. Ý nghĩa khoa học lớn nhất của method

Hypothesis trung tâm của bạn có thể diễn đạt:

> Domain adaptation itself changes the functional importance of channels; therefore, model compression for SFOD should be decided using target-domain adaptation signals rather than source-domain or static weight statistics.

Và hypothesis thứ hai:

> Compression should be introduced progressively while the Student is still adapting, so self-training and representation regularization can recover pruning-induced degradation.

Hypothesis thứ ba:

> Pruning decisions should be conditioned on pseudo-label reliability because structural capacity reduction is particularly risky when self-training supervision is unreliable.

Ba câu này gần như là backbone lý luận của paper.

---

# 44. Main contributions có thể đóng khung như thế nào

Không nên claim quá sớm “first ever”, nhưng contribution logic hiện tại có thể là:

**Target-aware importance.**

RASP uses target-domain SFOD gradients to estimate structured channel sensitivity instead of relying on source data or static weight magnitude.

**Probabilistic redundancy discovery.**

Rather than enforcing a fixed pruning ratio, RASP models importance distributions using GMMs to identify whether a low-importance population is actually present.

**Adaptive efficiency-aware budget.**

Candidate groups are ranked by information loss relative to compute saving, while Kneedle determines the pruning frontier without target labels.

**Reliability-aware progressive compression.**

Pruning is permitted only when Teacher/DHF pseudo supervision is sufficiently reliable, followed by continued RT-SFOD/MARD adaptation for recovery.

---

# 45. Ablation nào chứng minh từng component?

Full RASP phải so với:

```text
Dense RT-SFOD
```

để chứng minh compression.

So với:

```text
post-adaptation pruning
```

để chứng minh in-loop recovery.

So với:

```text
fixed-budget Taylor
```

để chứng minh adaptive budget.

So:

```text
GMM vs Otsu
```

để chứng minh probabilistic redundancy detection.

Có thể thêm:

```text
RASP without reliability gate
```

để chứng minh gate.

Nếu full RASP:

```text
accuracy gần dense
+
Params ↓
+
FLOPs ↓
+
latency ↓
```

thì main objective đạt.

---

# 46. Main expected outcome

Ví dụ final table lý tưởng:

| Method               |     mAP50 ↑ | mAP50-95 ↑ |  Params ↓ |   FLOPs ↓ | FPS ↑ |
| -------------------- | ----------: | ---------: | --------: | --------: | ----: |
| Source-only YOLO26-M |           x |          x |     21.8M |     74.8G |     x |
| AdaBN                |           x |          x |     21.8M |     74.8G |     x |
| RT-SFOD-Y26          |    **48.x** |          x |     21.8M |     74.8G |     x |
| Post-pruned RT-SFOD  |        46.x |          x |     16.xM |     55.xG |     x |
| **RASP-SFOD-Y26**    | **47–48.x** |          x | **16.xM** | **55.xG** | **↑** |

Những số trên chỉ minh họa logic, **không phải expected result được phép báo trước**.

Mục tiêu thực nghiệm hợp lý là:

[
\text{significant compression}
]

với:

[
\Delta mAP_{50}
]

nhỏ, lý tưởng dưới khoảng 1–1.5 điểm so với dense RT-SFOD ở matched protocol.

---

# 47. Một câu mô tả proposed method hoàn chỉnh

Nếu viết abstract/method overview, phiên bản an toàn là:

> **We propose RASP-SFOD, a reliability-aware structured pruning framework for source-free object detection that integrates target-domain compression directly into Mean-Teacher adaptation. RASP estimates structured channel sensitivity from target-domain SFOD gradients, identifies redundant channel populations using probabilistic mixture modeling, ranks pruning candidates according to information loss relative to computational saving, automatically selects a compression frontier using knee-point detection, and activates pruning only when Teacher pseudo-labels are sufficiently reliable. The Student is progressively masked during adaptation while the dense Teacher remains shape-compatible through EMA, allowing continued RT-SFOD and MARD optimization to recover pruning-induced degradation. Physical graph compaction is performed only after adaptation to obtain actual reductions in parameters, FLOPs, model size, and inference latency.**

---

# 48. Nếu nói cực ngắn để giải thích cho giáo viên/reviewer

Bạn có thể nói:

> “RT-SFOD giúp YOLO26-M thích nghi từ Clear Cityscapes sang Foggy Cityscapes mà không dùng target labels. Tôi mở rộng framework này bằng cách nén Student ngay trong quá trình adaptation. Thay vì đặt trước pruning ratio, tôi dùng gradient trên target domain để đo channel importance, GMM để xác định nhóm redundant, compute cost để ưu tiên channel mang lại saving tốt, Kneedle để tự chọn mức prune, và Teacher confidence để quyết định thời điểm prune. Sau mỗi pruning step, Student tiếp tục RT-SFOD + MARD để recover. Teacher vẫn dense để giữ pseudo-label ổn định, và chỉ cuối cùng mới compact graph thật để đo Params/FLOPs/FPS.”

---

# 49. Cách tôi muốn bạn hình dung RASP

Không nên hình dung nó là:

```text
RT-SFOD
+
pruning algorithm
```

mà nên hình dung là một **compression controller nằm bên cạnh Student**:

```text
                  ┌─────────────────────┐
                  │    Dense Teacher    │
                  │   EMA parameters    │
                  └──────────┬──────────┘
                             │
                         DHF pseudo
                             │
                             ▼
Target image ───────→ ┌───────────────┐
                      │    Student    │
                      │   YOLO26-M    │
                      └───────┬───────┘
                              │
                       SFOD + MARD loss
                              │
                              ▼
                    ┌───────────────────┐
                    │ RASP Controller   │
                    │                   │
                    │ Taylor importance │
                    │       ↓           │
                    │ GMM redundancy    │
                    │       ↓           │
                    │ Cost ranking      │
                    │       ↓           │
                    │ Kneedle budget    │
                    │       ↓           │
Teacher reliability ─→ Reliability gate│
                    │       ↓           │
                    │ Channel masks     │
                    └─────────┬─────────┘
                              │
                              ▼
                      compressed Student
                              │
                       continued recovery
                              │
                              ▼
                       compact export
```