# MedRT-SFSeg + DURR — Proposed Method Architecture

> **Repository:** `~/MedRT-SFOD`  
> **Branch:** `seg`  
> **Bài toán:** Real-time Source-Free Medical Image Segmentation  
> **Current task:** Polyp segmentation  
> **Source:** Kvasir-SEG  
> **Target:** CVC-ClinicDB  
> **Base model:** YOLO26-S-Seg (`Segment26`, native O2O + O2M)  
> **Proposed experimental pipeline:** **AdaBN → Mean Teacher → Mask-DHF → DURR → Student Loss + SegMARD-v2**  
> **DURR status:** proposed/experimental. Do not claim an accuracy improvement until the full experiment is evaluated.

> **BraTS2024 extension:** [protocol và lệnh chạy](docs/BraTS2024.md). Giữ MedRT-SFSeg,
> mở rộng DURR theo bốn lớp NETC/SNFH/ET/RC và tạo bốn hướng modality độc lập. Kết quả
> polyp bên dưới là kết quả của bài toán cũ, không được coi là kết quả BraTS.

---

# 1. Mục tiêu

Project nghiên cứu **Source-Free Domain Adaptation (SFDA)** cho segmentation ảnh y khoa:

- source labels chỉ dùng khi train source model;
- adaptation chỉ đọc **unlabeled target images**;
- target GT chỉ được dùng sau training để evaluation / visualization;
- các module adaptation phải training-only để giữ O2O-only inference real-time.

Pipeline tổng quát:

```text
Kvasir labels
    ↓
supervised source training
    ↓
source checkpoint
    ↓
CVC images only
    ↓
AdaBN
    ↓
Mean Teacher
    ↓
Mask-DHF
    ↓
DURR
    ↓
Student native loss + SegMARD
    ↓
adapted Student
    ↓
O2O-only inference
```

---

# 2. YOLO26-S-Seg dual-head architecture

`Segment26` có hai nhánh native:

## O2O — One-to-One
- duplicate-free;
- precision cao;
- được dùng như trusted pseudo anchor;
- là branch inference chính.

## O2M — One-to-Many
- dense hơn;
- recall cao hơn;
- có nhiều predictions quanh cùng instance;
- dùng cho coverage, witnesses và uncertainty.

```text
P3/P4/P5
    │
    ▼
Segment26
 /      \
O2O     O2M
 │       ├─ matched witness
 │       └─ novel coverage candidate
 └─ trusted anchor
```

Instance mask được reconstruct từ mask coefficients và prototype:

\[
L_i(x)=\sum_c a_{ic}\Phi_c(x),\qquad
P_i(x)=\sigma(L_i(x)).
\]

Hard geometry:

\[
M_i(x)=\mathbf 1[P_i(x)\ge0.5].
\]

---

# 3. End-to-end pipeline

```text
RAW Kvasir
   ↓
Prepare YOLO segmentation dataset
   ↓
Source train YOLO26-S-Seg
   ↓
Source validation + freeze
   ↓
RAW CVC
   ↓
Prepare target dataset
   ↓
Source-only CVC baseline
   ↓
AdaBN
   ↓
Freeze AdaBN checkpoint
   ↓
Dual-head audit
   ↓
Mask-DHF audit
   ↓
Mask-stability audit → tau_r
   ↓
Stage-2 Mean Teacher
   ├─ Mask-DHF pseudo geometry
   ├─ DURR pixel routing
   ├─ native SFSeg loss
   └─ SegMARD-v2
   ↓
Student update
   ↓
epoch-level EMA Teacher
   ↓
Final Student + Final EMA Teacher
   ↓
automatic post-training trace
   ↓
official evaluation + speed benchmark
```

---

# 4. Source model

Canonical source checkpoint:

```text
runs/seg/source/frozen/kvasir_yolo26s_seg_source_best.pt
```

Kvasir canonical split:
- train 800;
- val 200;
- seed 29.

Recorded source metrics:

| Metric | Value |
|---|---:|
| Mask mAP50 | 0.901027 |
| Mask mAP50-95 | 0.708971 |
| Dice | 0.839441 |
| IoU | 0.781679 |
| Precision | 0.873094 |
| Sensitivity | 0.835901 |
| Specificity | 0.981745 |
| Params | 10.366M |

CVC source-only:

| Metric | Value |
|---|---:|
| mAP50 | 0.781710 |
| mAP50-95 | 0.524527 |
| Dice | 0.727507 |
| IoU | 0.655414 |
| Precision | 0.727876 |
| Sensitivity | 0.789525 |
| Specificity | 0.974837 |

---

# 5. AdaBN

Adaptive Batch Normalization cập nhật target-domain BN statistics bằng target images không nhãn.

Canonical Stage-1 checkpoint:

```text
runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt
```

Recorded CVC AdaBN metrics:

| Metric | Value |
|---|---:|
| mAP50 | 0.790490 |
| mAP50-95 | 0.543404 |
| Dice | 0.743922 |
| IoU | 0.672770 |
| Precision | 0.740146 |
| Sensitivity | 0.808458 |
| Specificity | 0.974264 |

---

# 6. Mean Teacher

Initialize:

\[
\theta_S=\theta_{AdaBN},\qquad
\theta_T=\theta_{AdaBN}.
\]

Teacher nhận weak target view, Student nhận strong view.

Cuối mỗi epoch:

\[
\theta_T\leftarrow
\mu\theta_T+(1-\mu)\theta_S
\]

với:

\[
\mu=0.999.
\]

DURR runner dùng **epoch-level EMA**.

---

# 7. Mask-DHF

Mask-DHF là pseudo-instance constructor nền.

## 7.1 O2O anchor
Giữ nếu:

\[
s_{O2O}\ge\tau_{O2O}=0.5.
\]

O2O quyết định:
- box;
- class;
- hard mask geometry.

O2O không bị reject chỉ vì reliability thấp.

## 7.2 O2M candidate
Giữ nếu:

\[
s_{O2M}\ge0.5.
\]

Valid mask:

\[
|M_i|\ge16.
\]

Same-class box IoU được dùng để chia vai trò:

- matched witness nếu `IoU >= tau_match`;
- novel coverage nếu `IoU <= tau_no = 0.2`.

Coverage extras đi qua class-wise NMS:

\[
\tau_{dup}=0.7.
\]

---

# 8. Mask stability và reliability

Soft mask được threshold hai lần:

\[
M_{lo}=\mathbf1[P\ge0.40],
\qquad
M_{hi}=\mathbf1[P\ge0.60].
\]

Stability:

\[
q_{stab}=IoU(M_{lo},M_{hi}).
\]

Instance reliability:

\[
r=\sqrt{s\,q_{stab}}.
\]

Canonical CVC threshold:

\[
\tau_r=0.744898.
\]

Nhưng method definition là:

\[
\boxed{\tau_r=Q_{0.25}(R_T)}
\]

với \(R_T\) là target O2M reliability distribution từ initial AdaBN Teacher.

Do đó `0.744898`:
- là CVC-specific realization;
- label-free;
- không phải “probability mask đúng”;
- không dùng cho O2O rejection;
- sang domain khác phải recompute Q25.

---

# 9. SegMARD-v2

SegMARD = **Segmentation-Guided Multi-scale Adaptive Representation Diversification**.

Student P3/P4/P5 features được hook trước `Segment26`.

## Regions

\[
R_{FG}=M_i
\]

\[
R_{HBG}=B_i\setminus Dilate(M_i)
\]

\[
R_{EBG}=\Omega\setminus \bigcup B_i.
\]

Final validated V2:
- erode kernel = 1;
- dilate kernel = 1;
- hard-bg ratio = 0.5;
- FG points = 8;
- BG points = 128 → khoảng 64 HBG + 64 EBG.

## Variance loss

\[
L_{var}
=
\frac1C\sum_c
\max\left(
0,\gamma-\sqrt{Var(Z_c)+\epsilon}
\right).
\]

## Covariance loss

\[
L_{cov}
=
\frac{1}{C(C-1)}
\sum_{i\ne j}Cov(\tilde Z)_{ij}^2.
\]

## SegMARD

\[
L_{SegMARD}
=
\sum_{\ell\in\{P3,P4,P5\}}
\alpha L^\ell_{var}
+
\beta L^\ell_{cov}
\]

default:
- \(\gamma=1\);
- \(\alpha=1\);
- \(\beta=0.1\).

Training:

\[
L=L_{SFSeg}+\lambda_M(t)L_{SegMARD}.
\]

Default:
- lambda0 0.05;
- lambda max 0.20;
- warmup 5 epochs;
- confidence gate 0.5.

Validated V2:

| Metric | Value |
|---|---:|
| mAP50 | 0.819903 |
| mAP50-95 | 0.560140 |
| Dice | 0.807439 |
| IoU | 0.729380 |
| Precision | 0.805404 |
| Sensitivity | 0.880193 |
| Specificity | 0.967746 |
| ASD ↓ | 23.733813 px |
| HD95 ↓ | 48.708117 px |

V3 reliability-weighted feature statistics không được chọn vì mAP/sensitivity giảm.

---

# 12. DURR — Dual-head Uncertainty Reliability Routing

DURR là current experimental module.

Core idea:

> Mask-DHF quyết định instance nào đáng tin. DURR quyết định pixel nào cần supervision theo hướng nào. SegMARD điều tiết representation.

```text
Teacher O2O/O2M
      ↓
Mask-DHF
      ↓
hard pseudo geometry
      ↓
DURR
 ┌────┼─────────┐
 │    │         │
signed rescue   safe hallucination
route  pressure suppression
 └────┼─────────┘
      ↓
Student loss
      +
SegMARD-v2
```

DURR training-only, không thêm inference parameters.

---

# 13. DURR-1 — Signed Boundary Routing

Matched O2M:
- same class;
- box IoU ≥ 0.5;
- tối đa 5 witnesses;
- rank bằng `IoU * reliability`.

O2M aggregate:

\[
P_m(x)=\sum_jw_jP_{m,j}(x),
\qquad
w_j\propto s_jq_{stab,j}.
\]

Aggregate O2M reliability:

\[
r_m=\sum_jw_jr_j.
\]

Pair reliability:

\[
r_{pair}=\sqrt{r_or_m}.
\]

Signed disagreement:

\[
\boxed{\Delta(x)=P_m(x)-P_o(x)}.
\]

Interpretation:

\[
\Delta>0
\Rightarrow
\text{expand / increase foreground}
\]

\[
\Delta<0
\Rightarrow
\text{shrink / decrease foreground}.
\]

Two-sided band:

\[
B=
Dilate(M,k)\setminus Erode(M,k)
\]

default \(k=5\).

Active when:

\[
|\Delta|\ge0.02.
\]

Weight:

\[
W_{dir}=|\Delta|r_{pair}.
\]

Directional target:

\[
T_{dir}
=
clip(P_o+g\Delta,0,1)
\]

default gain \(g=1\).

Current loss dùng weighted BCE trên Student differentiable semantic logits:

\[
L_{dir}
=
\frac{
\sum W_{dir}BCEWithLogits(S,T_{dir})
}{
\sum W_{dir}+\epsilon
}.
\]

---

# 16. Total objective

DURR warmup:

\[
\lambda_D(t)
=
\min\left(
1,\frac{step}{warmup\_steps}
\right)
\]

default 5 epochs.

Total:

\[
\boxed{
L_{total}
=
L_{SFSeg}
+
\lambda_ML_{SegMARD}
+
\lambda_D
[
\lambda_{dir}L_{dir}
+
\lambda_{rescue}L_{rescue}
+
\lambda_{hall}L_{hall}
]
}
\]

Defaults:

\[
\lambda_{dir}=0.10,\quad
\lambda_{rescue}=0.20,\quad
\lambda_{hall}=0.10.
\]

---

# 17. Teacher-empty images

Old Stage-2 logic có thể skip image nếu không pseudo labels.

DURR:

```text
Teacher pseudo empty
      ↓
Student auxiliary forward
      ↓
safe hallucination check
      ↓
nếu trigger → optimize L_hall
```

Do đó Teacher-empty images không còn hoàn toàn invisible với optimizer.

---

# 18. Optimization

DURR Stage-2:

- SGD;
- lr = 1e-4;
- momentum = 0.937;
- weight decay = 5e-4;
- Nesterov;
- cosine annealing;
- min lr = 0.01 × initial lr;
- gradient clip = 10;
- epochs = 60;
- batch = 4;
- seed = 29;
- EMA = 0.999 epoch-level.

---

# 19. Checkpoints mới

DURR runner save cả Student và EMA Teacher:

```text
checkpoints/
├── durr_student_epoch_10.pt
├── durr_teacher_ema_epoch_10.pt
├── ...
├── durr_student_epoch_60.pt
└── durr_teacher_ema_epoch_60.pt
```

Điều này cho phép tracking Teacher drift qua các epoch, không còn limitation của các run cũ chỉ save Student.

---

# 20. Automatic Teacher + Student trace

Sau training, runner tự tạo:

```text
<out-dir>/final_analysis/
├── panels/
├── per_image_trace.csv
└── summary.json
```

Nếu có:

```text
--analysis-gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target
```

GT chỉ được load **sau training và checkpoint saving**.

Panel gồm:
1. Image
2. GT
3. Initial Teacher O2O
4. Initial Teacher fused
5. Final EMA Teacher fused
6. Final Teacher O2M witnesses
7. Final signed O2M-O2O delta
8. Final O2M rescue
9. Final Student prediction
10. Absolute Student error

Important deltas:
- teacher drift;
- Student vs initial Teacher;
- Student vs final EMA Teacher.

---

# 21. Deployment

Deployment chỉ dùng:

```text
input
  ↓
adapted Student
  ↓
native O2O inference
  ↓
segmentation
```

Không deploy:
- EMA Teacher;
- O2M routing;
- DURR;
- SegMARD;
- rescue/hallucination logic.

---

# 22. Evaluation

Main metrics:
- Mask mAP50;
- Mask mAP50-95;
- Dice;
- IoU;
- Precision;
- Sensitivity;
- Specificity;
- ASD;
- HD95.

Boundary:
- original GT pixel resolution;
- one-pixel inner surface from 3×3 erosion;
- if one mask empty → image diagonal penalty.

---

# 23. Real-time protocol

Primary:
- RTX 4060 Ti;
- imgsz 640;
- batch 1;
- FP32;
- warmup 100;
- timed 500;
- CUDA synchronize.

Report:
- network forward;
- E2E;
- latency;
- FPS.

Real-time:

\[
FPS\ge30
\quad\text{or}\quad
latency\le33.3ms.
\]

TensorRT FP16 report riêng.

---

# 24. Module status

| Module | Status |
|---|---|
| YOLO26-S-Seg | giữ |
| AdaBN | giữ |
| Mean Teacher | giữ |
| Mask-DHF | giữ |
| SegMARD-v2 | validated |
| SegMARD-v3 reliability weighting | reject |
| CC-DHF | reject |
| BDL-v1 | reject |
| RASP | supplementary / negative |
| DURR-v1 | current experiment |

---

# 25. Current research hypothesis

Hypothesis hiện tại:

> Native O2O/O2M predictions chứa structured target-domain evidence ở cả instance coverage và signed boundary direction. Reliability-aware routing có thể đồng thời sửa missed foreground, boundary expand/shrink và unsupported Student hallucination mà không tăng inference cost.

Chỉ xem DURR là thành công nếu được validate bằng:
- Dice / IoU;
- ASD / HD95;
- sensitivity và specificity;
- ít nhất 3 seeds nếu viết paper;
- nhiều target domains/directions nếu có thể.

---

# 26. Claim discipline

Có thể claim nếu được validate:
- native O2O/O2M signed disagreement exploitation;
- reliability-aware directional routing;
- safe Teacher-empty supervision;
- no inference-time overhead.

Không claim:
- first uncertainty method;
- first boundary-aware polyp segmentation;
- first medical pixel uncertainty;
- DURR improves accuracy trước official experiment.

---

# 27. Main files

```text
scripts/YOLO26/medseg/
├── prepare_kvasir.py
├── train_source_seg.py
├── check_yolo26s_seg.py
├── eval_source_seg.py
├── prepare_cvc_clinicdb.py
├── stage1_adabn_seg.py
├── audit_stage2_seg.py
├── mask_dhf_seg.py
├── audit_mask_dhf_seg.py
├── audit_mask_stability_extras.py
├── segmard_seg.py / segmard_seg_v3.py
├── cc_dhf_seg.py
├── boundary_dhf_seg.py
├── stage2_dense_sfseg_bdl.py
├── durr_seg.py
├── stage2_dense_sfseg_durr.py
├── analyze_seg_results.py
├── benchmark_seg_speed.py
└── trace_teacher_student_masks.py
```

Folders:

```text
runs/seg/source/
runs/seg/stage1/
runs/seg/dense_sfseg/
runs/seg/rasp_sfseg/
runs/seg/analysis/
runs/seg/benchmarks/

logs/seg/source/
logs/seg/stage1/
logs/seg/dense_sfseg/
logs/seg/rasp_sfseg/
logs/seg/benchmarks/
```

---

# 28. Một câu mô tả toàn project

> **MedRT-SFSeg adapts a native dual-head YOLO26 segmentation model without source data by combining target-domain AdaBN, Mean-Teacher self-training, Mask-DHF pseudo-instance selection, DURR reliability-routed signed dual-head supervision, and mask-guided SegMARD representation diversification, while retaining O2O-only inference for real-time deployment.**

---

# 29. Paper-ready Method decomposition

A clean paper Method section can be organized as follows.

## 29.1 Problem formulation

Let the labeled source domain be:

\[
\mathcal D_s=\{(x_i^s,y_i^s)\}_{i=1}^{N_s},
\]

and the unlabeled target domain be:

\[
\mathcal D_t=\{x_i^t\}_{i=1}^{N_t}.
\]

After the source model is trained, source images and source annotations are unavailable during adaptation. The objective is to adapt the source segmentation model to \(\mathcal D_t\) using only target images.

The deployment model remains the adapted YOLO26-S-Seg Student. Teacher, O2M routing, DURR, and SegMARD are training-only.

## 29.2 Target-domain warm start

Adaptive Batch Normalization updates BatchNorm running statistics using unlabeled target images and produces the target-initialized checkpoint \(\theta_0\).

\[
\theta_S^{(0)}=\theta_T^{(0)}=\theta_0.
\]

## 29.3 Mask-aware dual-head pseudo-instance construction

The Teacher produces native O2O and O2M predictions on a weak target view.

O2O detections above \(\tau_o\) form trusted anchors. O2M predictions are separated into:

- matched O2M witnesses for the same O2O instance;
- low-overlap O2M coverage candidates that may recover O2O misses.

For a predicted soft mask \(P_i\), threshold stability is:

\[
q_i^{stab}=
IoU(
\mathbf1[P_i\ge\tau_{lo}],
\mathbf1[P_i\ge\tau_{hi}]
).
\]

Instance reliability is:

\[
r_i=\sqrt{s_iq_i^{stab}}.
\]

The O2M coverage reliability threshold is defined from unlabeled target statistics:

\[
\tau_r=Q_{0.25}(R_T).
\]

For Kvasir→CVC-ClinicDB, the current label-free realization is \(0.744898\). It is not a universal constant and must be recomputed for another target domain.

## 29.4 DURR: Dual-head Uncertainty Reliability Routing

### A. Signed boundary routing

For O2O anchor \(i\), matched O2M witnesses are aggregated:

\[
P_i^m(x)=\sum_j\bar w_{ij}P_{ij}^m(x),
\qquad
\bar w_{ij}
=
\frac{s_{ij}q_{ij}^{stab}}
{\sum_k s_{ik}q_{ik}^{stab}}.
\]

DURR preserves the direction of cross-head disagreement:

\[
\Delta_i(x)=P_i^m(x)-P_i^o(x).
\]

The O2M aggregate reliability is:

\[
r_i^m=\sum_j\bar w_{ij}r_{ij},
\]

and pair reliability is:

\[
r_i^{pair}=\sqrt{r_i^o r_i^m}.
\]

A two-sided contour band is constructed:

\[
\mathcal B_i=
Dilate(M_i,k)\setminus Erode(M_i,k).
\]

Only pixels satisfying:

\[
|\Delta_i(x)|\ge\tau_\Delta
\]

are routed. The routing weight is:

\[
W_i(x)=|\Delta_i(x)|r_i^{pair}.
\]

The target is moved in the signed direction:

\[
T_i^{dir}(x)=
clip(P_i^o(x)+g\Delta_i(x),0,1).
\]

Hence \(\Delta>0\) produces expansion pressure and \(\Delta<0\) produces shrink pressure.

The directional loss is:

\[
\mathcal L_{dir}
=
\frac{
\sum_{i,x}W_i(x)
BCEWithLogits(S(x),T_i^{dir}(x))
}{
\sum_{i,x}W_i(x)+\epsilon
}.
\]

In the implementation, \(S\) is the differentiable single-class semantic logit field from the non-detached O2M Proto26 branch. This is an auxiliary training signal; deployment remains O2O-only.

### B. Reliable O2M rescue

When no O2O anchor exists, accepted O2M coverage instances can receive additional positive persistence supervision if:

\[
s_m\ge\tau_{rescue}^{conf},
\qquad
q_m^{stab}\ge\tau_{rescue}^{stab}.
\]

Optional O2M-to-O2M consensus can be required through mask IoU.

The promoted instance remains an O2M prediction; DURR does not relabel it as a native O2O prediction.

\[
\mathcal L_{rescue}
=
\frac{
\sum_xW_R(x)BCEWithLogits(S(x),1)
}{
\sum_xW_R(x)+\epsilon
}.
\]

### C. Safe hallucination suppression

Teacher-empty is not assumed to mean true background.

A broad, low-threshold dual-head Teacher evidence map is constructed:

\[
E_T(x)=\max(P_{O2O}^{raw}(x),P_{O2M}^{raw}(x)).
\]

Safe background is:

\[
B_{safe}(x)=\mathbf1[E_T(x)<\tau_{safe}].
\]

For a Teacher-empty image, hallucination suppression is activated only when the Student high-confidence foreground area exceeds a threshold. Pixel-level BCE-to-background is then applied only where:

\[
B_{safe}(x)=1
\quad\text{and}\quad
P_S(x)\ge\tau_S.
\]

A weak differentiable excess-area penalty is added to prevent giant foreground masks without treating the whole image as background.

## 29.5 SegMARD-v2

SegMARD uses hard pseudo-mask geometry, not DURR soft routing maps.

For each pseudo instance:

\[
R_{FG}=M_i,
\qquad
R_{HBG}=B_i\setminus M_i,
\qquad
R_{EBG}=\Omega\setminus B_i.
\]

The final validated configuration uses an equal hard/easy background budget (`hard_bg_ratio=0.5`).

For multi-scale Student tokens \(Z_\ell\):

\[
\mathcal L_{var}^{\ell}
=
\frac1C\sum_c
\max
\left(
0,\gamma-\sqrt{Var(Z_{\ell,c})+\epsilon}
\right),
\]

\[
\mathcal L_{cov}^{\ell}
=
\frac{1}{C(C-1)}
\sum_{p\ne q}
Cov(\tilde Z_\ell)_{pq}^{2}.
\]

\[
\mathcal L_{SegMARD}
=
\sum_{\ell\in\{P3,P4,P5\}}
\alpha\mathcal L_{var}^{\ell}
+
\beta\mathcal L_{cov}^{\ell}.
\]

## 29.6 Overall objective

\[
\boxed{
\mathcal L_{total}
=
\mathcal L_{SFSeg}
+
\lambda_M(t)\mathcal L_{SegMARD}
+
\lambda_D(t)
\left[
\lambda_{dir}\mathcal L_{dir}
+
\lambda_{rescue}\mathcal L_{rescue}
+
\lambda_{hall}\mathcal L_{hall}
\right]
}
\]

Default first-run settings:

\[
\lambda_{dir}=0.10,\quad
\lambda_{rescue}=0.20,\quad
\lambda_{hall}=0.10.
\]

Both SegMARD and DURR use five-epoch warm-up/ramp schedules.

## 29.7 Teacher update

The EMA Teacher is updated once per epoch:

\[
\theta_T
\leftarrow
\mu\theta_T+(1-\mu)\theta_S,
\qquad
\mu=0.999.
\]

## 29.8 Deployment

After adaptation:

```text
Target image
    ↓
Adapted YOLO26-S-Seg Student
    ↓
native O2O inference
    ↓
polyp instance mask
```

DURR, O2M witness processing, rescue logic, hallucination suppression, Teacher, and SegMARD are removed from the deployment path.

---

# 30. Recommended ablation sequence

Use the same source checkpoint, target images, target protocol, 60 epochs, and evaluation protocol.

| Variant | Purpose |
|---|---|
| Source-only | domain-shift reference |
| AdaBN | target-statistics warm start |
| Mean Teacher + Mask-DHF | pseudo-label adaptation |
| Mask-DHF + SegMARD-v2 | validated pre-DURR baseline |
| + DURR signed routing only | test directionality |
| + O2M rescue | test O2O-miss persistence |
| + safe hallucination suppression | test Teacher-empty stabilization |
| Full DURR + SegMARD-v2 | proposed complete method |

For a paper, repeat the final important variants across at least three seeds and report mean ± standard deviation.

---

# 31. Evaluation and real-time reporting

Primary segmentation metrics:

- Mask mAP50;
- Mask mAP50-95;
- Dice;
- IoU;
- Precision;
- Sensitivity;
- Specificity;
- ASD;
- HD95.

Efficiency/deployment metrics:

- full training-graph parameters;
- fused deployment parameters;
- checkpoint size;
- deployment GMACs/GFLOPs when profiler support is available;
- network-forward latency/FPS;
- end-to-end in-memory latency/FPS.

Official real-time protocol:

```text
GPU       RTX 4060 Ti
Input     640 × 640
Batch     1
Precision FP32
Warm-up   100
Timed     500
CUDA sync yes
Disk I/O  excluded from E2E timing
```

Use the same hardware, precision, input resolution, batch size, and timing protocol for every compared model.

---

# 32. Claim discipline

DURR is a proposed experimental component until the full evaluation is complete.

Safe wording before validation:

> We propose a reliability-routed signed dual-head supervision mechanism that exploits native O2O/O2M segmentation evidence for directional boundary correction, O2M rescue, and safe suppression of unsupported Student foreground.

Do not claim:

- "first uncertainty method";
- "first boundary-aware polyp segmentation";
- "first pixel-wise uncertainty in medical imaging";
- accuracy improvement before the result is measured.

A stronger novelty statement should only be made after an updated literature search verifies that the specific use of **native signed O2O/O2M disagreement as directional corrective supervision** has not already been reported.
