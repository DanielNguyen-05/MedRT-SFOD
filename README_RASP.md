# RASP-SFOD — Phase 1 implementation

**Reliability-Aware Structured Pruning for Source-Free Object Detection**

This overlay implements the selected project direction without changing the YOLO26 backbone or the original RT-SFOD Mean-Teacher architecture.

## 1. Method in one paragraph

RASP keeps the RT-SFOD **Teacher dense** and initializes the Student from the same Stage-1 checkpoint. During target-only adaptation, the Student preserves dense latent parameters for EMA compatibility but applies structured hidden-channel gates inside standard YOLO Bottleneck blocks. Target-domain first-order Taylor sensitivity is accumulated from the existing detection+MARD loss. Per-block GMMs identify statistically separable low-importance channels; cost-aware ranking favors channels that save more MACs; a knee detector discovers how far the current pruning cycle should go; and DHF confidence gates whether pruning is allowed at all. Accepted masks are monotonic, pruning events are separated by recovery/adaptation epochs, and importance is re-estimated after each structural change. After adaptation, the learned masks are physically compacted into smaller Bottleneck widths and verified numerically against the masked Student.

## 2. What stays unchanged from RT-SFOD

- Stage-1 AdaBN checkpoint initializes **both Teacher and Student**.
- Teacher receives the weak target view.
- Student receives the strong target view.
- Teacher O2O/O2M predictions are fused by DHF.
- Student uses the native YOLO26 end-to-end detection loss.
- MARD remains on P3/P4/P5.
- Teacher is updated once per epoch by EMA from the Student's **dense latent parameters**.
- Target validation labels are not used for main adaptation/model selection.

## 3. What RASP adds

| Question | RASP answer |
|---|---|
| Where can pruning be physically safe? | Bottleneck hidden `cv1.out -> cv2.in`, with optional DepGraph locality audit |
| Which channels should be removed? | Target Taylor importance + per-block GMM low-importance posterior |
| Which blocks get more pruning? | Global cost-aware ranking, not uniform layer sparsity |
| How much should be removed? | Kneedle / max-distance knee on cumulative compute-vs-importance curve |
| When may pruning happen? | Only after warm-up, adequate DHF reliability, and a recovery interval |
| How is accuracy recovered? | Continued RT-SFOD + MARD training between pruning cycles |
| When do Params/FLOPs really decrease? | Only after physical compact export |

## 4. Files

```text
scripts/YOLO26/
├── rasp_pruning.py                    # RASP core algorithm
├── stage2_rasp_rtsfod_yolo26.py       # Mean-Teacher + DHF + MARD + RASP
├── inspect_rasp_prunable_groups.py    # pre-training go/no-go audit
├── export_rasp_compact.py             # physical compaction + equivalence test
├── eval_rasp_student.py               # masked/compact final evaluation
├── test_rasp_smoke.py                 # standalone correctness smoke test
└── run_rasp_ablation.sh               # baseline vs main-method launcher
requirements-rasp.txt
```

The overlay expects the existing repository to retain:

```text
scripts/YOLO26/stage2_rtsfod_yolo26.py
ultralytics/...
```

Do **not** replace the baseline Stage-2 script; it is the control experiment.

## 5. Install dependencies

From the MedRT-SFOD repository root:

```bash
pip install -r requirements-rasp.txt
PYTHONPATH="$PWD" python scripts/YOLO26/test_rasp_smoke.py
```

Expected smoke output contains:

```text
[OK] RASP smoke test
```

## 6. Mandatory go/no-go audit before training

Run on the Stage-1 YOLO26-M checkpoint:

```bash
PYTHONPATH="$PWD" python scripts/YOLO26/inspect_rasp_prunable_groups.py \
  --model /path/to/yolo26m_stage1.pt \
  --imgsz 256 \
  --device 0 \
  --depgraph \
  --out runs/rasp_audit.json
```

Review:

- `eligible_hidden_groups`
- `eligible_hidden_channels`
- `controlled_params`
- `controlled_macs`
- `DepGraph local-safe X/Y`

If the controllable parameter/MAC space is too small, do **not** run a 60-epoch study yet. Expand the safe pruning space first.

## 7. Baseline reproduction

Keep the original script untouched:

```bash
PYTHONPATH="$PWD" python scripts/YOLO26/stage2_rtsfod_yolo26.py \
  --stage1_model /path/to/yolo26m_stage1.pt \
  --data dataset/c2f_yolo/foggy_cityscapes.yaml \
  --out_dir runs/c2f_baseline \
  --imgsz 1024 \
  --batch 4 \
  --device 0
```

Do not pass `--eval` for the main source-free run.

## 8. Main RASP run

```bash
PYTHONPATH="$PWD" python scripts/YOLO26/stage2_rasp_rtsfod_yolo26.py \
  --stage1_model /path/to/yolo26m_stage1.pt \
  --data dataset/c2f_yolo/foggy_cityscapes.yaml \
  --out_dir runs/c2f_rasp \
  --imgsz 1024 \
  --batch 4 \
  --device 0 \
  --rasp_enable
```

Recommended default RASP policy:

```text
warm-up                 5 epochs
recovery cycle          3 epochs minimum
DHF reliability gate    0.50
Taylor EMA beta          0.90
minimum hidden keep     50%
channel pack size        8
GMM low posterior        0.80
GMM BIC gain             10
GMM separation           1.0
max new compute / event  5% of baseline prunable MAC space
```

These values govern *stability*, not a final fixed sparsity. The final pruning ratio is discovered adaptively.

## 9. Resume after Colab interruption

Each save epoch creates:

```text
yolo26_rasp_latent_epoch_N.pt
rasp_training_state_epoch_N.pt
rasp_training_state_latest.pt
```

Resume with:

```bash
PYTHONPATH="$PWD" python scripts/YOLO26/stage2_rasp_rtsfod_yolo26.py \
  --stage1_model /path/to/yolo26m_stage1.pt \
  --data dataset/c2f_yolo/foggy_cityscapes.yaml \
  --out_dir runs/c2f_rasp \
  --imgsz 1024 --batch 4 --device 0 --rasp_enable \
  --resume_state runs/c2f_rasp/checkpoints/rasp_training_state_latest.pt
```

The training state restores Teacher, dense latent Student, optimizer, scheduler, RASP masks, Taylor EMA and pruning-cycle state.

## 10. Physical compact export

Training-time masks do **not** reduce stored parameter count. Export the actual compact Student only after adaptation:

```bash
PYTHONPATH="$PWD" python scripts/YOLO26/export_rasp_compact.py \
  --state runs/c2f_rasp/checkpoints/rasp_training_state_epoch_60.pt \
  --latent_model runs/c2f_rasp/checkpoints/yolo26_rasp_latent_epoch_60.pt \
  --out runs/c2f_rasp/yolo26m_rasp_compact.pt \
  --device 0 \
  --verify \
  --count_macs
```

`--verify` compares the masked dense-latent Student against the physically compact Student. Do not report compact-model efficiency if this equivalence check fails.

## 11. Final target evaluation

Compact model:

```bash
PYTHONPATH="$PWD" python scripts/YOLO26/eval_rasp_student.py \
  --model runs/c2f_rasp/yolo26m_rasp_compact.pt \
  --data dataset/c2f_yolo/foggy_cityscapes.yaml \
  --imgsz 1024 --batch 4 --device 0
```

Masked latent Student, for debugging only:

```bash
PYTHONPATH="$PWD" python scripts/YOLO26/eval_rasp_student.py \
  --model runs/c2f_rasp/checkpoints/yolo26_rasp_latent_epoch_60.pt \
  --state runs/c2f_rasp/checkpoints/rasp_training_state_epoch_60.pt \
  --data dataset/c2f_yolo/foggy_cityscapes.yaml \
  --imgsz 1024 --batch 4 --device 0
```

## 12. Paper experiment order

Minimum experiment sequence:

1. Original RT-SFOD YOLO26-M baseline.
2. RASP main method.
3. Post-adaptation static pruning baseline using the same compactable hidden space.
4. Fixed-budget/importance-only ablation.
5. GMM vs Otsu threshold ablation if needed.
6. Knee budget vs fixed budget ablation.
7. Reliability gate on/off.
8. Only after pruning is stable: Phase-2 QAT.

The key comparison is **post-training pruning vs in-loop adaptive pruning**. It isolates whether continued source-free adaptation can recover pruning-induced damage.

## 13. Important scope limitation

The v1 implementation intentionally limits *physical* pruning to hidden channels of standard Bottleneck units. This is a design choice for correctness: those groups can be compacted exactly without changing P3/P4/P5 or the Detect head, and masked training remains compatible with the dense Teacher EMA. DepGraph is used to audit this structural locality. If the go/no-go audit shows insufficient reduction, the next engineering step is to extend the same RASP selection policy to broader DepGraph groups; do not silently claim arbitrary external-channel masks as real structural compression.

## 14. Future segmentation transfer

For YOLO26 Segment / polyp adaptation, the algorithmic core transfers unchanged:

```text
Dense segmentation Teacher
        ↓
pseudo box/mask supervision
        ↓
Student + RASP hidden pruning
        ↓
continued segmentation SF adaptation
        ↓
physical compact export
```

The first segmentation experiment should keep pruning restricted to the same backbone/neck Bottleneck hidden groups. Only after that is stable should mask/proto-branch pruning be added.
