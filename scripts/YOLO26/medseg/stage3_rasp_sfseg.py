#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
YOLO26_DIR = REPO_ROOT / "scripts" / "YOLO26"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(YOLO26_DIR))


from rasp_pruning import (  # noqa: E402
    RASPController,
    add_rasp_args,
)

from smoke_stage2_mt_seg import (  # noqa: E402
    TargetMTDataset,
    build_pseudo_batch,
    collate,
    list_images,
    resolve_device,
    seed_everything,
    setup_teacher_student,
    update_teacher_ema,
)

from mask_dhf_seg import (  # noqa: E402
    generate_mask_dhf_pseudo_masks,
)

from stage2_dense_sfseg import (  # noqa: E402
    SegmentInputFeatureHook,
    average_confidence,
    compute_mard_loss,
    mard_weight,
)


def write_jsonl(path: Path, row: dict):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(row)
            + "\n"
        )


def save_training_state(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    teacher,
    student,
    optimizer,
    scheduler,
    criterion,
    rasp,
    args,
):
    payload = {
        "format": "RASP-SFSeg-v1",
        "epoch": int(epoch),
        "global_step": int(global_step),

        "teacher_state":
            teacher.state_dict(),

        "student_state":
            student.state_dict(),

        "optimizer_state":
            optimizer.state_dict(),

        "scheduler_state":
            scheduler.state_dict(),

        "criterion_state":
            (
                criterion.state_dict()
                if hasattr(
                    criterion,
                    "state_dict",
                )
                else None
            ),

        "rasp":
            rasp.state_dict(),

        "args":
            vars(args),

        "rng": {
            "torch":
                torch.get_rng_state(),

            "cuda":
                (
                    torch.cuda
                    .get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
        },
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        payload,
        path,
    )


def main(args):
    seed_everything(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    images = list_images(
        Path(
            args.target_images
        ).resolve()
    )

    dataset = TargetMTDataset(
        images,
        args.imgsz,
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
        collate_fn=collate,
        generator=generator,
    )

    (
        teacher,
        student,
        student_wrapper,
        criterion,
    ) = setup_teacher_student(
        args.weights,
        device,
        epochs=args.epochs,
    )

    optimizer = optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
    )

    scheduler = (
        optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01,
        )
    )

    # ------------------------------------------------------------
    # RASP
    # ------------------------------------------------------------

    rasp = RASPController(
        student,
        args,
    )

    if not rasp.enabled:
        raise RuntimeError(
            "This script requires --rasp_enable"
        )

    group_names = list(
        rasp.pruner.groups.keys()
    )

    unsafe = [
        name
        for name in group_names
        if any(
            token in name.lower()
            for token in (
                "proto",
                "one2one",
                "one2many",
                "mask",
            )
        )
    ]

    if unsafe:
        raise RuntimeError(
            "Unsafe segmentation groups discovered: "
            + str(unsafe)
        )

    print("=" * 78)
    print("RASP-SFSEG STAGE-3")
    print("=" * 78)

    print(
        "initial weights :",
        Path(
            args.weights
        ).resolve(),
    )

    print(
        "target images   :",
        len(images),
    )

    print("labels used     : NO")
    print("GT masks used   : NO")

    print(
        "eligible groups :",
        len(group_names),
    )

    print(
        "eligible hidden :",
        sum(
            x.hidden
            for x in
            rasp.pruner.groups.values()
        ),
    )

    print(
        "Proto/head prune: NO"
    )

    print(
        "RASP warmup     :",
        args.rasp_warmup_epochs,
    )

    print(
        "RASP cycle      :",
        args.rasp_cycle_epochs,
    )

    print(
        "RASP rel gate   :",
        args.rasp_reliability_threshold,
    )

    print(
        "RASP GMM BIC    :",
        args.rasp_gmm_bic_gain,
    )

    print(
        "RASP pack       :",
        args.rasp_round_to,
    )

    print(
        "RASP step cap   :",
        args.rasp_max_step_cost_fraction,
    )

    print()

    out_dir = Path(
        args.out_dir
    ).resolve()

    checkpoint_dir = (
        out_dir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        out_dir
        / "rasp_history.jsonl"
    )

    if history_path.exists():
        history_path.unlink()

    hook = SegmentInputFeatureHook(
        student
    )

    global_step = 0

    try:
        for epoch_idx in range(
            args.epochs
        ):
            epoch = epoch_idx + 1

            student.train()

            rasp.begin_epoch()

            start = time.time()

            successful = 0
            skipped = 0

            sums = {
                "loss": 0.0,
                "seg": 0.0,
                "semseg": 0.0,
                "mard": 0.0,
                "lambda": 0.0,
            }

            pseudo_total = 0
            anchors_total = 0
            box_extra_total = 0
            mask_extra_total = 0
            reject_rel_total = 0

            conf_sum = 0.0
            conf_count = 0

            for batch_i, (
                weak,
                strong,
                _paths,
            ) in enumerate(
                loader,
                start=1,
            ):
                weak = weak.to(
                    device,
                    non_blocking=True,
                )

                strong = strong.to(
                    device,
                    non_blocking=True,
                )

                (
                    labels,
                    masks,
                    ps,
                ) = generate_mask_dhf_pseudo_masks(
                    teacher,
                    weak,
                    tau_o2o=args.tau_o2o,
                    tau_o2m=args.tau_o2m,
                    tau_no=args.tau_no,
                    tau_dup=args.tau_dup,
                    mask_threshold=args.mask_thr,
                    stability_low=args.stability_low,
                    stability_high=args.stability_high,
                    reliability_threshold=args.mask_rel_thr,
                    min_mask_pixels=args.min_mask_pixels,
                )

                valid = [
                    i
                    for i, rows
                    in enumerate(labels)
                    if rows.shape[0] > 0
                ]

                if not valid:
                    skipped += 1
                    global_step += 1
                    continue

                strong_valid = strong[
                    valid
                ]

                labels_valid = [
                    labels[i]
                    for i in valid
                ]

                masks_valid = [
                    masks[i]
                    for i in valid
                ]

                n_pseudo = sum(
                    x.shape[0]
                    for x in labels_valid
                )

                avg_conf = (
                    average_confidence(
                        labels_valid
                    )
                )

                conf_sum += (
                    avg_conf
                    * n_pseudo
                )

                conf_count += (
                    n_pseudo
                )

                hook.latest = None

                student_outputs = student(
                    strong_valid
                )

                feats = hook.latest

                if feats is None:
                    raise RuntimeError(
                        "MARD feature hook "
                        "captured no features"
                    )

                pseudo_batch = (
                    build_pseudo_batch(
                        labels_valid,
                        masks_valid,
                        strong_valid.shape,
                    )
                )

                det_vec, loss_items = (
                    criterion(
                        student_outputs,
                        pseudo_batch,
                    )
                )

                sfseg_loss = (
                    det_vec.sum()
                )

                mard_loss, _ = (
                    compute_mard_loss(
                        feats,
                        labels_valid,
                        int(
                            strong_valid
                            .shape[2]
                        ),
                        int(
                            strong_valid
                            .shape[3]
                        ),
                        args,
                    )
                )

                lambda_mard = (
                    mard_weight(
                        args,
                        global_step,
                        len(loader),
                        avg_conf,
                    )
                )

                total_loss = (
                    sfseg_loss
                    + lambda_mard
                    * mard_loss
                )

                if not torch.isfinite(
                    total_loss
                ):
                    raise RuntimeError(
                        "Non-finite total loss"
                    )

                optimizer.zero_grad(
                    set_to_none=True
                )

                # RASP Taylor hooks collect
                # |activation * gradient| here.
                total_loss.backward()

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        student.parameters(),
                        args.grad_clip,
                    )
                )

                if not torch.isfinite(
                    torch.as_tensor(
                        grad_norm
                    )
                ):
                    raise RuntimeError(
                        "Non-finite gradient"
                    )

                optimizer.step()

                global_step += 1
                successful += 1

                pseudo_total += (
                    n_pseudo
                )

                anchors_total += (
                    ps["anchors"]
                )

                box_extra_total += (
                    ps[
                        "box_dhf_extras"
                    ]
                )

                mask_extra_total += (
                    ps[
                        "mask_dhf_extras"
                    ]
                )

                reject_rel_total += (
                    ps[
                        "rejected_reliability"
                    ]
                )

                sums["loss"] += float(
                    total_loss.detach()
                )

                sums["seg"] += float(
                    loss_items[1]
                    .detach()
                )

                sums["semseg"] += float(
                    loss_items[4]
                    .detach()
                )

                sums["mard"] += float(
                    mard_loss.detach()
                )

                sums["lambda"] += float(
                    lambda_mard
                )

                if (
                    batch_i == 1
                    or batch_i
                    % args.print_freq == 0
                    or batch_i == len(loader)
                ):
                    print(
                        f"[E{epoch:02d}] "
                        f"batch={batch_i:03d}/{len(loader):03d} "
                        f"pseudo={n_pseudo} "
                        f"loss={float(total_loss.detach()):.4f} "
                        f"seg={float(loss_items[1]):.4f} "
                        f"semseg={float(loss_items[4]):.4f} "
                        f"mard={float(mard_loss.detach()):.4f} "
                        f"lambda={lambda_mard:.6f} "
                        f"conf={avg_conf:.4f} "
                        f"rasp_s={rasp.pruner.sparsity():.4f} "
                        f"grad_preclip={float(grad_norm):.2f}",
                        flush=True,
                    )

            if successful == 0:
                raise RuntimeError(
                    "Epoch had zero successful batches"
                )

            scheduler.step()

            if hasattr(
                criterion,
                "update",
            ):
                criterion.update()

            # Keep Teacher dense.
            # EMA once per epoch.
            update_teacher_ema(
                teacher,
                student,
                args.ema,
            )

            epoch_reliability = float(
                conf_sum
                / max(
                    conf_count,
                    1,
                )
            )

            # RASP decision happens only
            # at epoch boundary.
            prune_stats = (
                rasp.end_epoch(
                    epoch,
                    epoch_reliability,
                )
            )

            denom = max(
                successful,
                1,
            )

            row = {
                "epoch": epoch,

                "time_sec":
                    time.time() - start,

                "valid_batches":
                    successful,

                "skipped_batches":
                    skipped,

                "pseudo":
                    pseudo_total,

                "anchors":
                    anchors_total,

                "box_extras":
                    box_extra_total,

                "mask_extras":
                    mask_extra_total,

                "mask_rejected":
                    reject_rel_total,

                "avg_pseudo_conf":
                    epoch_reliability,

                "loss":
                    sums["loss"]
                    / denom,

                "seg":
                    sums["seg"]
                    / denom,

                "semseg":
                    sums["semseg"]
                    / denom,

                "mard":
                    sums["mard"]
                    / denom,

                "lambda":
                    sums["lambda"]
                    / denom,

                "lr":
                    optimizer
                    .param_groups[0]["lr"],

                **prune_stats,
            }

            write_jsonl(
                history_path,
                row,
            )

            print()
            print(
                f"[Epoch {epoch:02d}] DONE "
                f"time={row['time_sec']:.1f}s "
                f"valid={successful} "
                f"skipped={skipped} "
                f"pseudo={pseudo_total} "
                f"conf={epoch_reliability:.4f} "
                f"loss={row['loss']:.4f} "
                f"seg={row['seg']:.4f} "
                f"mard={row['mard']:.4f} "
                f"RASP("
                f"status={prune_stats.get('rasp_status','-')}, "
                f"s={prune_stats.get('rasp_hidden_sparsity',0):.4f}, "
                f"new_packs={prune_stats.get('rasp_new_packs',0)}, "
                f"knee={prune_stats.get('rasp_knee_packs',0)}/"
                f"{prune_stats.get('rasp_candidates',0)}, "
                f"saved_macs="
                f"{prune_stats.get('rasp_saved_prunable_macs_frac',0):.4f}"
                f")",
                flush=True,
            )

            save_this_epoch = (
                epoch
                % args.save_interval
                == 0
                or epoch
                == args.epochs
            )

            if save_this_epoch:

                # Remove MARD hook while
                # serializing model object.
                hook.close()
                hook = None

                latent_path = (
                    checkpoint_dir
                    / (
                        f"yolo26s_rasp_sfseg_"
                        f"latent_epoch_{epoch}.pt"
                    )
                )

                student.eval()

                # Save dense latent weights.
                # RASP masks live in the
                # separate training state.
                with rasp.suspended():
                    student_wrapper.model = (
                        student
                    )

                    student_wrapper.save(
                        str(latent_path)
                    )

                state_path = (
                    checkpoint_dir
                    / (
                        f"rasp_sfseg_state_"
                        f"epoch_{epoch}.pt"
                    )
                )

                save_training_state(
                    state_path,
                    epoch=epoch,
                    global_step=global_step,
                    teacher=teacher,
                    student=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    rasp=rasp,
                    args=args,
                )

                latest = (
                    checkpoint_dir
                    / "rasp_sfseg_state_latest.pt"
                )

                save_training_state(
                    latest,
                    epoch=epoch,
                    global_step=global_step,
                    teacher=teacher,
                    student=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    rasp=rasp,
                    args=args,
                )

                print(
                    "[SAVE]",
                    latent_path,
                    flush=True,
                )

                print(
                    "[SAVE]",
                    state_path,
                    flush=True,
                )

                if epoch < args.epochs:
                    student.train()

                    hook = (
                        SegmentInputFeatureHook(
                            student
                        )
                    )

    finally:
        if hook is not None:
            hook.close()

        if (
            rasp.enabled
            and rasp.pruner is not None
        ):
            rasp.pruner.uninstall()

    print()
    print("=" * 78)
    print("[PASS] RASP-SFSEG RUN COMPLETED")
    print("=" * 78)


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--weights",
        required=True,
    )

    ap.add_argument(
        "--target-images",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        required=True,
    )

    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=60,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    ap.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
    )

    ap.add_argument(
        "--ema",
        type=float,
        default=0.999,
    )

    # Mask-DHF
    ap.add_argument(
        "--tau-o2o",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--tau-o2m",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--tau-no",
        type=float,
        default=0.2,
    )

    ap.add_argument(
        "--tau-dup",
        type=float,
        default=0.7,
    )

    ap.add_argument(
        "--mask-thr",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--stability-low",
        type=float,
        default=0.40,
    )

    ap.add_argument(
        "--stability-high",
        type=float,
        default=0.60,
    )

    ap.add_argument(
        "--mask-rel-thr",
        type=float,
        default=0.744898,
    )

    ap.add_argument(
        "--min-mask-pixels",
        type=int,
        default=16,
    )

    # MARD
    ap.add_argument(
        "--mard-lambda0",
        type=float,
        default=0.05,
    )

    ap.add_argument(
        "--mard-lambda-max",
        type=float,
        default=0.2,
    )

    ap.add_argument(
        "--mard-gamma",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--mard-alpha",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--mard-beta",
        type=float,
        default=0.1,
    )

    ap.add_argument(
        "--mard-warmup-epochs",
        type=float,
        default=5.0,
    )

    ap.add_argument(
        "--mard-gate-threshold",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--mard-topk-boxes",
        type=int,
        default=15,
    )

    ap.add_argument(
        "--mard-fg-points",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--mard-bg-points",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--mard-eta",
        type=float,
        default=12.0,
    )

    ap.add_argument(
        "--mard-box-conf",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--device",
        default="0",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=29,
    )

    ap.add_argument(
        "--print-freq",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--save-interval",
        type=int,
        default=10,
    )

    add_rasp_args(ap)

    return ap.parse_args()


if __name__ == "__main__":
    main(
        parse_args()
    )
