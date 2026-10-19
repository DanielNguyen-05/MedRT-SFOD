#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402


IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def resolve_device(s: str) -> torch.device:
    if s.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")

    if s.isdigit():
        return torch.device(f"cuda:{s}")

    return torch.device(s)


def make_divisible(x: int, divisor: int = 32) -> int:
    return int(math.ceil(float(x) / divisor) * divisor)


def list_images(root: Path) -> list[Path]:
    imgs = sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMG_EXTS
        ],
        key=lambda p: (
            int(p.stem)
            if p.stem.isdigit()
            else p.stem
        ),
    )

    if not imgs:
        raise RuntimeError(
            f"No target images found: {root}"
        )

    return imgs


class TargetAuditDataset(Dataset):
    """
    Target image-only dataset.

    No labels and no masks are accessible here.
    """

    def __init__(
        self,
        images: list[Path],
        imgsz: int,
    ):
        self.images = images
        self.imgsz = imgsz

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]

        im = cv2.imread(str(path))

        if im is None:
            raise RuntimeError(path)

        im = cv2.cvtColor(
            im,
            cv2.COLOR_BGR2RGB,
        )

        h, w = im.shape[:2]

        scale = self.imgsz / float(max(h, w))

        nh = int(round(h * scale))
        nw = int(round(w * scale))

        im = cv2.resize(
            im,
            (nw, nh),
            interpolation=cv2.INTER_LINEAR,
        )

        tensor = (
            torch.from_numpy(
                np.ascontiguousarray(im)
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        return tensor, str(path)


def collate(batch):
    images, paths = zip(*batch)

    max_h = make_divisible(
        max(x.shape[1] for x in images),
        32,
    )

    max_w = make_divisible(
        max(x.shape[2] for x in images),
        32,
    )

    padded = []

    for x in images:
        c, h, w = x.shape

        canvas = torch.full(
            (c, max_h, max_w),
            114.0 / 255.0,
            dtype=x.dtype,
        )

        canvas[:, :h, :w] = x

        padded.append(canvas)

    return torch.stack(padded), list(paths)


def classwise_nms_full(
    rows: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """
    NMS while preserving mask coefficients.
    """

    if rows.numel() == 0:
        return rows

    kept = []

    classes = rows[:, 5].long().unique()

    for cls in classes:
        idx = torch.where(
            rows[:, 5].long() == cls
        )[0]

        keep = torchvision.ops.nms(
            rows[idx, :4],
            rows[idx, 4],
            iou_threshold,
        )

        kept.append(
            rows[idx[keep]]
        )

    if not kept:
        return rows.new_zeros(
            (0, rows.shape[1])
        )

    return torch.cat(
        kept,
        dim=0,
    )


def dual_head_fusion_seg(
    one2one: torch.Tensor,
    one2many: torch.Tensor,
    tau_o2o: float,
    tau_o2m: float,
    tau_no: float,
    tau_dup: float,
) -> tuple[torch.Tensor, int, int]:
    """
    Box-level DHF while preserving mask coefficients.

    This is only an AUDIT implementation.
    No pseudo-label training occurs here.
    """

    anchors = one2one[
        one2one[:, 4] >= tau_o2o
    ]

    candidates = one2many[
        one2many[:, 4] >= tau_o2m
    ]

    n_anchor = int(anchors.shape[0])

    if candidates.numel() == 0:
        extras = candidates

    elif anchors.numel() == 0:
        extras = classwise_nms_full(
            candidates,
            tau_dup,
        )

    else:
        max_iou = box_iou(
            candidates[:, :4],
            anchors[:, :4],
        ).max(dim=1).values

        extras = candidates[
            max_iou <= tau_no
        ]

        extras = classwise_nms_full(
            extras,
            tau_dup,
        )

    n_extra = int(extras.shape[0])

    if anchors.numel() and extras.numel():
        fused = torch.cat(
            [anchors, extras],
            dim=0,
        )

    elif anchors.numel():
        fused = anchors

    else:
        fused = extras

    if fused.numel():
        fused = fused[
            torch.argsort(
                fused[:, 4],
                descending=True,
            )
        ]

    return fused, n_anchor, n_extra


def main():
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
        "--imgsz",
        type=int,
        default=640,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--device",
        default="0",
    )

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
        "--out",
        default=(
            "runs/seg/dense_sfseg/"
            "stage2_seg_pseudolabel_audit.json"
        ),
    )

    args = ap.parse_args()

    device = resolve_device(
        args.device
    )

    image_root = Path(
        args.target_images
    ).resolve()

    images = list_images(
        image_root
    )

    dataset = TargetAuditDataset(
        images,
        args.imgsz,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
        collate_fn=collate,
    )

    print("=" * 72)
    print("STAGE-2 SEGMENTATION PSEUDO-LABEL AUDIT")
    print("=" * 72)

    print(
        "weights :",
        args.weights,
    )

    print(
        "images  :",
        len(images),
    )

    print(
        "labels loaded : NO"
    )

    print(
        "masks loaded  : NO"
    )

    wrapper = YOLO(
        args.weights
    )

    model = (
        wrapper.model
        .to(device)
        .float()
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    head = model.model[-1]

    print()
    print(
        "model:",
        type(model).__name__,
    )

    print(
        "head :",
        type(head).__name__,
    )

    print(
        "end2end:",
        getattr(model, "end2end", None),
    )

    print(
        "nc:",
        getattr(head, "nc", None),
    )

    print(
        "nm:",
        getattr(head, "nm", None),
    )

    print(
        "npr:",
        getattr(head, "npr", None),
    )

    if "segment" not in type(head).__name__.lower():
        raise RuntimeError(
            "Final head is not a segmentation head"
        )

    if not getattr(model, "end2end", False):
        raise RuntimeError(
            "Model is not end-to-end dual-head"
        )

    if not hasattr(head, "one2one"):
        raise RuntimeError(
            "Missing O2O branch"
        )

    if not hasattr(head, "one2many"):
        raise RuntimeError(
            "Missing O2M branch"
        )

    nm = int(head.nm)

    total_o2o = 0
    total_o2m = 0
    total_anchor = 0
    total_extra = 0
    total_fused = 0

    conf_o2o = []
    conf_o2m = []
    conf_fused = []

    mask_coeff_norms = []

    proto_shapes = set()

    first_batch_printed = False

    with torch.inference_mode():
        for ims, paths in loader:
            ims = ims.to(
                device,
                non_blocking=True,
            )

            outputs = model(
                ims,
                augment=False,
                visualize=False,
            )

            if not (
                isinstance(outputs, tuple)
                and len(outputs) == 2
                and isinstance(outputs[1], dict)
            ):
                raise RuntimeError(
                    "Unexpected Segment26 eval output"
                )

            first, branch_preds = outputs

            if not (
                isinstance(first, tuple)
                and len(first) == 2
            ):
                raise RuntimeError(
                    "Expected ((O2O, proto), branches)"
                )

            final_o2o, proto = first

            if "one2many" not in branch_preds:
                raise RuntimeError(
                    "Missing one2many predictions"
                )

            raw_o2m = branch_preds[
                "one2many"
            ]

            if "mask_coefficient" not in raw_o2m:
                raise RuntimeError(
                    "O2M branch has no mask coefficients"
                )

            if "one2one" not in branch_preds:
                raise RuntimeError(
                    "Missing one2one raw branch"
                )

            if (
                "mask_coefficient"
                not in branch_preds["one2one"]
            ):
                raise RuntimeError(
                    "O2O branch has no mask coefficients"
                )

            proto_shapes.add(
                tuple(proto.shape[1:])
            )

            decoded_o2m = head._inference(
                raw_o2m
            ).permute(0, 2, 1)

            final_o2m = head.postprocess(
                decoded_o2m
            )

            expected_dim = 6 + nm

            if final_o2o.shape[-1] != expected_dim:
                raise RuntimeError(
                    f"O2O dim={final_o2o.shape[-1]}, "
                    f"expected={expected_dim}"
                )

            if final_o2m.shape[-1] != expected_dim:
                raise RuntimeError(
                    f"O2M dim={final_o2m.shape[-1]}, "
                    f"expected={expected_dim}"
                )

            if not first_batch_printed:
                print()
                print("[First batch]")
                print(
                    "input:",
                    tuple(ims.shape),
                )
                print(
                    "O2O:",
                    tuple(final_o2o.shape),
                )
                print(
                    "O2M:",
                    tuple(final_o2m.shape),
                )
                print(
                    "proto:",
                    tuple(proto.shape),
                )
                print(
                    "raw O2M coeff:",
                    tuple(
                        raw_o2m[
                            "mask_coefficient"
                        ].shape
                    ),
                )

                first_batch_printed = True

            for i in range(
                ims.shape[0]
            ):
                o2o = final_o2o[i]

                o2m = final_o2m[i]

                o2o = o2o[
                    o2o[:, 4] > 0
                ]

                o2m = o2m[
                    o2m[:, 4] > 0
                ]

                total_o2o += int(
                    (o2o[:, 4] >= args.tau_o2o)
                    .sum()
                    .item()
                )

                total_o2m += int(
                    (o2m[:, 4] >= args.tau_o2m)
                    .sum()
                    .item()
                )

                if o2o.numel():
                    keep = o2o[:, 4] >= args.tau_o2o

                    if keep.any():
                        conf_o2o.extend(
                            o2o[keep, 4]
                            .detach()
                            .cpu()
                            .tolist()
                        )

                if o2m.numel():
                    keep = o2m[:, 4] >= args.tau_o2m

                    if keep.any():
                        conf_o2m.extend(
                            o2m[keep, 4]
                            .detach()
                            .cpu()
                            .tolist()
                        )

                fused, na, ne = dual_head_fusion_seg(
                    o2o,
                    o2m,
                    args.tau_o2o,
                    args.tau_o2m,
                    args.tau_no,
                    args.tau_dup,
                )

                total_anchor += na
                total_extra += ne
                total_fused += int(
                    fused.shape[0]
                )

                if fused.numel():
                    conf_fused.extend(
                        fused[:, 4]
                        .detach()
                        .cpu()
                        .tolist()
                    )

                    coeff = fused[:, 6:]

                    norms = torch.linalg.vector_norm(
                        coeff,
                        dim=1,
                    )

                    mask_coeff_norms.extend(
                        norms
                        .detach()
                        .cpu()
                        .tolist()
                    )

    def stats(values):
        if not values:
            return {
                "n": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        x = np.asarray(
            values,
            dtype=np.float64,
        )

        return {
            "n": int(len(x)),
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "min": float(x.min()),
            "max": float(x.max()),
        }

    result = {
        "weights": str(
            Path(args.weights).resolve()
        ),
        "target_images": str(image_root),
        "num_images": len(images),
        "labels_used": False,
        "masks_used": False,
        "head": type(head).__name__,
        "end2end": bool(
            getattr(model, "end2end", False)
        ),
        "nc": int(head.nc),
        "nm": nm,
        "npr": int(head.npr),
        "tau_o2o": args.tau_o2o,
        "tau_o2m": args.tau_o2m,
        "tau_no": args.tau_no,
        "tau_dup": args.tau_dup,
        "proto_shapes": [
            list(x)
            for x in sorted(proto_shapes)
        ],
        "o2o_selected": total_o2o,
        "o2m_selected": total_o2m,
        "dhf_anchors": total_anchor,
        "dhf_extras": total_extra,
        "dhf_fused": total_fused,
        "avg_fused_per_image":
            total_fused / len(images),
        "o2o_conf": stats(conf_o2o),
        "o2m_conf": stats(conf_o2m),
        "fused_conf": stats(conf_fused),
        "mask_coeff_norm": stats(
            mask_coeff_norms
        ),
    }

    out = Path(args.out).resolve()

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("AUDIT RESULTS")
    print("=" * 72)

    print(
        "O2O selected :",
        total_o2o,
    )

    print(
        "O2M selected :",
        total_o2m,
    )

    print(
        "DHF anchors  :",
        total_anchor,
    )

    print(
        "DHF extras   :",
        total_extra,
    )

    print(
        "DHF fused    :",
        total_fused,
    )

    print(
        "avg/image    :",
        f"{total_fused / len(images):.4f}",
    )

    print(
        "fused conf   :",
        stats(conf_fused),
    )

    print(
        "mask coeff norm:",
        stats(mask_coeff_norms),
    )

    if total_fused == 0:
        raise RuntimeError(
            "No pseudo labels survived DHF"
        )

    if not mask_coeff_norms:
        raise RuntimeError(
            "No mask coefficients survived DHF"
        )

    print()
    print("[PASS] dual-head segmentation outputs ready")
    print("saved:", out)


if __name__ == "__main__":
    main()
