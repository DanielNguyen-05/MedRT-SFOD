from __future__ import annotations
import torch
from mask_dhf import InstancePredictions, mask_aware_dual_head_fusion, mask_stability_quality


def square(h, w, x1, y1, x2, y2, p=0.95):
    m = torch.zeros(h, w)
    m[y1:y2, x1:x2] = p
    return m


def main():
    h = w = 32
    # O2O sees object A, misses object B.
    o2o = InstancePredictions(
        boxes=torch.tensor([[2.,2.,12.,12.]]),
        scores=torch.tensor([0.95]),
        classes=torch.tensor([0]),
        masks=torch.stack([square(h,w,2,2,12,12)]),
    )
    # O2M contains duplicate A + missed B.
    o2m = InstancePredictions(
        boxes=torch.tensor([[3.,3.,12.,12.],[20.,18.,29.,29.]]),
        scores=torch.tensor([0.90,0.85]),
        classes=torch.tensor([0,0]),
        masks=torch.stack([square(h,w,3,3,12,12), square(h,w,20,18,29,29)]),
    )
    q = mask_stability_quality(o2m.masks)
    assert torch.all(q > 0.9)
    fused = mask_aware_dual_head_fusion(o2o, o2m, mode="hybrid", tau_no=0.2)
    assert len(fused) == 2, f"expected anchor A + recovered B, got {len(fused)}"
    assert fused.scores.max() >= 0.95
    print("[OK] Mask-DHF: duplicate O2M suppressed, missed instance recovered")

if __name__ == "__main__":
    main()
