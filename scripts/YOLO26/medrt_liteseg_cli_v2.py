"""Filesystem status for the revised publication-oriented MedRT-SFOD roadmap."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONES = {
    "M0 — faithful detection baseline": [
        "scripts/YOLO26/stage0_stage1_adabn_rc_yolo26_v2.py",
        "scripts/YOLO26/stage2_rtsfod_yolo26.py",
    ],
    "M1 — reliability-guided compression (detection ablation)": [
        "scripts/YOLO26/card_compression_v2.py",
        "scripts/YOLO26/stage2_card_rtsfod_yolo26_v2.py",
        "scripts/YOLO26/test_card_v2_smoke.py",
    ],
    "M2 — medical instance-seg baseline": [
        "ultralytics/cfg/models/26/yolo26-lite-seg.yaml",
        "scripts/YOLO26/mask_dhf.py",
        "scripts/YOLO26/stage2_medseg_rtsfod_yolo26.py",
        "scripts/YOLO26/test_mask_dhf_smoke.py",
    ],
    "M3 — medical data + source model": [
        "scripts/YOLO26/polyp_kvasir_to_yolo_v2.py",
        "scripts/YOLO26/train_source_supervised.py",
    ],
    "M4 — deployment evidence (NOT done until these exist)": [
        "scripts/YOLO26/export_structured_pruned.py",
        "scripts/YOLO26/export_int8.py",
        "scripts/YOLO26/deployment_benchmark.py",
    ],
}

def report():
    out={"project":"MedRT-SFOD", "milestones":[]}
    for name, files in MILESTONES.items():
        rows=[{"path":p,"exists":(REPO_ROOT/p).exists()} for p in files]
        out["milestones"].append({"name":name,"complete":all(r["exists"] for r in rows),"files":rows})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); args=ap.parse_args(); r=report()
    if args.json: print(json.dumps(r,indent=2)); return
    for m in r['milestones']:
        print(('✅' if m['complete'] else '⚠️'), m['name'])
        for f in m['files']: print('   ', '✅' if f['exists'] else '❌', f['path'])
    if not all(m['complete'] for m in r['milestones']): sys.exit(1)
if __name__=='__main__': main()
