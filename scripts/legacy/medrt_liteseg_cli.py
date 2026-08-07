"""Status CLI for the RT-SFOD-Lite / MedRT-SFOD roadmap.

Rewritten: the previous version printed a hardcoded
`"milestones": ["Milestone 1", "Milestone 2", "Milestone 3"]` list that never
checked anything — it would report the exact same "status" whether or not a
single file existed. This version checks each listed config/script against
the filesystem and reports ✅ found / ❌ missing per item, so `--json` output
can actually be used to gate CI or a paper-writing checklist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MILESTONES = {
    "Milestone 1 — Try 1 (lite backbone) + Try 2 (CARD compression)": [
        "ultralytics/nn/modules/lite_block.py",
        "ultralytics/cfg/models/26/yolo26-lite.yaml",
        "scripts/YOLO26/card_compression.py",
        "scripts/YOLO26/stage2_card_rtsfod_yolo26.py",
        "scripts/YOLO26/test_card_lite_smoke.py",
    ],
    "Milestone 2 — Try 3 (instance segmentation groundwork)": [
        "ultralytics/cfg/models/26/yolo26-lite-seg.yaml",
        # DHF mask-IoU extension and a MARD/DHF-aware seg self-training loop
        # are NOT yet implemented — only the architecture exists so far.
    ],
    "Milestone 3 — Try 4 (medical domain: polyp)": [
        "scripts/YOLO26/polyp_kvasir_to_yolo.py",
        "scripts/YOLO26/train_source_supervised.py",
        # Source-free self-training (stage0/stage2) on polyp specifically has
        # not been run/validated against a real Kvasir-SEG download here.
    ],
}


def check_milestone(files: list[str]) -> tuple[list[dict], bool]:
    results = []
    for rel_path in files:
        exists = (REPO_ROOT / rel_path).exists()
        results.append({"path": rel_path, "exists": exists})
    all_exist = all(r["exists"] for r in results)
    return results, all_exist


def build_status_report() -> dict:
    report = {
        "project_name": "RT-SFOD-Lite / MedRT-SFOD",
        "summary": "Efficient source-free detection/segmentation via a lighter backbone and compression-aware self-training, en route to medical-imaging domain shift.",
        "milestones": [],
    }
    for name, files in MILESTONES.items():
        results, complete = check_milestone(files)
        report["milestones"].append({"name": name, "complete": complete, "files": results})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="RT-SFOD-Lite / MedRT-SFOD roadmap status")
    parser.add_argument("--json", action="store_true", help="Print the status report as JSON")
    args = parser.parse_args()

    report = build_status_report()
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"Project: {report['project_name']}")
    print(report["summary"])
    print()
    any_incomplete = False
    for milestone in report["milestones"]:
        status = "✅" if milestone["complete"] else "⚠️ INCOMPLETE"
        print(f"{status}  {milestone['name']}")
        for f in milestone["files"]:
            mark = "✅" if f["exists"] else "❌"
            print(f"    {mark} {f['path']}")
        if not milestone["complete"]:
            any_incomplete = True
    print()
    if any_incomplete:
        print("Some files are missing — see ❌ above. This is expected for milestones not yet reached.")
        sys.exit(1)


if __name__ == "__main__":
    main()
