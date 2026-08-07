"""Smoke-list images for a target domain — sanity check that dataset paths
resolve before running any real training.

Rewritten to fix a path-convention mismatch in the previous version, which
hardcoded `REPO_ROOT / "dataset" / ...` (singular, *inside* the repo). Every
other script/doc in this project (README_RTSFOD_LITE.md, the `--data` YAML
convention in stage0/stage2, `polyp_kvasir_to_yolo.py --dst`) puts datasets
*outside* the repo, in a sibling `datasets/` (plural) directory, specifically
so multi-GB data never ends up inside a git checkout or code zip. This
version takes the dataset root as a CLI argument (with that convention as
the default) instead of assuming a fixed internal path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASETS_ROOT = REPO_ROOT.parent / "datasets"  # sibling of the repo, e.g. <workdir>/datasets/

IMG_EXTS = (".png", ".jpg", ".jpeg")


def collect_domain_images(domain: str, datasets_root: Path, limit: Optional[int] = None) -> List[str]:
    """Return a small list of image paths for smoke testing a target domain.

    `domain` selects the expected sub-layout under `datasets_root`:
      - "city"  -> <datasets_root>/foggy_cityscapes/images/{train,val}/
      - "polyp" -> <datasets_root>/polyp_detect/images/{train,val}/  (or polyp_seg)
    These match the layout `polyp_kvasir_to_yolo.py` and the Cityscapes/Foggy
    Cityscapes YOLO-format conversion (see README_RTSFOD_LITE.md, mục 4)
    produce — not the raw upstream dataset layout.
    """
    if domain.lower() == "city":
        root = datasets_root / "foggy_cityscapes" / "images"
    elif domain.lower() == "polyp":
        root = datasets_root / "polyp_detect" / "images"
    else:
        raise ValueError(f"Unsupported domain: {domain} (choices: city, polyp)")

    if not root.exists():
        print(f"[domain_benchmark] WARNING: {root} does not exist — dataset not converted/placed yet?")
        return []

    candidates = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(IMG_EXTS):
                candidates.append(str(Path(dirpath) / name))
    candidates.sort()
    return candidates[:limit] if limit is not None else candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-list images for a target domain")
    parser.add_argument("domain", nargs="?", choices=["city", "polyp", "all"], default="all")
    parser.add_argument(
        "--datasets-root",
        type=str,
        default=str(DEFAULT_DATASETS_ROOT),
        help="Root dir containing per-domain dataset folders (default: sibling 'datasets/' next to the repo)",
    )
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root)
    domains = ["city", "polyp"] if args.domain == "all" else [args.domain]

    for domain in domains:
        images = collect_domain_images(domain, datasets_root, limit=args.limit)
        print(f"{domain}: {len(images)} images found under {datasets_root}")
        for path in images[:3]:
            print(f"  {path}")


if __name__ == "__main__":
    main()
