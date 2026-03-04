"""Download SSv2 dataset from HuggingFace mirror to Lambda.

Usage:
    python src/download_ssv2.py              # Download all 20 parts + labels
    python src/download_ssv2.py --parts 0 1  # Download specific parts only
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

DATA_DIR = Path("data/ssv2")
REPO_ID = "morpheushoc/something-something-v2"


def download_with_hf_hub():
    """Download SSv2 files using huggingface_hub."""
    from huggingface_hub import hf_hub_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download labels and metadata first (small files)
    for fname in ["labels.json", "train.json", "validation.json"]:
        out = DATA_DIR / fname
        if out.exists():
            print(f"  Already exists: {fname}")
            continue
        print(f"  Downloading {fname}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )

    # Download video archives
    for i in range(20):
        fname = f"videos/20bn-something-something-v2-{i:02d}"
        out = DATA_DIR / fname
        if out.exists():
            print(f"  Already exists: {fname}")
            continue
        print(f"  Downloading {fname} ({i+1}/20)...")
        t0 = time.time()
        hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )
        elapsed = time.time() - t0
        size_mb = out.stat().st_size / 1e6 if out.exists() else 0
        print(f"    Done: {size_mb:.0f}MB in {elapsed:.0f}s ({size_mb/elapsed:.0f}MB/s)")


def extract_videos():
    """Extract the concatenated tar archive."""
    video_dir = DATA_DIR / "20bn-something-something-v2"
    if video_dir.exists() and len(list(video_dir.glob("*.webm"))) > 1000:
        print(f"  Videos already extracted: {len(list(video_dir.glob('*.webm')))} files")
        return

    print("\n  Extracting videos (this takes a few minutes)...")
    videos_dir = DATA_DIR / "videos"
    t0 = time.time()
    # Concatenate parts and extract
    cmd = f"cd {DATA_DIR} && cat videos/20bn-something-something-v2-?? | tar zx"
    subprocess.run(cmd, shell=True, check=True)
    elapsed = time.time() - t0

    n_videos = len(list(video_dir.glob("*.webm")))
    print(f"  Extracted {n_videos} videos in {elapsed:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction step")
    args = parser.parse_args()

    print("=" * 60)
    print("DOWNLOADING SSv2 DATASET")
    print("=" * 60)

    download_with_hf_hub()

    if not args.skip_extract:
        extract_videos()

    print("\nDone!")
