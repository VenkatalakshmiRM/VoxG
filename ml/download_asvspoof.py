"""Download ASVspoof2019 LA (parquet-native HF repo) to a local directory.

Everything lands under D:/VoxGData/asvspoof (NOT C: — the C: drive is nearly
full). Downloads are resumable per-file: re-running skips completed shards.

Usage:
    python ml/download_asvspoof.py [--dest D:/VoxGData/asvspoof] [--shards 0,1,2]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests

REPO = "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
N_SHARDS = 9
SMALL_FILES = [
    "protocols/ASVspoof2019.LA.cm.eval.trl.txt",
    "README.md",
]
SHARD_TEMPLATE = "data/test-{i:05d}-of-00009.parquet"


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        # Cheap completeness check via Content-Length when available.
        head = requests.head(url, allow_redirects=True, timeout=30)
        expected = int(head.headers.get("Content-Length", "0") or 0)
        if expected and dest.stat().st_size == expected:
            print(f"  [ok] {dest.name} already complete ({dest.stat().st_size / 1e6:.0f} MB)")
            return
        print(f"  [..] {dest.name} incomplete ({dest.stat().st_size / 1e6:.0f} MB), redownloading")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        os.replace(tmp, dest)
    print(f"  [ok] {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="D:/VoxGData/asvspoof")
    ap.add_argument("--shards", default="", help="comma list of shard indices; empty = all")
    args = ap.parse_args()

    dest = Path(args.dest)
    base = f"https://huggingface.co/datasets/{REPO}/resolve/main"

    print(f"Downloading {REPO} -> {dest}")
    for rel in SMALL_FILES:
        download(f"{base}/{rel}", dest / rel)

    shards = (
        [int(s) for s in args.shards.split(",") if s.strip()]
        if args.shards
        else list(range(N_SHARDS))
    )
    for i in shards:
        rel = SHARD_TEMPLATE.format(i=i)
        print(f"Shard {i + 1}/{N_SHARDS}: {rel}")
        download(f"{base}/{rel}", dest / rel)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
