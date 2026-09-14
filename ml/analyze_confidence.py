"""Threshold-sweep analysis over the v1 weak spots (free, no API calls).

Loads the classifier directly, computes raw synthetic-probability for every
chunk of the weak generator types (attack_A10, A13, A18) plus the human
clips, then sweeps the decision threshold to find the best targeted fix.

Usage:  python ml/analyze_confidence.py [--chunk-seconds 3.0]
Output: prints per-group probability distributions + accuracy at each
        threshold; saves ml/threshold_sweep.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import classifier, chunking  # noqa: E402
from app.config import HELDOUT_AUDIO_DIR, HELDOUT_MANIFEST_PATH  # noqa: E402

WEAK_TYPES = ("attack_A10", "attack_A13", "attack_A18")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-seconds", type=float, default=3.0)
    args = ap.parse_args()

    clips = json.loads(HELDOUT_MANIFEST_PATH.read_text(encoding="utf-8")).get("clips", [])
    groups: dict[str, list[float]] = {}
    t0 = __import__("time").perf_counter()

    classifier.load()

    for clip in clips:
        gen = clip.get("generator_type", "unknown")
        if gen in WEAK_TYPES or gen == "human":
            groups.setdefault(gen, [])

    for clip in clips:
        gen = clip.get("generator_type", "unknown")
        if gen not in groups:
            continue
        audio_path = HELDOUT_AUDIO_DIR / clip["filename"]
        if not audio_path.exists():
            continue
        wav_np, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np.T)
        n = max(1, int(float(clip.get("duration_s", 3.0)) // args.chunk_seconds))
        for i in range(n):
            win = chunking.slice_waveform(wav, sr, i * args.chunk_seconds, args.chunk_seconds)
            if win.shape[-1] == 0:
                continue
            is_synth, conf = classifier.classify(win, sr)
            # classifier returns confidence = max(p_synth, 1 - p_synth); the
            # is_synthetic flag tells us which side we are on, so recover the
            # signed synthetic probability correctly even for misclassifications
            p_synth = conf if is_synth else 1.0 - conf
            groups[gen].append(round(p_synth, 4))

    print(f"Analyzed {sum(len(v) for v in groups.values())} chunks in "
          f"{__import__('time').perf_counter() - t0:.0f}s\n")

    sweep = {}
    for thr in [round(x, 2) for x in np.arange(0.10, 0.91, 0.02)]:
        correct = total = 0
        for gen, ps in groups.items():
            truth = 1 if gen.startswith("attack") else 0
            for p in ps:
                total += 1
                if (1 if p >= thr else 0) == truth:
                    correct += 1
        sweep[thr] = round(correct / total, 4) if total else 0.0

    best_thr = max(sweep, key=sweep.get)
    print("Synthetic-probability distributions (mean / min / max, n):")
    for gen, ps in sorted(groups.items()):
        if ps:
            a = np.array(ps)
            print(f"  {gen:12s} n={len(a):3d}  mean={a.mean():.3f}  min={a.min():.3f}  "
                  f"max={a.max():.3f}  acc@0.5={float(((a >= .5) == (gen.startswith('attack'))).mean()):.3f}")

    print(f"\nThreshold sweep (overall accuracy):")
    for thr, acc in sweep.items():
        marker = "  <-- best" if thr == best_thr else ""
        if 0.25 <= thr <= 0.65:
            print(f"  thr={thr:.2f}  acc={acc:.4f}{marker}")

    print(f"\nBest threshold: {best_thr} (accuracy {sweep[best_thr]:.4f} vs "
          f"0.5 -> {sweep.get(0.5, 0.0):.4f})")

    out = ROOT / "ml" / "threshold_sweep.json"
    out.write_text(json.dumps({
        "groups": {g: ps for g, ps in sorted(groups.items())},
        "sweep": {str(k): v for k, v in sweep.items()},
        "best_threshold": best_thr,
    }, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
