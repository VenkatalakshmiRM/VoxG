"""Zero-shot baseline evaluation: score the held-out manifest with the raw
pretrained checkpoint, no fine-tuning. This is the "before" number for the
PRISM before/after story (and the v1 fallback if fine-tuning stalls).

Usage:
    python ml/zero_shot_eval.py [--manifest ml/heldout_manifest.json] [--limit N]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from backend.app.config import MODEL_ID

ROOT = Path(__file__).resolve().parents[1]
TARGET_SR = 16000


def resolve_synthetic_index(model) -> int:
    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        if any(w in str(label).lower() for w in ("synthetic", "spoof", "fake")):
            return int(idx)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "ml" / "heldout_manifest.json"))
    ap.add_argument("--data-dir", default=str(ROOT / "ml" / "data" / "heldout"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk-seconds", type=float, default=3.0)
    args = ap.parse_args()

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest missing: {manifest_path}. Run ml/prepare_data.py first.")
        return 1
    clips = json.loads(manifest_path.read_text())["clips"]
    if args.limit:
        clips = clips[: args.limit]

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).eval()
    synth_idx = resolve_synthetic_index(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    overall = {"total": 0, "correct": 0}

    with torch.no_grad():
        for i, clip in enumerate(clips):
            wav_path = Path(args.data_dir) / f"{clip['id']}.wav"
            if not wav_path.exists():
                continue
            wav, sr = sf.read(str(wav_path), dtype="float32")
            chunk_n = int(TARGET_SR * args.chunk_seconds)
            n_chunks = max(1, len(wav) // chunk_n)
            clip_preds = []
            for ci in range(n_chunks):
                window = wav[ci * chunk_n : (ci + 1) * chunk_n]
                if len(window) < chunk_n // 2:
                    continue
                if len(window) < chunk_n:
                    window = np.pad(window, (0, chunk_n - len(window)))
                inputs = extractor(window, sampling_rate=TARGET_SR, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
                is_synth = bool(probs[synth_idx] >= 0.5)
                clip_preds.append(is_synth)
            if not clip_preds:
                continue
            pred = "synthetic" if sum(clip_preds) >= len(clip_preds) / 2 else "real"
            g = clip["generator_type"]
            for key in ("all", g):
                s = stats[key]
                s["total"] += 1
                overall["total"] += 1 if key == "all" else 0
                if pred == clip["label"]:
                    s["correct"] += 1
                    if key == "all":
                        overall["correct"] += 1
            if (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(clips)}] acc so far: {overall['correct'] / overall['total']:.3f}")

    if overall["total"] == 0:
        print("No clips scored — is ml/data/heldout populated?")
        return 1

    acc = overall["correct"] / overall["total"]
    print(f"\n=== Zero-shot baseline (run_version=v1) ===")
    print(f"Overall accuracy: {acc:.4f} ({overall['correct']}/{overall['total']} clips)")
    print("\nBy generator type (PRISM-style diagnosis preview):")
    for gen, s in sorted(stats.items()):
        if gen == "all":
            continue
        a = s["correct"] / s["total"] if s["total"] else 0.0
        print(f"  {gen}: {a:.3f} ({s['correct']}/{s['total']})")

    out = ROOT / "ml" / "zero_shot_baseline.json"
    out.write_text(json.dumps({
        "model": MODEL_ID, "accuracy": acc,
        "total": overall["total"], "by_generator": {
            g: {"acc": s["correct"] / s["total"], "n": s["total"]}
            for g, s in stats.items() if g != "all"
        },
    }, indent=2))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
