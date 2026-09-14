"""Prepare ASVspoof 2019 LA data: demo clips + held-out evaluation manifest.

Produces:
  backend/clips/*.wav + labels.json     (10-20 curated demo clips, both classes,
                                         >=2 generator types — for the live UI demo)
  ml/heldout_manifest.json              (100-300 clips with ground truth +
                                         generator_type — for the PRISM batch)

Sources tried in order:
  1. HuggingFace: LanceaKing/asvspoof2019 (via datasets library)
  2. Local directory of ASVspoof files (--from-dir) if download fails

The script resamples to 16kHz mono WAV and writes clip metadata.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ROOT / "backend" / "clips"
MANIFEST_PATH = ROOT / "ml" / "heldout_manifest.json"

TARGET_SR = 16000
N_DEMO = 16          # demo clips for the UI (8 real / 8 synthetic)
N_HELDOUT = 200      # held-out batch for PRISM
MAX_CLIP_S = 10.0    # cap clip length for the demo


def load_via_huggingface() -> list[dict]:
    """Load ASVspoof 2019 LA eval subset via HF datasets. Returns records."""
    from datasets import load_dataset

    print("Downloading ASVspoof 2019 (LA, eval subset) from HuggingFace…")
    ds = load_dataset("LanceaKing/asvspoof2019", split="eval")
    records = []
    for item in ds:
        audio = item["audio"]
        label_raw = str(item.get("label", item.get("attack_type", ""))).lower()
        is_bona_fide = "bonafide" in label_raw or label_raw in ("real", "genuine", "0")
        # generator/attack type: keep the raw label for PRISM diagnosis
        gen = "human" if is_bona_fide else (label_raw or "unknown_synth")
        records.append(
            {
                "waveform": np.asarray(audio["array"], dtype=np.float32),
                "sr": int(audio["sampling_rate"]),
                "label": "real" if is_bona_fide else "synthetic",
                "generator_type": gen,
                "source_id": str(item.get("speaker", item.get("utterance_id", "src"))),
            }
        )
    return records


def load_from_dir(src: Path) -> list[dict]:
    """Load wavs from a local directory. Expects 'bonafide' in real filenames
    or a labels.txt with 'utt_id label' lines."""
    labels = {}
    labels_file = src / "labels.txt"
    if labels_file.exists():
        for line in labels_file.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                labels[parts[0]] = parts[1]
    records = []
    for wav in sorted(src.glob("**/*.wav")):
        stem = wav.stem
        if stem in labels:
            lab = labels[stem]
        elif "bonafide" in stem.lower():
            lab = "real"
        else:
            lab = "synthetic"
        data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        records.append(
            {
                "waveform": data.mean(axis=1),
                "sr": sr,
                "label": lab,
                "generator_type": "human" if lab == "real" else f"synth_{stem[:8]}",
                "source_id": stem,
            }
        )
    return records


def write_clip(rec: dict, dest: Path) -> float:
    """Resample to 16kHz mono, cap length, write WAV. Returns duration."""
    import torch
    import torchaudio.functional as AF

    wav = torch.from_numpy(rec["waveform"]).unsqueeze(0)
    sr = rec["sr"]
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    max_n = int(TARGET_SR * MAX_CLIP_S)
    if wav.shape[-1] > max_n:
        wav = wav[..., :max_n]
    sf.write(str(dest), wav.squeeze(0).numpy(), TARGET_SR)
    return round(wav.shape[-1] / TARGET_SR, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", type=str, default=None,
                    help="Use a local directory of wavs instead of downloading")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    if args.from_dir:
        records = load_from_dir(Path(args.from_dir))
    else:
        try:
            records = load_via_huggingface()
        except Exception as e:
            print(f"HuggingFace download failed: {e}\n"
                  f"Pass --from-dir with local ASVspoof wavs instead.")
            return 1

    if not records:
        print("No records found.")
        return 1

    real = [r for r in records if r["label"] == "real"]
    synth_by_gen: dict[str, list] = defaultdict(list)
    for r in records:
        if r["label"] == "synthetic":
            synth_by_gen[r["generator_type"]].append(r)
    synth_gens = sorted(synth_by_gen)
    print(f"Loaded {len(real)} real / {len(records) - len(real)} synthetic "
          f"across {len(synth_gens)} generator types")

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "ml" / "data").mkdir(parents=True, exist_ok=True)

    # ---- 1. Demo clips for the UI (hidden ground truth in labels.json) ----
    n_per_class = N_DEMO // 2
    demo = random.sample(real, min(n_per_class, len(real)))
    gens_for_demo = synth_gens[:2] if len(synth_gens) >= 2 else synth_gens
    per_gen = max(1, n_per_class // max(len(gens_for_demo), 1))
    for gen in gens_for_demo:
        demo += random.sample(synth_by_gen[gen], min(per_gen, len(synth_by_gen[gen])))

    demo_labels = []
    for rec in demo:
        cid = f"demo_{rec['label']}_{rec['source_id'][:12]}"
        dest = CLIPS_DIR / f"{cid}.wav"
        dur = write_clip(rec, dest)
        demo_labels.append({
            "id": cid, "label": rec["label"], "generator_type": rec["generator_type"],
            "filename": dest.name, "duration_s": dur, "run_version": "demo",
        })
    (CLIPS_DIR / "labels.json").write_text(
        json.dumps({"clips": demo_labels}, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(demo_labels)} demo clips -> {CLIPS_DIR / 'labels.json'}")

    # ---- 2. Held-out manifest for the PRISM batch ----
    remaining_real = [r for r in real if r not in demo]
    heldout_pool = list(remaining_real)
    for gen in synth_gens:
        heldout_pool += synth_by_gen[gen]
    random.shuffle(heldout_pool)
    heldout = heldout_pool[:N_HELDOUT]

    manifest = []
    for rec in heldout:
        cid = f"eval_{rec['label']}_{rec['source_id'][:12]}"
        manifest.append({
            "id": cid, "label": rec["label"], "generator_type": rec["generator_type"],
            "duration_s": min(len(rec["waveform"]) / rec["sr"], MAX_CLIP_S),
        })
    MANIFEST_PATH.write_text(json.dumps({"clips": manifest}, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} held-out entries -> {MANIFEST_PATH}")
    print("\nNext: start the backend, then run ml/replay_batch.py (run_version=v1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
