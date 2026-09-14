"""Prepare ASVspoof 2019 LA data: demo clips + held-out evaluation manifest.

Produces:
  backend/clips/*.wav + labels.json     (16 curated demo clips, 8 real / 8
                                         synthetic across >=2 attack types —
                                         for the live UI demo)
  ml/heldout_manifest.json + ml/data/heldout/*.wav
                                        (200 clips, stratified across real +
                                         all attack types, with ground truth —
                                         for the PRISM replay batch)

Source: local ASVspoof2019_LA parquet shards downloaded to
D:/VoxGData/asvspoof via ml/download_asvspoof.py (see that script; the raw
dataset is ~4.4 GB and is kept OFF C: on purpose). The official protocol file
provides per-utterance attack types (A01..A19) for PRISM diagnosis.

Two-pass design so we never decode more than the ~216 clips we keep:
  pass 1: read only the `path`/`label`/`notes` columns -> sample which clips
          to keep (demo + held-out, stratified)
  pass 2: stream parquet row groups and decode just the selected FLAC rows

Usage:
    python ml/prepare_data.py [--src D:/VoxGData/asvspoof] [--seed 42]
"""
from __future__ import annotations

import argparse
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ROOT / "backend" / "clips"
MANIFEST_PATH = ROOT / "ml" / "heldout_manifest.json"
HELDOUT_AUDIO_DIR = ROOT / "ml" / "data" / "heldout"

TARGET_SR = 16000
N_DEMO = 16          # demo clips for the UI (8 real / 8 synthetic)
N_HELDOUT = 200      # held-out batch for PRISM
MAX_CLIP_S = 10.0    # cap clip length

PROTOCOL_REL = "protocols/ASVspoof2019.LA.cm.eval.trl.txt"


def load_protocol(src: Path) -> dict[str, dict]:
    """Parse the official eval protocol: utterance_id -> {label, attack_type}.

    Line format:
      spoof:    SPEAKER_ID UTTERANCE_ID - ATTACK_ID spoof   (5 cols)
      bonafide: SPEAKER_ID UTTERANCE_ID - bonafide          (4 cols)
    The class tag is therefore the LAST field; ATTACK_ID (A01..A19) is field 4
    on attack lines only.
    """
    protocol: dict[str, dict] = {}
    proto_path = src / PROTOCOL_REL
    for line in proto_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        utt_id = parts[1]
        tag = parts[-1].lower()  # "spoof" or "bonafide"
        attack = parts[3] if tag == "spoof" else "bonafide"
        protocol[Path(utt_id).stem] = {
            "label": "real" if tag == "bonafide" else "synthetic",
            "generator_type": "human" if tag == "bonafide" else f"attack_{attack}",
        }
    return protocol


def scan_metadata(src: Path) -> list[dict]:
    """Pass 1: read only metadata columns from every shard (no audio decode)."""
    import pyarrow.parquet as pq

    records: list[dict] = []
    for shard in sorted((src / "data").glob("test-*.parquet")):
        f = pq.ParquetFile(shard)
        for rg in range(f.num_row_groups):
            t = f.read_row_group(rg, columns=["path", "label", "notes"]).to_pylist()
            for row in t:
                utt = Path(row["path"]).stem
                records.append({
                    "utt_id": utt,
                    "file": shard.name,
                    "row_group": rg,
                    "path_col": row["path"],
                    "label_col": int(row["label"]),
                })
        print(f"  scanned {shard.name}: {f.metadata.num_rows} rows")
    return records


def sample_clips(records: list[dict], protocol: dict[str, dict], seed: int):
    """Stratified selection: demo 8 real / 8 synth (>=2 attack types) + 200 held-out."""
    rng = random.Random(seed)
    matched = [r for r in records if r["utt_id"] in protocol]
    if len(matched) < len(records) * 0.9:
        print(f"warning: only {len(matched)}/{len(records)} rows matched the protocol file")
    for r in matched:
        p = protocol[r["utt_id"]]
        r["label"] = p["label"]
        r["generator_type"] = p["generator_type"]

    real = [r for r in matched if r["label"] == "real"]
    synth_by_gen: dict[str, list] = defaultdict(list)
    for r in matched:
        if r["label"] == "synthetic":
            synth_by_gen[r["generator_type"]].append(r)
    gens = sorted(synth_by_gen)
    print(f"pool: {len(real)} real / {len(matched) - len(real)} synthetic "
          f"across {len(gens)} attack types")

    # ---- demo: 8 real + 8 synthetic from the first two attack types ----
    demo = rng.sample(real, min(N_DEMO // 2, len(real)))
    gens_for_demo = gens[:2] if len(gens) >= 2 else gens
    per_gen = max(1, (N_DEMO - len(demo)) // max(len(gens_for_demo), 1))
    for gen in gens_for_demo:
        demo += rng.sample(synth_by_gen[gen], min(per_gen, len(synth_by_gen[gen])))
    demo_ids = {r["utt_id"] for r in demo}

    # ---- held-out: 200 stratified across real + every attack type ----
    remaining_real = [r for r in real if r["utt_id"] not in demo_ids]
    per_gen_heldout = max(1, (N_HELDOUT - 50) // max(len(gens), 1))
    heldout = rng.sample(remaining_real, min(50, len(remaining_real)))
    for gen in gens:
        pool = [r for r in synth_by_gen[gen] if r["utt_id"] not in demo_ids]
        heldout += rng.sample(pool, min(per_gen_heldout, len(pool)))
    rng.shuffle(heldout)
    heldout = heldout[:N_HELDOUT]
    return demo, heldout


def decode_rows(src: Path, wanted: dict[str, list[dict]]):
    """Pass 2: yield (utt_id, waveform_np, sr) for the wanted utt_ids only."""
    import pyarrow.parquet as pq

    remaining = {s: list(v) for s, v in wanted.items()}
    for shard_name, recs in sorted(remaining.items()):
        if not recs:
            continue
        by_rg: dict[int, list[dict]] = defaultdict(list)
        for r in recs:
            by_rg[r["row_group"]].append(r)
        f = pq.ParquetFile(src / "data" / shard_name)
        for rg, want_rows in sorted(by_rg.items()):
            t = f.read_row_group(rg, columns=["path", "audio"]).to_pylist()
            want_ids = {r["utt_id"] for r in want_rows}
            for row in t:
                utt = Path(row["path"]).stem
                if utt in want_ids:
                    wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    yield utt, wav, int(sr)


def write_clip(wav: np.ndarray, sr: int, dest: Path) -> float:
    import torch
    import torchaudio.functional as AF

    t = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
    if sr != TARGET_SR:
        t = AF.resample(t, sr, TARGET_SR)
    max_n = int(TARGET_SR * MAX_CLIP_S)
    if t.shape[-1] > max_n:
        t = t[..., :max_n]
    sf.write(str(dest), t.squeeze(0).numpy(), TARGET_SR)
    return round(t.shape[-1] / TARGET_SR, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/VoxGData/asvspoof")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    if not (src / PROTOCOL_REL).exists():
        print(f"Protocol file missing under {src}. "
              f"Run: python ml/download_asvspoof.py --dest {src}")
        return 1

    print("Parsing protocol file…")
    protocol = load_protocol(src)
    print(f"  {len(protocol)} protocol entries")

    print("Pass 1: scanning parquet metadata…")
    records = scan_metadata(src)

    demo, heldout = sample_clips(records, protocol, args.seed)
    print(f"selected: {len(demo)} demo clips, {len(heldout)} held-out clips")

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    HELDOUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    wanted: dict[str, list[dict]] = defaultdict(list)
    for r in demo + heldout:
        wanted[r["file"]].append(r)
    by_utt = {r["utt_id"]: r for r in demo + heldout}

    print("Pass 2: decoding selected clips…")
    n_done = 0
    demo_labels, manifest = [], []
    for utt, wav, sr in decode_rows(src, wanted):
        rec = by_utt[utt]
        cid = f"{'demo' if utt in {r['utt_id'] for r in demo} else 'eval'}_{rec['label']}_{utt[:14]}"
        if utt in {r["utt_id"] for r in demo}:
            dest = CLIPS_DIR / f"{cid}.wav"
            dur = write_clip(wav, sr, dest)
            demo_labels.append({
                "id": cid, "label": rec["label"],
                "generator_type": rec["generator_type"],
                "filename": dest.name, "duration_s": dur, "run_version": "demo",
            })
        else:
            dest = HELDOUT_AUDIO_DIR / f"{cid}.wav"
            dur = write_clip(wav, sr, dest)
            manifest.append({
                "id": cid, "label": rec["label"],
                "generator_type": rec["generator_type"],
                "filename": dest.name, "duration_s": dur, "run_version": "v1",
            })
        n_done += 1
        if n_done % 25 == 0:
            print(f"  decoded {n_done}/{len(demo) + len(heldout)}")

    (CLIPS_DIR / "labels.json").write_text(
        json.dumps({"clips": demo_labels}, indent=2), encoding="utf-8"
    )
    MANIFEST_PATH.write_text(
        json.dumps({"clips": manifest}, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(demo_labels)} demo clips -> {CLIPS_DIR / 'labels.json'}")
    print(f"Wrote {len(manifest)} held-out clips -> {HELDOUT_AUDIO_DIR}")
    print(f"Wrote manifest -> {MANIFEST_PATH}")
    print("\nNext: start the backend, then run ml/replay_batch.py (run_version=v1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
