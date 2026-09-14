"""Time-boxed fine-tuning of the audio classifier on an ASVspoof slice.

Strategy (per PLAN.md):
  - Freeze the SSL encoder, train only the classification head (fast, CPU-viable).
  - Hard wall-clock time budget; results/checkpoint written on every epoch so
    abandoning mid-run loses nothing.

Usage:
    python ml/finetune.py --data-dir ml/data/asvspoof_train --minutes 90
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from backend.app.config import MODEL_ID

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "ml" / "checkpoints"
TARGET_SR = 16000


class ChunkDataset(Dataset):
    """Wavs + binary labels; random 3s crops for augmentation."""

    def __init__(self, items: list[dict], chunk_s: float = 3.0):
        self.items = items
        self.chunk_n = int(TARGET_SR * chunk_s)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        wav = it["waveform"]
        if len(wav) > self.chunk_n:
            start = np.random.randint(0, len(wav) - self.chunk_n)
            wav = wav[start : start + self.chunk_n]
        if len(wav) < self.chunk_n:
            wav = np.pad(wav, (0, self.chunk_n - len(wav)))
        label = 1 if it["label"] == "real" else 0
        return torch.from_numpy(wav.astype(np.float32)), label


def load_items(data_dir: Path, limit: int = 0) -> list[dict]:
    labels = {}
    lf = data_dir / "labels.json"
    if lf.exists():
        for c in json.loads(lf.read_text())["clips"]:
            labels[c["filename"]] = c["label"]
    items = []
    for wav_path in sorted(data_dir.glob("**/*.wav")):
        lab = labels.get(wav_path.name) or (
            "real" if "bonafide" in wav_path.stem.lower() else "synthetic"
        )
        data, sr = sf.read(str(wav_path), dtype="float32")
        items.append({"waveform": data, "label": lab})
    return items[:limit] if limit else items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "ml" / "data" / "asvspoof_train"))
    ap.add_argument("--minutes", type=float, default=90.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    deadline = time.time() + args.minutes * 60
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | time budget: {args.minutes} min")

    items = load_items(Path(args.data_dir))
    if len(items) < 10:
        print(f"Not enough training wavs in {args.data_dir} (found {len(items)}).")
        return 1
    print(f"Training items: {len(items)}")

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(device)

    # Freeze the encoder; train the head only (CPU-friendly).
    for p in model.wav2vec2.parameters():
        p.requires_grad = False

    ds = ChunkDataset(items)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    lossf = nn.CrossEntropyLoss()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.train()
    epoch = 0
    stopped = False
    while not stopped:
        epoch += 1
        running = 0.0
        for step, (wav, y) in enumerate(dl):
            if time.time() > deadline:
                stopped = True
                break
            inputs = extractor(
                wav.numpy(), sampling_rate=TARGET_SR, return_tensors="pt", padding=True
            ).to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(**inputs)
            loss = lossf(out.logits, y)
            loss.backward()
            opt.step()
            running += loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}: loss={loss.item():.4f}")
        print(f"epoch {epoch} done, mean loss={running / max(len(dl), 1):.4f}")
        ckpt = CKPT_DIR / "latest"
        model.save_pretrained(str(ckpt))
        extractor.save_pretrained(str(ckpt))

    print(f"Saved checkpoint -> {CKPT_DIR / 'latest'}")
    print("Use it: set VOXG_MODEL_ID to a local path in .env, or copy into the hub cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
