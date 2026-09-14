"""Generate tiny synthetic WAV clips + labels.json for API smoke testing.

These are NOT real speech — they only exist so the chunking/API pipeline can
be exercised end to end before the real ASVspoof demo clips are curated.
Run ml/prepare_data.py later to replace them with real data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

CLIPS_DIR = Path(__file__).resolve().parents[1] / "backend" / "clips"


def make_tone(path: Path, duration_s: float, sr: int = 16000, freq: float = 220.0) -> None:
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
    # fundamental + a couple of harmonics + slight vibrato = speech-ish tone
    sig = (
        0.4 * np.sin(2 * np.pi * freq * t)
        + 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.1 * np.sin(2 * np.pi * freq * 3 * t + 0.5 * np.sin(2 * np.pi * 5 * t))
    )
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t))  # amplitude modulation
    sig = (sig * envelope * 0.8).astype(np.float32)
    sf.write(path, sig, sr)


def main() -> None:
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    clips = [
        {"id": "test_real_01", "label": "real", "generator_type": "human",
         "filename": "test_real_01.wav", "duration_s": 8.0, "freq": 180.0},
        {"id": "test_real_02", "label": "real", "generator_type": "human",
         "filename": "test_real_02.wav", "duration_s": 6.5, "freq": 200.0},
        {"id": "test_synth_01", "label": "synthetic", "generator_type": "tone_gen_a",
         "filename": "test_synth_01.wav", "duration_s": 7.0, "freq": 260.0},
        {"id": "test_synth_02", "label": "synthetic", "generator_type": "tone_gen_b",
         "filename": "test_synth_02.wav", "duration_s": 9.0, "freq": 300.0},
    ]
    for c in clips:
        make_tone(CLIPS_DIR / c["filename"], c["duration_s"], freq=c["freq"])

    labels = {"clips": clips}
    (CLIPS_DIR / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"Wrote {len(clips)} test clips + labels.json to {CLIPS_DIR}")


if __name__ == "__main__":
    main()
