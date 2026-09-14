"""Environment configuration. Reads PRISM credentials from os.environ only.

Loads a root-level .env (if present) for local development. Never hardcodes
credentials in source; fail fast at startup with a clear message if missing.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the repo-root .env if present (backend is run from backend/).
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")

PRISMTRACE_HOST = os.environ.get("PRISMTRACE_HOST", "https://prism.blockconvey.com")
PRISMTRACE_PROJECT_ID = os.environ.get("PRISMTRACE_PROJECT_ID", "")
PRISMTRACE_API_KEY = os.environ.get("PRISMTRACE_API_KEY", "")

# PRISM credentials are optional for local development: classification works
# without them (traces are skipped), but the evaluation loop requires them.
PRISM_CONFIGURED = bool(PRISMTRACE_PROJECT_ID and PRISMTRACE_API_KEY)

CLIPS_DIR = _ROOT / "backend" / "clips"
LABELS_PATH = CLIPS_DIR / "labels.json"

# ASVspoof-style deepfake detection checkpoints (verified complete on HF Hub:
# model weights + config with id2label + preprocessor_config.json).
# Override with VOXG_MODEL_ID to swap checkpoints without code changes.
MODEL_ID = os.environ.get("VOXG_MODEL_ID", "Bisher/wav2vec2_ASV_deepfake_audio_detection")

AGENT_ID = "voice-scam-classifier"
CHUNK_SECONDS = 3.0


def require_prism_config() -> None:
    """Raise with a clear message if PRISM credentials are missing."""
    if not PRISM_CONFIGURED:
        missing = [
            name
            for name, val in (
                ("PRISMTRACE_PROJECT_ID", PRISMTRACE_PROJECT_ID),
                ("PRISMTRACE_API_KEY", PRISMTRACE_API_KEY),
            )
            if not val
        ]
        raise RuntimeError(
            f"PRISM credentials missing from environment: {', '.join(missing)}. "
            f"Copy .env.example to .env at the repo root and fill in real values."
        )
