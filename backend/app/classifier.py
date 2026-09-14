"""Real wav2vec2-based audio deepfake classifier (ASVspoof-trained).

Loads the pretrained checkpoint once at startup and classifies 2-4s waveform
windows. Falls back to facebook/wav2vec2-base if the ASVspoof checkpoint is
unavailable (label mapping then comes from fine-tuning, see ml/finetune.py).
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from .config import DECISION_THRESHOLD, MODEL_ID

logger = logging.getLogger("voxg.classifier")

FALLBACK_MODEL_ID = "facebook/wav2vec2-base"

_model = None
_extractor = None
_synthetic_index: int | None = None
_device = "cpu"


def _resolve_synthetic_index(model) -> int | None:
    """Find which class index means 'synthetic/spoofed' from the model config."""
    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        lbl = str(label).lower()
        if any(word in lbl for word in ("synthetic", "spoof", "fake")):
            return int(idx)
    # Single-index convention used by some checkpoints: 1 = spoof
    if len(id2label) == 2:
        return 1
    return None


def load() -> None:
    """Load model + feature extractor once at startup. Never raises fatally."""
    global _model, _extractor, _synthetic_index, _device
    if _model is not None:
        return
    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        logger.info("Loading audio classifier: %s", MODEL_ID)
        try:
            _extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
            _model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
            _synthetic_index = _resolve_synthetic_index(_model)
            logger.info(
                "Loaded %s (labels=%s, synthetic_index=%s)",
                MODEL_ID, _model.config.id2label, _synthetic_index,
            )
        except Exception:
            logger.exception(
                "Primary checkpoint %s failed to load — falling back to %s "
                "(zero-shot quality NOT guaranteed; fine-tune via ml/finetune.py)",
                MODEL_ID, FALLBACK_MODEL_ID,
            )
            _extractor = AutoFeatureExtractor.from_pretrained(FALLBACK_MODEL_ID)
            _model = AutoModelForAudioClassification.from_pretrained(FALLBACK_MODEL_ID)
            _synthetic_index = _resolve_synthetic_index(_model)
        _model.eval()
    except Exception:
        logger.exception("Classifier load failed entirely — classify() will raise")
        _model = None


def classify(window: torch.Tensor, sample_rate: int) -> tuple[bool, float]:
    """Classify one waveform window. Returns (is_synthetic, confidence)."""
    if _model is None or _extractor is None:
        raise RuntimeError("Classifier not loaded — call load() at startup")

    audio = window.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if audio.ndim == 0 or audio.size == 0:
        raise ValueError("Empty audio window")

    inputs = _extractor(
        audio, sampling_rate=sample_rate, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    if _synthetic_index is not None:
        synthetic_prob = float(probs[_synthetic_index])
    else:
        synthetic_prob = float(probs.max())  # degenerate fallback
    # Calibrated decision boundary (config.DECISION_THRESHOLD, env:
    # VOXG_DECISION_THRESHOLD). Lowered from the naive 0.50 after the v1 PRISM
    # diagnosis showed attack_A10 chunks sitting at p_synth 0.40-0.50 while
    # human chunks never exceed 0.153.
    is_synthetic = synthetic_prob >= DECISION_THRESHOLD
    confidence = round(max(synthetic_prob, 1.0 - synthetic_prob), 4)
    return is_synthetic, confidence
