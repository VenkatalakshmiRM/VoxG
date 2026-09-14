"""Frozen Wav2Vec2 feature extraction for individual audio files.

This module deliberately exposes embeddings only.  It does not train, fine-tune,
cache a dataset, or modify the pretrained model.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from transformers import AutoModel, AutoProcessor

from src.audio_preprocessing import TARGET_SAMPLE_RATE, load_and_preprocess_audio


MODEL_IDENTIFIER = "facebook/wav2vec2-base-960h"
"""Standard English Wav2Vec2-base checkpoint (94.4M parameters)."""


class Wav2Vec2FeatureExtractor:
    """Load one frozen pretrained Wav2Vec2 encoder and extract embeddings."""

    def __init__(
        self,
        model_identifier: str = MODEL_IDENTIFIER,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_identifier = model_identifier
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

        self.processor = AutoProcessor.from_pretrained(model_identifier)
        # AutoModel loads Wav2Vec2Model (the encoder), not the CTC prediction head.
        self.model = AutoModel.from_pretrained(model_identifier).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()

    @property
    def hidden_state_dimension(self) -> int:
        """Width of each final encoder hidden-state vector."""
        return int(self.model.config.hidden_size)

    def extract(self, audio_path: str | Path, **preprocessing_kwargs: Any) -> torch.Tensor:
        """Return a finite CPU float32 embedding for one audio file.

        ``preprocessing_kwargs`` are forwarded to ``load_and_preprocess_audio``;
        for example, callers may set ``max_duration_seconds`` for their use case.
        The returned tensor has shape ``(hidden_size,)`` and no computation graph.
        """
        waveform, sample_rate = load_and_preprocess_audio(
            audio_path,
            target_sample_rate=TARGET_SAMPLE_RATE,
            **preprocessing_kwargs,
        )
        if sample_rate != TARGET_SAMPLE_RATE:  # Defensive contract check.
            raise RuntimeError(f"Expected {TARGET_SAMPLE_RATE} Hz preprocessing output, got {sample_rate} Hz.")

        # The processor accepts a 1-D NumPy waveform for a single unpadded item.
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        model_inputs = {name: value.to(self.device) for name, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**model_inputs)
            hidden_states = outputs.last_hidden_state
            attention_mask = model_inputs.get("attention_mask")
            if attention_mask is None:
                embedding = hidden_states.mean(dim=1)
            else:
                # Wav2Vec2 subsamples the waveform with convolutional feature layers;
                # convert the input mask to that shorter time axis before pooling.
                feature_mask = self.model._get_feature_vector_attention_mask(
                    hidden_states.shape[1], attention_mask
                ).to(dtype=hidden_states.dtype)
                mask = feature_mask.unsqueeze(-1)
                denominator = mask.sum(dim=1).clamp_min(1.0)
                embedding = (hidden_states * mask).sum(dim=1) / denominator

        embedding = embedding.squeeze(0).to(dtype=torch.float32).cpu().contiguous()
        if embedding.ndim != 1 or embedding.numel() != self.hidden_state_dimension:
            raise RuntimeError(f"Unexpected embedding shape: {tuple(embedding.shape)}.")
        if not torch.isfinite(embedding).all().item():
            raise RuntimeError("Wav2Vec2 produced a non-finite embedding.")
        return embedding


@lru_cache(maxsize=1)
def _default_extractor() -> Wav2Vec2FeatureExtractor:
    """Reuse a single frozen default model within the current Python process."""
    return Wav2Vec2FeatureExtractor()


def extract_wav2vec2_embedding(
    audio_path: str | Path, **preprocessing_kwargs: Any
) -> torch.Tensor:
    """Extract one fixed-size Wav2Vec2 embedding using the default frozen encoder."""
    return _default_extractor().extract(audio_path, **preprocessing_kwargs)


def timed_extract_wav2vec2_embedding(
    audio_path: str | Path, **preprocessing_kwargs: Any
) -> tuple[torch.Tensor, float]:
    """Extract one embedding and return its wall-clock processing time in seconds."""
    started = perf_counter()
    embedding = extract_wav2vec2_embedding(audio_path, **preprocessing_kwargs)
    return embedding, perf_counter() - started
