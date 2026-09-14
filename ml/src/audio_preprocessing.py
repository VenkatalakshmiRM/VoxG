"""Deterministic, non-destructive audio preparation for Wav2Vec2 inputs.

The public entry point returns a one-dimensional, float32, CPU tensor sampled
at 16 kHz.  It intentionally performs no feature extraction: Wav2Vec2's
processor will consume this waveform directly in a later phase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torchaudio


TARGET_SAMPLE_RATE = 16_000
"""Sampling rate required by the planned Wav2Vec2 pipeline."""

DEFAULT_MAX_DURATION_SECONDS = 300.0
"""Five-minute safety limit for one input recording; pass ``None`` to disable."""


class AudioPreprocessingError(ValueError):
    """Raised when an audio file cannot safely be prepared for inference."""


PathLike = Union[str, Path]


def _load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Read metadata then samples, translating backend errors into one API error."""
    try:
        metadata = torchaudio.info(str(path))
    except Exception as exc:  # Backend exception types differ by codec/backend.
        raise AudioPreprocessingError(
            f"Unable to read audio metadata for '{path}'. The file may be corrupted "
            "or in an unsupported format."
        ) from exc

    if metadata.sample_rate <= 0 or metadata.num_frames <= 0:
        raise AudioPreprocessingError(f"Audio file '{path}' is empty or has invalid metadata.")

    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except Exception as exc:
        raise AudioPreprocessingError(
            f"Unable to decode audio file '{path}'. The file may be corrupted or unreadable."
        ) from exc

    if sample_rate <= 0 or waveform.numel() == 0 or waveform.size(0) == 0:
        raise AudioPreprocessingError(f"Audio file '{path}' contains no audio samples.")
    return waveform, sample_rate


def load_and_preprocess_audio(
    audio_path: PathLike,
    *,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    max_duration_seconds: float | None = DEFAULT_MAX_DURATION_SECONDS,
    silence_threshold: float = 1e-8,
) -> tuple[torch.Tensor, int]:
    """Load an audio file as a normalized, mono 16-kHz waveform.

    Args:
        audio_path: Path to one supported audio file (for example WAV, FLAC, OGG,
            MP3, where supported by the installed TorchAudio backend).
        target_sample_rate: Required output rate. Defaults to 16,000 Hz.
        max_duration_seconds: Reject longer inputs before decoding them completely.
            Set to ``None`` only when the caller has separately bounded input size.
        silence_threshold: Peak absolute amplitude at or below this is treated as
            silence and left unchanged, preventing divide-by-zero normalization.

    Returns:
        ``(waveform, sample_rate)`` where waveform is a finite, one-dimensional
        ``torch.float32`` CPU tensor and sample_rate equals ``target_sample_rate``.

    Raises:
        AudioPreprocessingError: For missing, unreadable, empty, non-finite, or
            too-long audio, and for invalid configuration values.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise AudioPreprocessingError(f"Audio file does not exist or is not a file: '{path}'.")
    if target_sample_rate <= 0:
        raise AudioPreprocessingError("target_sample_rate must be greater than zero.")
    if silence_threshold < 0:
        raise AudioPreprocessingError("silence_threshold must be non-negative.")
    if max_duration_seconds is not None and max_duration_seconds <= 0:
        raise AudioPreprocessingError("max_duration_seconds must be positive or None.")

    # Check duration from metadata first so unusually large files do not allocate
    # substantial memory. This only reads file metadata; it never changes the file.
    try:
        metadata = torchaudio.info(str(path))
    except Exception as exc:
        raise AudioPreprocessingError(
            f"Unable to read audio metadata for '{path}'. The file may be corrupted "
            "or in an unsupported format."
        ) from exc
    if metadata.sample_rate <= 0 or metadata.num_frames <= 0:
        raise AudioPreprocessingError(f"Audio file '{path}' is empty or has invalid metadata.")
    duration_seconds = metadata.num_frames / metadata.sample_rate
    if max_duration_seconds is not None and duration_seconds > max_duration_seconds:
        raise AudioPreprocessingError(
            f"Audio file '{path}' is {duration_seconds:.2f}s long, exceeding the "
            f"configured {max_duration_seconds:.2f}s maximum."
        )

    waveform, sample_rate = _load_audio(path)
    waveform = waveform.to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(waveform).all().item():
        raise AudioPreprocessingError(f"Audio file '{path}' contains non-finite samples.")

    # TorchAudio layouts are [channels, frames]. Averaging preserves all channels
    # while producing the one-channel form expected by speech processors.
    waveform = waveform.mean(dim=0)
    if waveform.numel() == 0:
        raise AudioPreprocessingError(f"Audio file '{path}' contains no usable frames.")

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)

    peak = waveform.abs().max()
    # Keep silence as zeros instead of dividing by its zero peak. For non-silent
    # audio, peak normalization creates a stable [-1, 1] scale.
    if peak.item() > silence_threshold:
        waveform = waveform / peak

    if not torch.isfinite(waveform).all().item():
        raise AudioPreprocessingError(f"Preprocessing produced non-finite samples for '{path}'.")
    return waveform.contiguous(), target_sample_rate
