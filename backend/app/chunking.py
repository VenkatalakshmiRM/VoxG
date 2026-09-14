"""Waveform slicing into fixed 2-4s windows for chunked inference."""
from __future__ import annotations

import torch


def slice_waveform(
    waveform: torch.Tensor, sample_rate: int, offset_s: float, duration_s: float
) -> torch.Tensor:
    """Return the [offset_s, offset_s + duration_s) slice of a waveform.

    waveform: shape (channels, samples) or (samples,). Returned tensor is
    mono (1, samples). The slice is clamped to the end of the clip; if the
    requested window starts past the end, an empty (1, 0) tensor is returned.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    total_samples = waveform.shape[-1]
    start = int(offset_s * sample_rate)
    end = min(int((offset_s + duration_s) * sample_rate), total_samples)
    start = min(start, total_samples)
    if start >= end:
        return torch.empty(1, 0)
    return waveform[..., start:end]
