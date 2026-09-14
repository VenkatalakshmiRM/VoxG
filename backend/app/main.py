"""VoxG FastAPI backend: /clips, /call-session, /classify-chunk."""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from . import classifier, chunking, prism
from .config import AGENT_ID, CLIPS_DIR, LABELS_PATH
from .schemas import (
    CallSessionResponse,
    ClassifyChunkRequest,
    ClassifyChunkResponse,
    ClipInfo,
    ClipsResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voxg")

app = FastAPI(title="VoxG — Cloned-Voice Scam Call Detector")

# ground-truth metadata per clip id, loaded from labels.json
_labels: dict[str, dict] = {}


@app.on_event("startup")
def load_model() -> None:
    """Load the audio classifier once at startup (real model, Phase 2)."""
    classifier.load()


@app.on_event("startup")
def load_labels() -> None:
    global _labels
    if LABELS_PATH.exists():
        data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        _labels = {c["id"]: c for c in data.get("clips", [])}
        logger.info("Loaded %d clip labels from %s", len(_labels), LABELS_PATH)
    else:
        logger.warning(
            "No labels.json at %s — run ml/prepare_data.py to curate demo clips",
            LABELS_PATH,
        )


@app.get("/clips", response_model=ClipsResponse)
def list_clips() -> ClipsResponse:
    if not _labels:
        return ClipsResponse(clips=[])
    clips = [
        ClipInfo(
            id=cid,
            label=meta.get("label", "unknown"),
            generator_type=meta.get("generator_type", "unknown"),
            duration_s=float(meta.get("duration_s", 0.0)),
            url=f"/audio/{Path(meta.get('filename', '')).name}",
        )
        for cid, meta in sorted(_labels.items())
    ]
    return ClipsResponse(clips=clips)


@app.post("/call-session", response_model=CallSessionResponse)
def start_call_session() -> CallSessionResponse:
    session_id = f"call-{uuid.uuid4().hex[:8]}"
    logger.info("Started call session %s", session_id)
    return CallSessionResponse(session_id=session_id, agent_id=AGENT_ID)


# Serve clip audio files at /audio/<filename> for frontend playback.
if CLIPS_DIR.exists():
    app.mount("/audio", StaticFiles(directory=str(CLIPS_DIR)), name="audio")


@app.post("/classify-chunk", response_model=ClassifyChunkResponse)
def classify_chunk(req: ClassifyChunkRequest) -> ClassifyChunkResponse:
    meta = _labels.get(req.clip_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Unknown clip_id: {req.clip_id}")

    filename = meta.get("filename", "")
    audio_path = CLIPS_DIR / filename
    if not audio_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Audio file missing: {filename}. Run ml/prepare_data.py"
        )

    t0 = time.perf_counter()
    waveform_np, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform_np.T)  # (channels, samples)
    window = chunking.slice_waveform(waveform, sample_rate, req.offset_s, req.duration_s)
    if window.shape[-1] == 0:
        raise HTTPException(status_code=400, detail="Chunk window is empty (past end of clip)")

    is_synthetic, confidence = classifier.classify(window, sample_rate)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    chunk_id = f"{req.session_id}_chunk{req.chunk_index}"
    prediction = "synthetic" if is_synthetic else "real"
    ground_truth = meta.get("label", "unknown")
    trace_meta = {
        "ground_truth": ground_truth,
        "correct": prediction == ground_truth,
        "generator_type": meta.get("generator_type", "unknown"),
        "chunk_length_s": req.duration_s,
        "source_clip": req.clip_id,
        "run_version": req.run_version or meta.get("run_version", "v1"),
    }
    prism.emit_trace(
        chunk_id=chunk_id,
        prediction=prediction,
        confidence=confidence,
        latency_ms=latency_ms,
        session_id=req.session_id,
        metadata=trace_meta,
    )

    return ClassifyChunkResponse(
        chunk_id=chunk_id,
        is_synthetic=is_synthetic,
        confidence=confidence,
        latency_ms=latency_ms,
    )
