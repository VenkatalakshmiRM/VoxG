"""Pydantic request/response models for the VoxG API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ClipInfo(BaseModel):
    id: str
    label: str  # "real" | "synthetic"
    generator_type: str
    duration_s: float
    url: str


class ClipsResponse(BaseModel):
    clips: list[ClipInfo]


class CallSessionResponse(BaseModel):
    session_id: str
    agent_id: str


class ClassifyChunkRequest(BaseModel):
    clip_id: str
    session_id: str
    chunk_index: int = Field(ge=0)
    offset_s: float = Field(ge=0)
    duration_s: float = Field(gt=0, le=10)
    # Tags the PRISM trace for before/after comparison (v1 baseline, v2 fix, …)
    run_version: str = "v1"


class ClassifyChunkResponse(BaseModel):
    chunk_id: str
    is_synthetic: bool
    confidence: float
    latency_ms: int
