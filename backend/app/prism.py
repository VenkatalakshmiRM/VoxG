"""PRISM (PRISMtrace) emission: one trace per chunk classification.

Credentials come from environment variables only. Emission is fire-and-forget:
a PRISM outage or missing credentials must never break classification.
"""
from __future__ import annotations

import logging
from typing import Any

from .config import PRISM_CONFIGURED, AGENT_ID

logger = logging.getLogger("voxg.prism")

_client = None
_init_attempted = False


def _get_client():
    global _client, _init_attempted
    if _client is not None or _init_attempted:
        return _client
    _init_attempted = True
    if not PRISM_CONFIGURED:
        logger.warning(
            "PRISM not configured (missing PRISMTRACE_PROJECT_ID / PRISMTRACE_API_KEY). "
            "Traces will be skipped; set them in .env to enable."
        )
        return None
    try:
        from prismtrace import PRISMtrace

        from .config import PRISMTRACE_API_KEY, PRISMTRACE_HOST, PRISMTRACE_PROJECT_ID

        _client = PRISMtrace(
            api_key=PRISMTRACE_API_KEY,
            project_id=PRISMTRACE_PROJECT_ID,
            host=PRISMTRACE_HOST,
        )
        logger.info("PRISMtrace client initialized (host=%s)", PRISMTRACE_HOST)
    except Exception:
        logger.exception("Failed to initialize PRISMtrace client; traces disabled")
        _client = None
    return _client


def emit_trace(
    chunk_id: str,
    prediction: str,
    confidence: float,
    latency_ms: int,
    session_id: str,
    metadata: dict[str, Any],
    misclassified: bool = False,
) -> None:
    """Emit one classification trace to PRISM. Never raises.

    Misclassified chunks carry an explicit MISCLASSIFIED marker in the output
    text so PRISM's status classifier flags them — giving the RCA failure
    clustering real signal instead of a sea of healthy traces.
    """
    client = _get_client()
    if client is None:
        return
    try:
        output = f"prediction={prediction}, confidence={confidence:.2f}"
        if misclassified:
            output = f"MISCLASSIFIED: {output}"
        client.trace_llm(
            model="wav2vec2-asvspoof-classifier",
            input_messages=[
                {
                    "role": "user",
                    "content": (
                        f"audio_chunk_id={chunk_id}, "
                        f"duration={metadata.get('chunk_length_s')}s, "
                        f"source_clip={metadata.get('source_clip')}"
                    ),
                }
            ],
            output=output,
            latency_ms=latency_ms,
            session_id=session_id,
            agent_id=AGENT_ID,
            metadata=metadata,
        )
    except Exception:
        logger.exception("PRISM trace emission failed for %s (non-fatal)", chunk_id)
