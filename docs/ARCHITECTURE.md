# VoxG — Architecture

> Cloned-Voice Scam Call Detector · ForgeAI Hackathon
> This document explains the full system for everyone. If you're new to the domain, start with [How It Works (Plain English)](#how-it-works-plain-english); senior engineers can jump to [System Diagram](#system-diagram) and [Component Reference](#component-reference).

---

## How It Works (Plain English)

Scammers use AI to clone the voice of a family member ("I'm in trouble, send money") or a bank official. A human ear often can't tell the difference anymore. VoxG is a tool that listens to a phone call and tells you, live, how likely it is that the voice is machine-generated.

VoxG works like this:

1. **A fake call plays** on a web page that looks like a phone call screen.
2. **The audio is cut into short 3-second slices** while it plays — because a real product would need to judge a call while it's happening, not after it ends.
3. **An AI model that has studied thousands of real and fake voices** examines each slice and gives a score: how likely is this slice to be machine-made?
4. **The scores roll up into one live indicator**: "Voice authenticity: 82% likely human."
5. **Every judgment is recorded** (what was decided, how confident, what the true answer was) and sent to an observability platform called PRISM, which is like a "black box recorder + detective" for AI systems: it finds patterns in where the model gets fooled (e.g., always wrong on very short clips, or on one particular type of voice generator).

The hackathon's key twist: we don't just show the detector — we **prove it improved**. PRISM diagnoses a weakness, we fix it, and we show the before/after numbers.

---

## System Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                BROWSER                                   │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ React frontend (Vite dev server :5173)                             │  │
│  │                                                                    │  │
│  │  App.jsx ──► clip picker (labels hidden from judges)               │  │
│  │     │                                                              │  │
│  │     ▼                                                              │  │
│  │  CallScreen.jsx ──► "incoming call" UI                             │  │
│  │     │  useCallSession.js                                           │  │
│  │     │    • POST /api/call-session        → session_id              │  │
│  │     │    • every 3s: POST /api/classify-chunk (offset, duration)   │  │
│  │     │    • rolling EMA of P(human) → live confidence bar           │  │
│  │     ▼                                                              │  │
│  │  VerdictScreen.jsx ──► final verdict + SVG trend chart + table     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│        │  <audio> element plays the clip directly from /audio/...       │
└────────┼───────────────────────────────────────────────────────────────-┘
         │  /api/* proxied by Vite  →  http://127.0.0.1:8000
         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND  (uvicorn :8000)                     │
│                                                                          │
│  GET  /clips            → demo clips + metadata (from labels.json)       │
│  POST /call-session     → fresh session_id (groups PRISM traces)         │
│  POST /classify-chunk   → the core loop:                                 │
│       1. soundfile loads backend/clips/<clip>.wav                        │
│       2. chunking.slice_waveform(offset_s, duration_s)                   │
│       3. classifier.classify(window):                                    │
│            Wav2Vec2 feature extractor → model → softmax                  │
│            → (is_synthetic, confidence)                                  │
│       4. prism.emit_trace(...)  — fire-and-forget, never blocks          │
│       → { chunk_id, is_synthetic, confidence, latency_ms }               │
│  GET  /audio/<file>     → static clip audio for browser playback         │
└─────────┬──────────────────────────────────────────────┬─────────────────┘
          │                                              │
          ▼                                              ▼
┌────────────────────────┐                 ┌──────────────────────────────┐
│  ML ARTIFACTS          │                 │  PRISM (PRISMtrace cloud)    │
│  • pretrained wav2vec2 │                 │  • one trace per chunk       │
│    checkpoint (HF Hub) │                 │    with metadata:            │
│  • ml/prepare_data.py  │                 │      ground_truth, correct,  │
│    → ASVspoof 2019 LA  │                 │      generator_type,         │
│  • ml/finetune.py      │                 │      chunk_length_s,         │
│    → time-boxed        │                 │      run_version             │
│  • ml/replay_batch.py  │                 │  • dashboard clusters        │
│    → held-out batch    │                 │    failures by root cause    │
└────────────────────────┘                 └──────────────────────────────┘
```

### Sequence of one simulated call

```
Browser                    Backend                          PRISM
   │  POST /call-session      │                                │
   │─────────────────────────►│  create session_id             │
   │◄───── session_id ────────│                                │
   │                          │                                │
   │  <audio> plays clip      │                                │
   │  every 3s:               │                                │
   │  POST /classify-chunk    │  load wav → slice → infer      │
   │  (offset, duration) ────►│  ──── trace(chunk) ───────────►│
   │◄─ {verdict, confidence} ─│                                │
   │  update live bar         │                                │
   │ ... repeat ...           │                                │
   │  clip ends → endCall     │                                │
   ▼                          │                                │
VerdictScreen renders trend   │        dashboard shows session trajectory
```

---

## Component Reference

### Frontend — `frontend/src/`

| File | Role |
|---|---|
| `App.jsx` | Top-level state machine: clip picker → call → verdict. Hosts the `<audio>` element that "plays" the call. |
| `CallScreen.jsx` | Phone-call look: ringing state with Accept/Decline, then live call with timer and the rolling "Voice authenticity" bar (green ≥50% human, red below). |
| `useCallSession.js` | The engine: session lifecycle (`idle → ringing → active → ended`), a 250ms ticker, chunk cadence (one classify call per 3s of playback), and an EMA of per-chunk P(human) for the rolling indicator. |
| `VerdictScreen.jsx` | Final verdict banner, SVG trend chart of per-chunk P(human) with the 50% threshold line, and a per-chunk table (window, verdict, confidence, latency). |
| `api.js` | Thin fetch wrapper for the three endpoints, with error extraction. |

**Design decision — server-side chunking.** The frontend sends only `clip_id + offset_s + duration_s`; the backend slices the waveform. That keeps the trace metadata exact (ground truth, generator type), avoids uploading audio, and matches how the evaluation batch replays through the *same* endpoint the UI uses — one code path for demo and evaluation.

### Backend — `backend/app/`

| File | Role |
|---|---|
| `main.py` | FastAPI app; the three API endpoints plus `/audio` static mount; per-request orchestration (load → slice → classify → trace → respond). |
| `classifier.py` | Loads the pretrained checkpoint once at startup (`Bisher/wav2vec2_ASV_deepfake_audio_detection`, override with `VOXG_MODEL_ID`); resolves which class index means "synthetic" from the model's own `id2label`; inference = feature extractor → logits → softmax → binary verdict + confidence. |
| `chunking.py` | Pure slicing of a waveform into `[offset, offset+duration)` windows, clamped to clip length, mono-mixed. |
| `prism.py` | PRISMtrace client singleton; one `trace_llm(...)` call per classification with the diagnosis metadata; **fire-and-forget** — any PRISM failure is logged and swallowed so the demo never breaks because of observability. |
| `config.py` | Environment-only credentials (`.env` loaded from repo root); fail-fast messages; paths; `run_version` default. |
| `schemas.py` | Pydantic request/response models for the API contract. |

### ML pipeline — `ml/`

| File | Role |
|---|---|
| `prepare_data.py` | Downloads ASVspoof 2019 LA (HuggingFace; local-dir fallback), writes 16 curated demo clips (both classes, ≥2 generator types) into `backend/clips/` + `labels.json`, and a 200-clip held-out manifest for evaluation. |
| `replay_batch.py` | POSTs every held-out clip, chunk by chunk, through `/classify-chunk` with `run_version=v1` (then `v2` after the fix) so PRISM gets comparable before/after batches. Prints running accuracy. |
| `finetune.py` | Head-only fine-tuning (SSL encoder frozen — CPU-viable), hard wall-clock budget, checkpoint saved every epoch so aborting loses nothing. |
| `zero_shot_eval.py` | Scores the held-out manifest directly with the raw checkpoint — the fallback baseline and the "before" number if fine-tuning stalls. |

### Ops scripts — `scripts/`

| File | Role |
|---|---|
| `verify_prism.sh` | The two-step PRISM gate: credential handshake, then setup-doctor status; exits non-zero unless `live_connected: true`. |
| `export_results.py` | Pulls traces from PRISM's read API, groups by `run_version`, writes the before/after accuracy CSV for pitch slides. |
| `make_test_clips.py` | Generates the 4 synthetic test tones currently in `backend/clips/` so the full pipeline is demoable before real data is prepared. |

---

## The Classifier Model

- **Checkpoint:** `Bisher/wav2vec2_ASV_deepfake_audio_detection` (HuggingFace) — a Wav2Vec 2.0 encoder fine-tuned for fake/real speech detection in the ASVspoof style. Chosen after auditing candidate checkpoints for completeness (weights + `config.json` with `id2label` + `preprocessor_config.json`); this one is base-size, i.e., the fastest on CPU.
- **Why wav2vec2-family:** neural vocoders leave statistical fingerprints (spectral continuity, phase, prosody micro-patterns) that are learnable from labeled real/fake audio but not writable as rules — a self-supervised speech encoder pretrained on massive raw audio captures exactly the features that expose them.
- **Input:** 16kHz mono waveform; 3s windows during calls (`2–4s` per the PRD), whole-clip chunks in evaluation.
- **Output:** softmax over {fake, real}; `is_synthetic = P(fake) ≥ 0.5`; reported confidence = the winning label's probability.
- **Known limits (honest scope):** zero-shot on this demo (fine-tuning on an ASVspoof slice is scripted and time-boxed); synthetic test tones are correctly flagged, but real-world accuracy claims come from the held-out ASVspoof batch, which is exactly what the evaluation loop measures.

---

## Data Flow of One Classification (annotated)

```
client {clip_id, session_id, chunk_index, offset_s, duration_s, run_version}
  → labels.json lookup  → 404 if unknown clip
  → soundfile read      → 404 if file missing
  → slice [offset, offset+3s) → 400 if empty window
  → wav2vec2 inference  → (is_synthetic, confidence)   ~150–700ms CPU
  → PRISM trace (async, non-blocking):
       model:        "wav2vec2-asvspoof-classifier"
       input:        chunk id / duration / source clip
       output:       "prediction=synthetic, confidence=0.77"
       session_id:   groups the call's chunks into one trajectory
       agent_id:     "voice-scam-classifier" (stable across the project)
       metadata:     ground_truth, correct, generator_type,
                     chunk_length_s, run_version, source_clip
  → JSON response {chunk_id, is_synthetic, confidence, latency_ms}
```

The `metadata` block is the heart of the evaluation story: PRISM's Root Cause clustering uses `correct × generator_type` and `correct × chunk_length_s` to surface exactly the weaknesses the pitch claims to fix, and `run_version` separates before/after batches.

---

## Environment & Configuration

| Variable | Purpose | Where |
|---|---|---|
| `PRISMTRACE_HOST` | PRISM ingest host (`https://prism-api-prod.up.railway.app`) | `.env` (gitignored) |
| `PRISMTRACE_PROJECT_ID` | Isolates keys/traces/rules for this project | `.env` |
| `PRISMTRACE_API_KEY` | Ingest-scoped credential (`pt-sk-...`) | `.env` |
| `VOXG_MODEL_ID` | Override the classifier checkpoint | optional, `.env` |

PRISM credentials are optional at runtime: without them the classifier still works and traces are skipped with a warning (useful for offline dev); the evaluation loop requires them.

See also: [TECH_STACK.md](TECH_STACK.md) for why each technology was chosen, and [WORKFLOWS.md](WORKFLOWS.md) for step-by-step operational walkthroughs.
