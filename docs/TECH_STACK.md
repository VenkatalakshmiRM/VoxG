# VoxG — Tech Stack & Rationale

> Every technology in VoxG, what it does, and **why it was chosen** over alternatives. Written so a non-engineer can follow the reasoning and a senior dev can challenge it.

---

## Stack at a Glance

| Layer | Technology | One-line reason |
|---|---|---|
| ML model | Wav2Vec 2.0 (HuggingFace Transformers) | Best open speech-representation encoder; deepfake artifacts are learned patterns, not rules |
| ML checkpoint | `Bisher/wav2vec2_ASV_deepfake_audio_detection` | Verified-complete ASVspoof-style fake/real classifier; base-size = fastest on CPU |
| Inference runtime | PyTorch (CPU) | Zero-config inference at ~150–700ms/3s-chunk; GPU optional, not required |
| Audio I/O | soundfile + NumPy | Reliable WAV decode on Windows where torchaudio's loader needs TorchCodec |
| Dataset | ASVspoof 2019 LA (HuggingFace `datasets`) | THE standard academic benchmark for voice spoofing; free, labeled by attack type |
| Backend | Python + FastAPI + uvicorn | Python is mandatory for the ML stack; FastAPI gives typed endpoints + async + static files in one |
| Frontend | React 19 + Vite | Fastest path to a polished, stateful call-screen UI with live updates |
| Styling | Hand-written CSS | Full control of the "phone call" aesthetic; no framework weight for one screen |
| Observability | PRISM (PRISMtrace SDK ≥0.4.3) | Hackathon's evaluation platform: auto-scores, clusters and diagnoses failures from traces |
| Config/secrets | `.env` + python-dotenv, env-only reads | Keys never touch source control |
| Language tooling | Python 3.14, Node 24 | Whatever the machine has; both fully supported |

---

## Why These Choices (the reasoning)

### 1. Wav2Vec 2.0 for detection — because the problem *requires* learned AI

There is no human-writable rule for "this frequency pattern indicates a neural vocoder rather than a human vocal tract." The tell-tale artifacts of synthetic speech are statistical regularities in the raw signal — over-smooth spectral envelopes, unnatural phase coherence, too-perfect prosody. A self-supervised model like Wav2Vec 2.0, pretrained on enormous amounts of raw audio, learns general speech representations; a small classification head on top learns to separate real from fake. This is the standard approach in the audio-deepfake research literature (most ASVspoof submissions use SSL encoders), which is exactly the defensibility we want: we're standing on published science, not a gimmick.

**Alternatives rejected:**
- *Hand-crafted DSP features + classical ML* — weaker on unseen generators; the pitch would fail the "why AI?" test.
- *Training from scratch* — infeasible in a hackathon window and explicitly out of scope in the PRD.
- *Mel-spectrogram CNNs (e.g., AASIST-style)* — strong published results, but the wav2vec2 route lets us reuse an off-the-shelf checkpoint immediately; spectrogram models would need training before showing anything.

### 2. This specific checkpoint — verified completeness

We audited HuggingFace candidates via the Hub API for all three artifacts (weights, `config.json` with a readable `id2label`, `preprocessor_config.json`). Our first pick (`Sara1708/...`) shipped no feature extractor and silently degraded to a random-head fallback; `Bisher/wav2vec2_ASV_deepfake_audio_detection` is complete and its `id2label` ({0: fake, 1: real}) is resolved at load time rather than assumed. The model id is an env var (`VOXG_MODEL_ID`) so swapping checkpoints (e.g., after fine-tuning) requires zero code changes.

### 3. CPU-only inference — latency budget math

Per-chunk CPU latency measured at ~150–700ms against a 3-second playback cadence: inference comfortably finishes before the next chunk is due, so a GPU is unnecessary for the demo. The plan's only GPU consumer is optional fine-tuning, which is time-boxed and safely skippable (the zero-shot baseline still powers the full PRISM story). This keeps the demo laptop-portable with no CUDA install pain (a 2.9GB CUDA wheel download also proved flaky on this connection — CPU eliminates that failure mode entirely).

### 4. FastAPI backend — one language for API + ML

The ML ecosystem (transformers, torch, datasets, soundfile) is Python-only, so the API server must be Python too. FastAPI was chosen over Flask/Django because: Pydantic models give a self-documenting, validated API contract (`schemas.py`); async endpoints keep the UI snappy while inference runs; `StaticFiles` serves the clip audio for playback in the same process; and uvicorn is a one-command dev server.

**API contract** (3 endpoints, deliberately minimal):

| Endpoint | Method | Purpose |
|---|---|---|
| `/clips` | GET | Demo clip list with metadata (label/generator type kept for the team UI, hidden from judges) |
| `/call-session` | POST | Mints a `session_id` so one call's chunks form one PRISM trajectory |
| `/classify-chunk` | POST | Slice → classify → trace → respond; the single code path used by *both* the live UI and the evaluation replay |

**Server-side chunking** (client sends offsets, server slices) keeps trace metadata exact and lets `replay_batch.py` reuse the identical endpoint — demo and evaluation can never drift apart.

### 5. React + Vite frontend — stateful live UI, fast to build

The demo needs sequenced states (ringing → active → ended), a timer, polling-style periodic requests, and a live-updating indicator — classic React state territory. Vite gives instant dev-server startup and a build in under a second. No UI framework (Tailwind/MUI): one screen, and hand-written CSS gives the exact "phone call" look with zero dependency risk.

Key frontend decisions:
- **EMA rolling confidence** (α=0.6) — smooths per-chunk noise into a stable live indicator that still reacts within one chunk.
- **The `<audio>` element is the clock** — chunk classification is driven by playback time, so the UI's judgment cadence *feels* synchronous with the call.
- **End-on-ended** — the clip's `ended` event closes the call automatically, no magic timers.

### 6. PRISM (PRISMtrace) — the evaluation engine, not decoration

PRISM is 20% of the judging weight, but more importantly it's what turns "we built a classifier" into "we measurably improved a classifier":

- Every chunk classification is **one trace** with `model`, input description, output verdict, latency, `session_id`, `agent_id`, and rich `metadata` (`ground_truth`, `correct`, `generator_type`, `chunk_length_s`, `run_version`).
- PRISM's **Root Cause clustering** groups failing traces — surfacing the two weaknesses we actually expect (short-clip accuracy, specific unseen generator types) without manually reading hundreds of failures.
- **`run_version`** tags batches so before/after runs sit side by side on the dashboard and in the exported CSV — the literal slide for the pitch.
- Integration is **fire-and-forget**: any PRISM outage logs a warning and never breaks a classification — observability must never be on the critical path of the product.

A custom MCP server wrapping PRISM was considered and explicitly rejected: none exists officially, the SDK path is simpler, and dashboard usage doesn't need tool-call access.

### 7. ASVspoof 2019 LA — the benchmark judges recognize

ASVspoof is the canonical spoofing-detection challenge dataset: real (bona fide) speech plus synthetic attacks **labeled by generator/attack type** — the latter being what makes PRISM's per-generator diagnosis possible at all. It's freely available (HuggingFace mirrors), so there's no access friction. The LA (logical access) subset matches our threat model exactly: voice synthesized/cloned by TTS + vocoders, as opposed to physical-access replay attacks.

### 8. Engineering-hygiene choices

- **Credentials**: environment variables only, `.env` gitignored, `.env.example` carries placeholders. An ingest-scoped key is used; a leaked-into-`.env.example` incident was caught and scrubbed before any commit.
- **Fail-fast config** with actionable error messages (missing keys tell you exactly what to copy where).
- **Graceful degradation everywhere**: no PRISM → traces skipped; no labels.json → server warns and `/clips` returns empty; checkpoint missing → documented fallback path.
- **Time-boxed fine-tuning** with per-epoch checkpointing: abandoning it mid-run loses nothing, honoring the PRD's "risk: fine-tuning stalls" mitigation.

---

## What Was Deliberately NOT Used (and why)

| Not used | Reason |
|---|---|
| Live call interception / telephony (Twilio, Android call screening) | OS/telecom permission labyrinths; PRD explicitly rules it out as a time trap |
| Mobile app | Web demo is faster to build and easier to show on a projector |
| Speaker diarization / identification | Binary real/fake is the product; diarization adds nothing to the judging criteria |
| LLM frameworks (LangChain etc.) for tracing | This is a classifier, not an LLM chain; the direct SDK client is the documented right tool |
| Cloud GPU / paid inference APIs | CPU latency fits the budget; zero cost, zero vendor risk |
| Chart libraries (Chart.js, recharts) | One SVG trend chart, hand-rolled in ~40 lines |

See also: [ARCHITECTURE.md](ARCHITECTURE.md) · [WORKFLOWS.md](WORKFLOWS.md)
