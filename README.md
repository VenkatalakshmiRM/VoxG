# VoxG — Cloned-Voice Scam Call Detector

Real-vs-AI voice classifier demo for the ForgeAI Hackathon. Plays a pre-recorded clip as a simulated "incoming call," classifies 3-second windows in near-real-time as **real human** vs **AI-synthesized**, and logs every classification to PRISM for failure diagnosis.

## Architecture

- **Backend** (`backend/`): Python + FastAPI. Serves `/clips`, `/call-session`, `/classify-chunk`. Runs wav2vec2-based audio deepfake detection per chunk and emits one PRISM trace per classification.
- **Frontend** (`frontend/`): React (Vite). Call-screen UI with a live-updating voice-authenticity confidence bar and a final verdict + trend graph.
- **ML** (`ml/`): ASVspoof 2019 LA data prep, time-boxed fine-tuning, zero-shot baseline eval, held-out batch replay through the API.
- **Observability** (PRISMtrace): every chunk classification is a trace with `ground_truth`, `correct`, `generator_type`, `chunk_length_s` metadata — enabling PRISM's Root Cause clustering to surface weaknesses (e.g., short clips, unseen synthesis generators).

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
cp ../.env.example ../.env   # fill in PRISM credentials
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## PRISM Verification

```bash
bash scripts/verify_prism.sh
```

Only proceed to logging the real evaluation batch when `live_connected` is `true`.

## Workflow

1. `python ml/prepare_data.py` — download ASVspoof subset, build demo + held-out sets.
2. Start backend + frontend, run a simulated call end to end.
3. `python ml/replay_batch.py` — log the held-out batch to PRISM (run_version=v1).
4. PRISM dashboard (Root Cause / Agent Intelligence) → identify ONE weakness → targeted fix → `replay_batch.py` with `run_version=v2` → export before/after numbers.

See `PLAN.md` for the full build plan, and the deep-dive docs:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the system works (plain-English intro, system diagrams, component reference, model details)
- [`docs/AI_MODEL.md`](docs/AI_MODEL.md) — the AI component: how it works, why it works, evaluation evidence, parameter calibration, fine-tuning verdict
- [`docs/TECH_STACK.md`](docs/TECH_STACK.md) — every technology, why it was chosen, and what was deliberately rejected
- [`docs/WORKFLOWS.md`](docs/WORKFLOWS.md) — setup, demo, the PRISM evaluation loop, event-night checklist, troubleshooting
