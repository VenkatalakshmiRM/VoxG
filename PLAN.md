# VoxG — Cloned-Voice Scam Call Detector

**ForgeAI Hackathon build plan.** Real-vs-AI voice classifier demo: simulated call UI → chunked wav2vec2 inference → PRISM trace logging → diagnose–fix–re-evaluate loop with before/after numbers.

---

## 1. Who Builds It & How

**The agent:** Buffy (Codebuff), working directly in the local checkout, verifying each phase end-to-end before moving on.

**Human in the loop for:** PRISM credentials (project ID + ingest-scoped API key in `.env`, never hardcoded), and the 120-minute PRISM dashboard session.

## 2. Repository Structure

```
voxg/
├── PLAN.md                   # this plan
├── .env.example              # PRISMTRACE_HOST / _PROJECT_ID / _API_KEY (placeholders only)
├── .gitignore                # .env, model cache, dataset files, node_modules, __pycache__
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py           # FastAPI app: mounts the 3 endpoints
│   │   ├── config.py         # env loading via os.environ only (never hardcoded keys)
│   │   ├── classifier.py     # model load at startup, per-chunk inference
│   │   ├── chunking.py       # 2–4s window slicing
│   │   ├── prism.py          # PRISMtrace client, one trace per classification
│   │   └── schemas.py        # pydantic request/response models
│   └── clips/
│       ├── labels.json       # curated demo clips + metadata
│       └── *.wav
├── ml/
│   ├── prepare_data.py       # ASVspoof 2019 LA download + demo/held-out set building
│   ├── finetune.py           # fine-tune head on a slice (time-boxed)
│   ├── zero_shot_eval.py     # v1 fallback scoring (also the "before" number)
│   └── replay_batch.py       # POST held-out batch through /classify-chunk → PRISM
├── frontend/
│   ├── package.json          # React (Vite)
│   └── src/
│       ├── App.jsx           # call flow: pick clip → simulated call → verdict
│       ├── CallScreen.jsx    # simulated call UI, live confidence bar
│       ├── useCallSession.js # session lifecycle, chunk cadence, state
│       └── VerdictScreen.jsx # final verdict + confidence trend graph
└── scripts/
    ├── verify_prism.sh       # handshake + live_connected check
    └── export_results.py     # pull before/after numbers from PRISM
```

## 3. Backend API

| Endpoint | Method | Purpose |
|---|---|---|
| `/clips` | GET | List curated demo clips (id, real/synthetic label, generator_type, duration) |
| `/call-session` | POST | Start a simulated call; returns `session_id` grouping all chunk traces in PRISM |
| `/classify-chunk` | POST | Classify one 3s window; returns `{is_synthetic, confidence, chunk_id, latency_ms}`; emits one PRISM trace |

Chunking is server-side: the frontend sends `clip_id + offset_s + duration_s`, the backend slices, classifies, and traces — keeping trace metadata accurate and avoiding audio re-upload.

### Classifier

- Model: `Sara1708/deepfake-audio-wav2vec2` (ASVspoof 2019 LA-trained). Fallback: `facebook/wav2vec2-base` + fine-tuned head.
- Loaded once at startup; synthetic-class index resolved from `model.config.id2label`.
- Inference: feature extractor → logits → softmax → confidence + binary verdict. 16kHz mono waveform via torchaudio.

### PRISM emission

- `prismtrace-sdk>=0.4.3` client, credentials from `os.environ` only.
- One trace per classification with mandatory metadata: `ground_truth`, `correct`, `generator_type`, `chunk_length_s`, `run_version`, `source_clip`.
- Fire-and-forget with try/except — a PRISM outage never breaks classification.

## 4. Frontend

- **CallScreen:** simulated incoming call (name/avatar, Accept/Decline), call timer, live confidence indicator ("Voice authenticity: 82% likely human").
- **useCallSession:** Accept → POST `/call-session` → every 3s POST `/classify-chunk` → rolling EMA confidence.
- **VerdictScreen:** final verdict + per-chunk confidence trend (SVG line chart, no chart lib).
- Vite dev server proxies `/api/*` to FastAPI.

## 5. Data & Model Pipeline

1. `prepare_data.py` — ASVspoof 2019 LA subset (HF `LanceaKing/asvspoof2019`; Kaggle backup) → 10–20 demo clips (both classes, ≥2 generator types) + 100–300 held-out clips with manifest.
2. `zero_shot_eval.py` — raw checkpoint scores held-out batch → "before" number + v1 fallback.
3. `finetune.py` — time-boxed (~90 min) head fine-tune on a training slice; abandoning it blocks nothing.
4. `replay_batch.py` — POST held-out batch through `/classify-chunk` with `run_version=v1` (then `v2` post-fix) so PRISM shows before/after.

## 6. PRISM Verification

`scripts/verify_prism.sh`: setup-doctor handshake → `GET /api/setup-doctor` → proceed only when `live_connected: true`.

## 7. Build Order (each phase ends with a verification gate)

| # | Phase | Gate |
|---|---|---|
| 0 | PLAN.md, .env.example, .gitignore, README | Files on disk |
| 1 | Scaffold backend + frontend, stub classifier | Both servers start; `/clips` returns JSON |
| 2 | Real wav2vec2 inference + chunking | curl `/classify-chunk` returns sane verdict; latency measured |
| 3 | Frontend integration | End-to-end: UI chunk → API → classifier |
| 4 | PRISM wiring + verify script + replay batch | `live_connected: true`; traces landing |
| 5 | ML scripts (data prep, fine-tune, zero-shot eval) | Scripts run; manifests produced |
| 6 | Diagnose → fix → v2 replay → demo polish | Demo runs cleanly twice in a row |

## 8. Explicitly Not Building (per PRD)

Live call interception · training from scratch · diarization · mobile app · universal generator coverage · a custom PRISM MCP server (none exists — Python SDK is the path).

## 9. Prerequisites From the User

1. PRISM signup → project ID + ingest-scoped API key into `.env`.
2. Python 3.10+ and Node 18+ (confirmed present).
3. GPU optional (plan works CPU-only via zero-shot baseline).
