# VoxG — Workflows & Operations Manual

> Step-by-step guides for running everything: local dev, the live demo, the evaluation loop that powers the pitch, and troubleshooting. Written for whoever operates the project — teammate, judge, or future-you at 2 AM.

---

## 0. Prerequisites

| Requirement | Check |
|---|---|
| Python 3.10+ | `python --version` |
| Node 18+ | `node --version` |
| PRISM account + API key | key in `.env` (see §1) |
| GPU | **not required** (CPU inference is fast enough) |

---

## 1. First-Time Setup (one machine, ~10 min)

```bash
# 1. Backend deps
cd backend
pip install -r requirements.txt

# 2. Frontend deps
cd ../frontend
npm install

# 3. Secrets — copy the template, then edit in your real values
cp ../.env.example ../.env
#    edit ../.env:
#      PRISMTRACE_HOST=https://prism-api-prod.up.railway.app
#      PRISMTRACE_PROJECT_ID=<your-project-uuid>
#      PRISMTRACE_API_KEY=pt-sk-...
```

Get PRISM credentials: sign up at prism.blockconvey.com → the project ID appears in the URL/onboarding → create an **ingest-scoped** API key (Settings → API Keys).

## 2. Verify PRISM Connectivity (before anything else)

```bash
bash scripts/verify_prism.sh
```

Expected final line: **`PRISM LIVE CONNECTED — proceed with the real batch.`**

This runs two checks in order: (1) credential handshake — proves the key works; (2) setup-doctor status — proves live traffic is actually arriving. Do not run the evaluation batch until it passes. If handshake passes but `live_connected` stays false, see §7.

## 3. Run the Demo (2 terminals)

```bash
# Terminal 1 — backend (from backend/)
python -m uvicorn app.main:app --port 8000
# wait for: "Application startup complete" (model load takes ~15-20s)

# Terminal 2 — frontend (from frontend/)
npm run dev
# open http://localhost:5173
```

Demo flow in the browser:

1. **Pick a clip** from the grid (👤 = real, 🤖 = synthetic — team view only; hide or blur this screen when judges watch).
2. Click **📲 Simulate Incoming Call** — a ringing phone screen appears.
3. Click **Accept** — the clip plays as a "call"; every 3 seconds a chunk is classified and the **Voice authenticity** bar updates live.
4. The call auto-ends when the clip ends → **Verdict screen**: final banner (✅ likely human / ⚠️ likely AI-generated), the per-chunk confidence trend chart, and the chunk table.
5. Click **New Call** to reset.

**PRISM proof:** while the demo runs, the PRISM dashboard receives one trace per chunk, grouped into a trajectory by `session_id` — open Root Cause / Agent Intelligence and show the live session.

## 4. Real Data Preparation (~30–60 min, mostly download)

```bash
python ml/prepare_data.py
```

What it does:
1. Downloads ASVspoof 2019 LA (eval subset) from HuggingFace.
2. Writes **16 demo clips** (8 real / 8 synthetic across ≥2 generator types) into `backend/clips/` + `labels.json` — replacing the synthetic test tones.
3. Writes `ml/heldout_manifest.json` — **200 clips** with ground truth + generator type for evaluation.

Restart the backend afterwards (labels load at startup). Fallback if the download fails: `python ml/prepare_data.py --from-dir <dir-with-wavs>`.

## 5. The Evaluation Loop (the pitch's core story)

### 5.1 Baseline ("before")

```bash
# backend must be running (§3)
python ml/replay_batch.py --run-version v1
```

Pushes every held-out clip, chunk by chunk, through `/classify-chunk`. Each chunk becomes a PRISM trace tagged `run_version=v1`. The script prints running accuracy — note the final number.

(Offline alternative without the server: `python ml/zero_shot_eval.py`.)

### 5.2 Diagnose (PRISM dashboard, human-driven)

Open PRISM → Root Cause / Agent Intelligence → filter to failing traces. Look for the two classic patterns:
- failures clustering on **short `chunk_length_s`** values, or
- failures clustering on one **`generator_type`** the model hasn't seen.

Pick **ONE** weakness. Write down what it is and the affected accuracy — this is the pitch's villain.

### 5.3 Fix

Pick the fix that matches the diagnosis:
- **Short clips** → change chunk length (frontend `CHUNK_SECONDS` in `useCallSession.js` + `--chunk-seconds` on replay) or trim short windows during training.
- **One generator type** → fine-tune on that generator's samples:
  `python ml/finetune.py --data-dir <slice-with-weak-generator> --minutes 90`
  then point `VOXG_MODEL_ID` in `.env` at `ml/checkpoints/latest` and restart the backend.

### 5.4 Re-evaluate ("after")

```bash
python ml/replay_batch.py --run-version v2
```

Same batch, new tag — PRISM now shows v1 vs v2 side by side.

### 5.5 Export for slides

```bash
python scripts/export_results.py --out ml/results_summary.csv
```

Grouped before/after accuracy per run version — paste into the deck.

## 6. Event-Night Checklist

- [ ] `bash scripts/verify_prism.sh` → LIVE CONNECTED
- [ ] Backend up; frontend up; one rehearsal call completed end to end
- [ ] Real demo clips in place (`backend/clips/labels.json` non-test)
- [ ] v1 batch replayed; PRISM dashboard open on the second screen
- [ ] Demo-clip verdicts precomputed (run each demo clip once beforehand) as lag fallback
- [ ] Pitch deck has placeholders ready for the before/after CSV numbers
- [ ] Team knows the ONE weakness story before walking in

## 7. Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `Cannot reach the backend` banner in UI | Backend not running, or still loading the model (wait ~20s) |
| `404: Unknown clip_id` / `Audio file missing` | Labels vs. files mismatch → re-run `ml/prepare_data.py`, restart backend |
| Handshake OK but `live_connected: false` | No real app trace yet → run one classification (`curl` or one UI call), re-check |
| Traces silently missing | Check backend log for `PRISM trace emission failed` (emission is non-fatal by design); verify `.env` host is `https://prism-api-prod.up.railway.app` |
| `TypeError ... unexpected keyword argument` from prismtrace | SDK signature drift — use `output=` (not `output_message=`); check `inspect.signature(PRISMtrace.trace_llm)` |
| Confidence always ≈ 0.51 | Model loaded with a randomly-initialized head (incomplete checkpoint) → verify `classifier.load()` logs the real `id2label` |
| First classification after startup is slow (~2s) | One-time warm-up; run a throwaway chunk before the demo |
| Audio doesn't play in the browser | Browser autoplay policy — click anywhere on the page once, then Accept |

## 8. Repo Map

```
VoxG/
├── PLAN.md                 ← approved build plan (phases + gates)
├── README.md               ← quick start
├── docs/                   ← ARCHITECTURE · TECH_STACK · WORKFLOWS (this file)
├── backend/                ← FastAPI app + classifier + PRISM emission + clips
├── frontend/               ← React call-screen demo UI
├── ml/                     ← data prep · fine-tune · zero-shot eval · batch replay
└── scripts/                ← PRISM verify · results export · test-clip generator
```

See also: [ARCHITECTURE.md](ARCHITECTURE.md) · [TECH_STACK.md](TECH_STACK.md)
