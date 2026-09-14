# VoxG — Standing Instructions for Coding Agents

## Project context

VoxG is a cloned-voice scam-call detector for the ForgeAI Hackathon: FastAPI +
wav2vec2 backend, React call-screen frontend, ASVspoof data pipeline in `ml/`,
PRISM (prismtrace) observability wired into every classification.

## Environment rules

- **Never download to C: drive.** It has ~15 GB free. All datasets, model
  caches (`HF_HOME`), and artifacts go to `D:\VoxGData\` (49 GB free).
  ASVspoof parquet shards already live at `D:\VoxGData\asvspoof\`.
- torch here is **CPU-only** (the CUDA wheel download times out). Do not retry
  CUDA installs; do not propose GPU-dependent steps for this machine.
- Windows environment: Git Bash for shell commands; avoid Unicode glyphs
  (`→`, `✓`) in Python print output (cp1252 console raises
  `UnicodeEncodeError`); use `os.replace` not `os.rename` for overwrites.
- `torchaudio.load` is broken here (needs TorchCodec) — use `soundfile` for
  all audio I/O.

## Verification gates (run before declaring work done)

1. `cd backend && python -m uvicorn app.main:app --port 8000` starts clean
   (model load ~20–25 s; watch `.uvicorn.log`).
2. `curl -s http://127.0.0.1:8000/clips` returns the 16 demo clips.
3. `python ml/replay_batch.py --run-version <tag>` completes with `errors=0`
   and writes `ml/results_<tag>.json`.
4. `bash scripts/verify_prism.sh` prints `live_connected: true`.

## PRISM rules

- Credentials come from `.env` (gitignored). Never hardcode, never commit.
- Trace emission is fire-and-forget: a PRISM outage must never fail a
  classification. Check `.uvicorn.log` for silent emission errors — the SDK
  call signature uses `output=` (not `output_message=`).
- Every trace carries: `ground_truth`, `correct`, `generator_type`,
  `chunk_length_s`, `source_clip`, `run_version`, `decision_threshold`.
  Misclassified chunks additionally carry `requires_review: true`,
  `review_reason: misclassification`, and a `MISCLASSIFIED` output marker so
  PRISM's RCA failure clustering sees real failure signal.
- Before/after comparisons are tagged by `run_version` (v1 = pre-fix baseline,
  v2 = post-fix). Do not reuse a version tag for different parameter states.

## Data rules

- Demo clips (UI) live in `backend/clips/` + `labels.json`; held-out eval
  clips live in `ml/data/heldout/` + `ml/heldout_manifest.json`. The backend
  resolves them separately; held-out clips never appear in the UI picker.
- `ml/results_*.json` and `ml/threshold_sweep.json` are run artifacts —
  regenerate them, never hand-edit.

## Git rules

- Never commit `.env`, `ml/data/`, `.uvicorn.log`, `.vite.log`, or HF caches.
- Commit messages: plain, imperative, no Unicode glyphs.
- `.env.example` contains key names only, never values.

## AI component

See `docs/AI_MODEL.md` for the model card, the threshold-calibration
rationale, and the fine-tuning verdict (currently: not needed; residual
errors are out-of-representation, not marginal).
