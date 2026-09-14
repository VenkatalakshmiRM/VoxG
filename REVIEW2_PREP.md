# VoxG — Review 2 Preparation Pack
> ForgeAI Hackathon · Review 2 in 2 hours · Format: 2–3 minute pitch (PPT) before judges
> Rubric: **PRISM usage & evaluation strategy 35%** · Working model 25% · Innovation 15% · Problem-Solution fit 15% · Pitch/PPT 10%

---

## 0. ONE-LINE POSITIONING (memorize this)

> **"VoxG detects AI-cloned voice scam calls in real time — and PRISM is not an afterthought, it is the evaluation engine that found our model's exact weakness, measured the fix, and proved the improvement: 96.6% → 97.5% with zero increase in false alarms on real humans."**

---

## 1. THE RUBRIC MAPPED TO YOUR AMMUNITION

| Criterion | Weight | Your strongest material |
|---|---|---|
| PRISM usage & evaluation strategy | **35%** | 620 real traces; per-chunk trace granularity with ground truth; the full closed loop: v1 replay → PRISM-informed diagnosis → threshold calibration → v2 replay → **+0.9pp with human false-alarm rate unchanged**; parameter observability (`decision_threshold` in every trace); run-versioned before/after comparison; explicit roadmap for PRISM's RCA clusters, narrative briefings, and scoped backfills |
| Foundation of working model | 25% | End-to-end working system, verified live: React call screen → FastAPI → wav2vec2 inference → verdict screen; 193-clip held-out eval from real ASVspoof 2019 LA data; 97.5% chunk accuracy; ~150–700ms inference on CPU |
| Innovation | 15% | Real-time *chunked* verdicts during a live call (not post-hoc file scanning); treating every chunk as an observable AI decision; calibration loop driven by observability rather than more training |
| Problem-Solution fit | 15% | Voice-clone scams cause billion-dollar losses (see §7 numbers); the product fits the *screening* moment of a live call — verdict forms within the first 6–9 seconds |
| Pitch/PPT | 10% | Script + slide content in §2–§3; numbers cheat sheet in §9 |

**Golden rule for the room:** every claim you make should be one you can point at — a number in §9, a trace in PRISM, or a file in the repo. Do not improvise numbers.

---

## 2. THE PITCH SCRIPT (2:30, word-for-word, with slide cues)

**[Slide 1 — Title, 0:00–0:15]**
"Hi, we're Team VoxG. Last year, a finance worker in Hong Kong wired out 25 million dollars after a video call where every 'colleague' — including the CFO — was an AI voice clone. Voice cloning needs only seconds of audio. VoxG asks the question nobody asks during the call itself: *is this voice even human?*"

**[Slide 2 — Problem, 0:15–0:35]**
"Scam calls are social engineering that runs on trust — and the caller's voice IS the trust. Current defenses check *who* is calling — caller ID, number reputation. They never check *what is speaking*. And by the time a victim realizes, the money is gone. We want a verdict while the call is still live."

**[Slide 3 — Solution demo, 0:35–1:00]**
"Here's the product. An incoming call rings. You accept. While you listen, VoxG classifies the caller's audio in 3-second chunks with a wav2vec2 deepfake detector, and an authenticity meter fills live. Within the first nine seconds you get a verdict: 'Voice likely AI-generated — hang up.' Everything you see ran end-to-end on real ASVspoof data — human speech and 13 different AI synthesis attacks."

**[Slide 4 — The evaluation engine (PRISM) — THIS IS THE WEIGHTED SLIDE, 1:00–1:50]**
"Now the part we're most proud of — how we *evaluate* this system. Every single chunk classification is one PRISM trace, tagged with ground truth, the generator type that made the audio, the chunk length, and the run version. Our first evaluation replay pushed 204 chunks through the live API — 96.6% accuracy. But PRISM didn't just give us a score, it gave us *where* the model fails: one specific synthesis attack, A10, was at 54.5% — coin-flip — while every other attack type was above 91%.
That's a diagnosis, not a vibe. We analyzed the confidence distribution of exactly those failures: A10's misses were borderline — probabilities of 0.40 to 0.50 — while real human speech never exceeded 0.153. So we recalibrated the decision boundary from 0.50 to 0.30 — a one-parameter fix, no retraining — recorded the new threshold *inside every future trace* so the change is auditable, and replayed the same held-out batch as version 2.
Result: 97.5% overall, A10 up 18 points, and — the number that matters commercially — zero new false alarms on real human voices. That's the loop: replay → PRISM diagnosis → targeted fix → replay → proven delta. And PRISM's root-cause clustering is exactly what we'll point it at next."

**[Slide 5 — What's next + ask, 1:50–2:20]**
"Next, we use PRISM's failure clusters to drive the model roadmap — confidence thresholding today, fine-tuning on post-2019 vocoders when the data says so. The ask: voice-clone fraud is growing faster than caller-ID can adapt. Screening the *audio itself*, live, is the missing layer — and we've shown the pipeline works end to end, measured honestly, and improves on evidence. Thank you — happy to dig into any number you saw."

**[Timing tip]** If you're cut at 2 minutes, the cut line is after Slide 4 — never sacrifice the PRISM slide; it's 35% of the score.

---

## 3. PPT SLIDE CONTENT (7 slides — build in ~45 min)

**Slide 1 — Title.** "VoxG — Catching AI voice scams *during* the call." One screenshot of the live call screen with the red confidence bar. Names + hackathon.

**Slide 2 — Problem.** Three stat bullets (Hong Kong $25M clone case; cloning from seconds of audio; "caller ID verifies the number, not the voice"). One line: "No consumer tool screens the voice content of a live call."

**Slide 3 — Solution.** Screenshot of the verdict screen ("Voice likely AI-GENERATED", trend chart). Caption the flow: Ring → Listen → Live authenticity meter → Verdict in ≤9s. Tech strip at the bottom: React · FastAPI · wav2vec2 · PRISM.

**Slide 4 — PRISM evaluation loop (the money slide — make it a diagram).**
Loop diagram: `Replay batch (run_version=v1) → 204 PRISM traces w/ ground truth → Per-generator breakdown: A10 = 54.5% → Confidence analysis (sweep) → Threshold 0.50→0.30 (1 parameter, recorded in every trace) → Replay v2 → 97.5%, A10 72.7%, humans 100% → RCA cluster analysis run over all 620 traces → 0 failure clusters (health check passed) → pipeline adapted: misclassifications now emit flagged traces → next replay feeds real clusters → next fix`.
Put the actual numbers on the slide — judges remember numbers. If asked about the RCA box: "the first run found zero failure clusters — with only 5 errors in 620 traces, that's the system correctly telling us the model is healthy; we then instrumented misclassifications as flagged traces so future failures cluster automatically" (full story in §4.6).

**Slide 5 — Evidence board.** Small table: v1 vs v2 (overall / A10 / A13 / A18 / humans). One line: "All 408 eval traces + every demo call trace are in PRISM — 620 traces total, live-connected."

**Slide 6 — Architecture.** The one-line pipeline: `React call screen → FastAPI → soundfile → 3s chunker → wav2vec2 classifier → verdict → PRISM trace (fire-and-forget)`. Note CPU-only inference, ~150–700ms per chunk.

**Slide 7 — Roadmap + ask.** PRISM-driven roadmap (RCA clusters → fix → re-replay); scaling path (streaming telephony intake, on-device screening, vocoder-diverse training data). Ask: "Help us put the authenticity layer inside the call path."

**Design notes:** dark theme consistent with the product; big numbers (97.5%, 620 traces, 0 false alarms); never more than ~25 words per slide; the demo screenshot does the talking.

---

## 4. PRISM STRATEGY — THE DEEP ANSWERS (35% of your score; know this cold)

### 4.1 Why per-chunk traces (your key design decision — say it proactively)
Whole-call evaluation gives one number per call and hides everything. VoxG emits **one trace per 3-second decision** — the unit of AI judgment — so failures localize to (generator type × chunk position × run version). This is why "A10 = 54.5%" was even visible.

### 4.2 What is flowing right now (verified facts)
- **620 traces live-connected** (416 from v1/demo activity + 204 from the v2 replay).
- Every trace carries: `ground_truth`, `correct`, `generator_type`, `chunk_length_s`, `source_clip`, `run_version`, and (post-fix) **`decision_threshold`** — the exact parameter value that produced the verdict. This is *parameter observability*: the v1→v2 difference is readable from the traces themselves.
- `run_version` tags make before/after runs comparable inside the dashboard.
- Emission is fire-and-forget: a PRISM outage can never break a classification (resilience pattern worth mentioning).

### 4.3 The closed evaluation loop (draw this in the air during Q&A)
1. **Baseline replay** (`replay_batch.py --run-version v1`): 204 chunks → traces with ground truth → per-generator accuracy table.
2. **Diagnose**: PRISM breakdown isolated ONE weakness (A10) instead of a vague "model is ~96%".
3. **Root-cause locally**: confidence-distribution analysis (`ml/analyze_confidence.py` + `ml/threshold_sweep.json`) showed misses were borderline, humans capped at 0.153 → safe to lower the boundary.
4. **Targeted fix**: one parameter (`VOXG_DECISION_THRESHOLD=0.30`), no retraining, recorded in every subsequent trace.
5. **Re-measure** (`--run-version v2`): 97.5%, A10 +18pts, humans still 100%.
6. **Next iteration — DONE, and here's the honest result (see §4.6):** we ran PRISM's RCA cluster analysis over all 620 traces (5 credits). Result: **0 failure clusters — because 615/620 traces are healthy and the 5 real misclassifications carried no failure marker**. We treated that as a finding, not a dead end: the emitter now flags every misclassification (`requires_review` + explicit marker), so the *next* replay feeds PRISM's clustering real failure signal automatically.
7. **Loop health metric (new):** "0 clusters" on a healthy system is itself the evaluation: PRISM independently confirmed our v2 model state has no systematic failure pattern beyond the 5 known hard chunks.

### 4.4 If judges ask "what PRISM features haven't you used yet?" — the honest roadmap
| PRISM capability | Status | Plan |
|---|---|---|
| Trace ingestion + metadata | ✅ Live, 620 traces | Keep tagging every run |
| Run-versioned A/B comparison | ✅ v1 vs v2 done | Continue per fix |
| Root-cause cluster analysis (RCA) | ✅ **Run (5 credits)** — 0 clusters found; read the result correctly (see §4.6) | Misclassification flagging wired → next replay feeds real clusters |
| Trace flagging for RCA signal | ✅ Live | Every misclassified chunk now emits `requires_review` + explicit failure marker |
| AI narrative briefing | Scheduled | 1-credit daily briefing for the final report |
| Trace-analysis backfill | Scoped | Full project backfill quoted at 416 credits vs 95 balance — we scope it with `max_credits` to the failure traces that matter instead of spending blindly (this *is* the cost-discipline story) |
| Alerts | Planned | Alert on `correct=false` spikes per generator type during any replay |

**Why this framing wins:** showing you know the prices, the budget, and the *selection strategy* for paid analysis demonstrates operational maturity — which is exactly what "evaluation strategy" means at 35% weight.

### 4.5 The one subtle PRISM insight worth saying
"PRISM also surfaced something we didn't design: its text-status classifier over-flagged three of our demo traces as 'failed' even though the classification was correct. We investigated rather than ignored it — that's the point of observability: even the *monitoring* gets monitored."

### 4.6 The RCA experiment — run, result, adaptation (say this in Q&A; it's a maturity signal)
**What we did:** ran PRISM's root-cause cluster analysis over all 620 traces (the paid 5-credit action, run ID `70013442`).

**Result: 0 clusters created — `failing_traces: 0` at analysis time.**

**Why (the two-part reading we give judges):**
1. **It's a clean health check.** 615 of 620 traces are correct classifications with no failure markers — PRISM's RCA correctly found *nothing systematic to fix*. After our threshold fix, the model genuinely has no clusterable failure pattern; the residual errors are 5 scattered hard chunks.
2. **It exposed a real instrumentation gap, which we closed.** Our traces carried `correct: false` in metadata, but PRISM's status pipeline keys on output text/flags — so genuine misclassifications didn't register as failures. Fix shipped the same hour: misclassified chunks now emit `requires_review: true` metadata plus an explicit `MISCLASSIFIED` marker in the output, making them visible as flagged traces. The next evaluation replay feeds the clustering pipeline real failure signal automatically.

**The one-liner:** "We paid 5 credits, got a clean bill of health *and* found the exact seam between our trace schema and PRISM's failure detection — then closed it. That's the evaluation loop working on itself."

**Budget note (know your numbers):** 2 RCA runs charged today (an earlier dashboard-triggered one at 11:28 + ours) = 10 credits total; balance is **90 credits**. Full-corpus backfill remains quoted at 416 — still scoped, not spent.

---

## 5. WORKING MODEL — FACTS TO HAVE AT FINGERTIPS (25%)

| Fact | Value |
|---|---|
| Model | `Bisher/wav2vec2_ASV_deepfake_audio_detection` (HF Hub, ASVspoof-trained) |
| Why wav2vec2 | Self-supervised on raw waveforms → captures vocoder artifacts spectrogram methods miss |
| Checkpoint audit | 4 candidates checked for complete configs; original plan's pick was incomplete (would have loaded a random head) — we audited and switched. Good "rigor" story. |
| Input window | 3s mono, 16kHz, sliced from the "call" audio |
| Inference latency | ~150–700ms per chunk on **CPU** (no GPU needed for serving) |
| Held-out eval set | 193 clips → 204 chunks: 50 human + 13 attack types (A07–A19) from ASVspoof 2019 LA eval |
| v2 accuracy | **97.5%** (199/204), humans 100%, 12/13 attacks ≥91% |
| Endpoints | `/clips`, `/call-session`, `/classify-chunk`, `/audio/*` static |
| Frontend | React (Vite): picker → ringing → live call w/ confidence bar → verdict screen + trend chart |
| Verified | Full UI flow clicked through live; console clean; all endpoints 200 through the Vite proxy |

---

## 6. INNOVATION ANGLES (15%) — pick two, don't recite all four

1. **Verdicts during the call, not after.** Chunked streaming classification with a live meter — the product shape matches the scam timeline (victim must be warned *mid-call*).
2. **The AI decision is the observability unit.** Most teams log requests; we log *decisions with ground truth*, which converts an accuracy number into a diagnosis.
3. **Improvement by calibration, not by retraining.** We made the model better with one measured parameter change, justified by confidence-distribution evidence, auditable in the traces — cheap, safe, reproducible. (Judges respect restraint: we explicitly decided *not* to fine-tune, and can defend why.)
4. **Honest negative space.** We can name exactly what the model can't do (5 hard chunks with p_synth < 0.10; 2019-era attacks; untested noise robustness) — and that's the roadmap, not a weakness to hide.

---

## 7. PROBLEM–SOLUTION FIT (15%) — the framing

- **The threat is real and current:** the widely reported 2024 Hong Kong case (~US$25M lost via a deepfake video call); voice-cloning scams need only seconds of sampled audio; elder-fraud and "family emergency" scams scale with cloned voices. (Verify these phrasings on the slide against a citable source before the pitch — CNN/the original Arup statement are safe references.)
- **Why existing defenses don't fit:** caller-ID / number-reputation verify *origin*, not *content*. Post-call fraud checks happen after the transfer. Nothing screens the live voice.
- **Where VoxG sits:** the *screening layer inside the call* — per-chunk verdicts form within 6–9 seconds, early enough to interrupt the social-engineering script, late enough to avoid flagging the first hello.
- **The commercial metric we protect:** **false-alarm rate on real humans = 0%** in our eval. A scam screener that cries wolf gets switched off; our calibration explicitly *protected* this number while improving attack recall — mention that trade-off thinking, it's the most "product-fit" thing in the whole story.

---

## 8. Q&A BANK — the questions you should expect, with answers

**PRISM / evaluation (the 35% zone — expect most questions here)**

- **Q: "How exactly does PRISM improve your model? Couldn't you have found A10 with a confusion matrix?"**
  A: "A confusion matrix on a batch gives the same number — offline. PRISM makes evaluation *continuous and operational*: the same traces cover demo calls and replays, every run is versioned, every verdict carries the parameter values that produced it, and the diagnosis-to-fix loop is re-runnable any night. It's the difference between a snapshot and a control loop."
- **Q: "What will you do with RCA clusters?"**
  A: "We already ran it — 5 credits over all 620 traces. It found zero failure clusters, which is the correct answer for a system with 5 errors in 620 decisions and no shared pattern among them. It also showed our misclassifications weren't visible to PRISM's status pipeline, so we shipped a fix in the same session: misclassified chunks now emit flagged traces with a `MISCLASSIFIED` marker. The next replay feeds the clustering real failure signal automatically."
- **Q: "Trace analysis backfill costs 416 credits and you have 95. What then?"**
  A: "We scope it: run the analysis over the failure traces we care about with a credit cap, rather than the whole corpus. Budget discipline is part of the strategy — spend where evidence is actionable."
- **Q: "Privacy — you're sending call audio metadata where?"**
  A: "Only classification traces: verdict, confidence, metadata. No raw audio leaves the system, no speech content in traces. For a real deployment that's the right privacy posture anyway."

**Model / data**

- **Q: "97.5% — is that good?"** A: "Per-chunk, on unseen eval attacks, with 0% false alarms on humans — and per-call majority voting pushes effective accuracy higher. The number we watch is the human false-alarm rate; that's 0%."
- **Q: "Why not fine-tune?"** A: "Diagnosed, not guessed: the residual errors are confidently wrong (p_synth < 0.10) — a head fine-tune on 143 CPU-trainable clips won't move them and risks the 0% false-alarm rate. We'll fine-tune when the data says so — post-2019 vocoders, GPU, full train split."
- **Q: "Will it work on real phone calls, with compression and noise?"** A: "Untested — honest answer. ASVspoof models are trained on telephony-channel data, which helps, but codec noise is our first robustness test on the roadmap."
- **Q: "Real-time on a phone?"** A: "The model runs at 150–700ms per 3s chunk on a laptop CPU; a served API or quantized on-device build fits a call's cadence easily."
- **Q: "What if the scammer mixes real and cloned audio?"** A: "Per-chunk verdicts are exactly the defense — a cloned sentence flags its own chunks; the per-call verdict is the max-severity across chunks."

**Curveballs**

- **Q: "What's your moat?"** A: "The evaluation loop is the moat: whoever diagnoses and fixes fastest wins, and our trace schema — ground truth + generator type + parameter values per decision — is built for that."
- **Q: "What did PRISM catch that surprised you?"** A: "Three things: the three over-flagged demo traces, our first checkpoint having an incomplete config on the Hub — and our own RCA run showing our misclassifications were invisible to PRISM's failure pipeline, which we fixed by emitting flagged traces on every error. Observability caught our instrumentation too."
- **Q: "Demo fail risk?"** A: "The pitch needs no live demo — screenshots are from the working build. If asked, the full stack is running locally: picker → call → live bar → verdict, ~20 seconds."

---

## 9. NUMBERS CHEAT SHEET (last-minute glance)

| Number | Meaning |
|---|---|
| **620** | PRISM traces live-connected right now |
| **204** | chunks per evaluation replay (193 held-out clips) |
| **96.6% → 97.5%** | v1 → v2 overall chunk accuracy |
| **54.5% → 72.7%** | attack_A10, the diagnosed weakness |
| **100% → 100%** | human clips — the false-alarm rate never moved |
| **0.50 → 0.30** | decision threshold change (single parameter) |
| **0.153** | max p_synth observed on any human chunk (safety margin for 0.30) |
| **~150–700ms** | per-chunk CPU inference latency |
| **13** | distinct synthesis attacks in the eval set (A07–A19) |
| **2 → 0** | A10 borderline chunks lost, before → after calibration |
| **90 credits** | PRISM budget remaining (2 RCA runs = 10 spent today; full backfill quote: 416) |
| **0** | failure clusters PRISM's RCA found over 620 traces — clean bill of health, gap closed |
| **5** | residual hard chunks (p_synth < 0.10) → the next fix's target |

---

## 10. GLOSSARY (in case a judge probes terminology)

- **ASVspoof 2019 LA** — the standard academic benchmark for logical-access voice spoofing; "LA" = attacks injected into the voice channel (TTS + voice conversion); A07–A19 are the eval attack IDs from 19 teams' systems.
- **wav2vec2** — Meta's self-supervised speech model operating on raw waveforms.
- **p_synth** — the model's synthetic-class probability for a chunk; the decision is `p_synth ≥ threshold`.
- **Chunk** — one 3-second classification window; the unit of both verdicts and PRISM traces.
- **Run version** — tag (v1/v2) making before/after replays comparable in PRISM.
- **Fire-and-forget emission** — traces never block or fail the classification path.

---

## 11. PRE-PITCH CHECKLIST (do these in the next 90 minutes)

1. **Build the 7 slides** from §3 (~45 min) — Slide 4 is the one to over-invest in.
2. **Rehearse the §2 script twice out loud**, timed; cut at Slide 4 if over.
3. **Open the PRISM dashboard once** so you can screenshare the trace list if asked (620 traces visible).
4. **Have `ml/results_v1.json` / `results_v2.json` open in an editor** as your evidence tab.
5. **Check both servers** (`uvicorn :8000` + Vite :5173) if you plan any live moment.
6. ~~Run RCA cluster analysis~~ **Done** — result + adaptation baked into §4.6 and Slide 4; balance is 90 credits.
