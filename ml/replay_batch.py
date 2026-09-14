"""Replay the held-out evaluation batch through /classify-chunk.

Every chunk becomes one PRISM trace with ground-truth metadata, tagged with
a run_version so before/after runs are comparable in the PRISM dashboard.

Usage:
    python ml/replay_batch.py [--run-version v1] [--base-url http://127.0.0.1:8000]
                              [--manifest ml/heldout_manifest.json] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "ml" / "heldout_manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-version", default="v1")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--chunk-seconds", type=float, default=3.0)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}\nRun ml/prepare_data.py first.")
        return 1

    clips = json.loads(manifest_path.read_text(encoding="utf-8")).get("clips", [])
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        print("Manifest has no clips.")
        return 1

    # Tag run_version into the backend's per-clip metadata by writing a small
    # override file the backend picks up on startup (simplest robust path:
    # pass run_version via the session instead).
    session_id = f"eval-{args.run_version}-{uuid.uuid4().hex[:6]}"
    print(f"Replaying {len(clips)} clips as session {session_id}")

    correct = 0
    total = 0
    errors = 0
    from collections import defaultdict
    stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    t0 = time.perf_counter()

    for i, clip in enumerate(clips):
        duration = float(clip.get("duration_s", 0.0)) or 3.0
        n_chunks = max(1, int(duration // args.chunk_seconds))
        for chunk_idx in range(n_chunks):
            payload = {
                "clip_id": clip["id"],
                "session_id": session_id,
                "chunk_index": chunk_idx,
                "offset_s": chunk_idx * args.chunk_seconds,
                "duration_s": args.chunk_seconds,
                "run_version": args.run_version,
            }
            try:
                r = requests.post(
                    f"{args.base_url}/classify-chunk", json=payload, timeout=120
                )
                r.raise_for_status()
                res = r.json()
                pred = "synthetic" if res["is_synthetic"] else "real"
                total += 1
                gen = clip.get("generator_type", "unknown")
                for key in ("all", gen):
                    stats[key]["total"] += 1
                if pred == clip.get("label"):
                    correct += 1
                    stats["all"]["correct"] += 1
                    stats[gen]["correct"] += 1
            except Exception as e:
                errors += 1
                print(f"  ! {clip['id']} chunk{chunk_idx}: {e}", file=sys.stderr)
        if (i + 1) % 10 == 0:
            acc = correct / total if total else 0.0
            print(f"  [{i + 1}/{len(clips)} clips] accuracy so far: {acc:.3f}")

    elapsed = time.perf_counter() - t0
    acc = correct / total if total else 0.0
    print(
        f"\nDone: {total} chunks, accuracy={acc:.3f} ({correct}/{total}), "
        f"errors={errors}, elapsed={elapsed:.0f}s"
    )
    print(f"Session id for PRISM: {session_id}")
    print("Verify traces: PRISM dashboard -> Root Cause / Agent Intelligence.")

    print("\nAccuracy by generator type (the PRISM diagnosis view):")
    for gen, s in sorted(stats.items()):
        if gen == "all":
            continue
        a = s["correct"] / s["total"] if s["total"] else 0.0
        print(f"  {gen}: {a:.3f} ({s['correct']}/{s['total']})")

    out = ROOT / "ml" / f"results_{args.run_version}.json"
    out.write_text(json.dumps({
        "run_version": args.run_version,
        "session_id": session_id,
        "model_chunks": total,
        "accuracy": acc,
        "correct": correct,
        "errors": errors,
        "elapsed_s": round(elapsed, 1),
        "by_generator": {
            g: {"acc": s["correct"] / s["total"], "n": s["total"]}
            for g, s in sorted(stats.items()) if g != "all"
        },
    }, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
