"""Pull per-run accuracy numbers from PRISM's read API for pitch slides.

Reads recent traces for the agent, groups by run_version (from metadata), and
prints/saves a before/after summary CSV.

Usage:
    python scripts/export_results.py [--out ml/results_summary.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> dict[str, str]:
    env_path = ROOT / ".env"
    env = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    env.setdefault("PRISMTRACE_HOST", os.environ.get("PRISMTRACE_HOST", ""))
    env.setdefault("PRISMTRACE_API_KEY", os.environ.get("PRISMTRACE_API_KEY", ""))
    env.setdefault("PRISMTRACE_PROJECT_ID", os.environ.get("PRISMTRACE_PROJECT_ID", ""))
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "ml" / "results_summary.csv"))
    args = ap.parse_args()

    env = load_env()
    host, key, project = env["PRISMTRACE_HOST"], env["PRISMTRACE_API_KEY"], env["PRISMTRACE_PROJECT_ID"]
    if not (key and project):
        print("PRISM credentials missing (.env).", file=sys.stderr)
        return 1

    headers = {"X-PRISMtrace-Key": key}
    # NOTE: endpoint path per PRISM's read-scoped API; adjust if the dashboard
    # shows a different traces-export path for your project.
    url = f"{host}/api/traces"
    r = requests.get(
        url,
        headers=headers,
        params={"project_id": project, "agent_id": "voice-scam-classifier", "limit": 1000},
        timeout=60,
    )
    r.raise_for_status()
    traces = r.json() if isinstance(r.json(), list) else r.json().get("traces", [])

    by_run: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for t in traces:
        meta = t.get("metadata") or {}
        run = str(meta.get("run_version", "unknown"))
        by_run[run]["total"] += 1
        if meta.get("correct") is True:
            by_run[run]["correct"] += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_version", "chunks", "correct", "accuracy"])
        for run in sorted(by_run):
            d = by_run[run]
            acc = d["correct"] / d["total"] if d["total"] else 0.0
            w.writerow([run, d["total"], d["correct"], f"{acc:.4f}"])
            print(f"{run}: accuracy={acc:.4f} ({d['correct']}/{d['total']} chunks)")

    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
