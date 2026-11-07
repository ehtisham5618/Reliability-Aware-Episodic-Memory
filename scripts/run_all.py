"""One command to reproduce everything.

  python scripts/run_all.py --cache-only   # replay every LLM call from cache/llm_cache.sqlite (no API, no GPU, ~minutes)
  python scripts/run_all.py                # live: deploys the Modal vLLM server, runs missing cells, stops the server

Steps: tests -> bank build -> scorer training/freeze -> main grid -> ablations -> Groq cross-check -> analysis.
Finished cells are skipped (resumable)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def sh(cmd, env=None):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-crosscheck", action="store_true")
    a = ap.parse_args()
    env = dict(os.environ)
    if a.cache_only:
        env["REM_CACHE_ONLY"] = "1"
        env.setdefault("REM_VLLM_URL", "http://cache-only.invalid/v1")
    if not a.skip_tests:
        sh([PY, "-m", "pytest", "-q"], env)
    if not a.cache_only:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        env.update({k: v for k, v in os.environ.items() if k.startswith("REM_")})
    try:
        if not (ROOT / "results" / "banks" / "clean_bank.jsonl").exists() or a.cache_only:
            sh([PY, "scripts/build_bank.py", "--backend", "vllm"], env)
        if not (ROOT / "artifacts" / "FROZEN.json").exists():
            sh([PY, "scripts/train_scorer.py", "--confirm"], env)
        sh([PY, "scripts/run_experiments.py", "--block", "main"], env)
        sh([PY, "scripts/run_experiments.py", "--block", "ablation"], env)
        if not a.skip_crosscheck and (ROOT / "results" / "raw" / "crosscheck").exists():
            sh([PY, "scripts/run_experiments.py", "--block", "crosscheck", "--backend", "groq", "--cell-workers", "2",
                "--task-workers", "2"], env)
        sh([PY, "scripts/analyze.py"], env)
    finally:
        if not a.cache_only:
            subprocess.run(["modal", "app", "stop", "-y", "rem-vllm"], cwd=ROOT)


if __name__ == "__main__":
    main()
