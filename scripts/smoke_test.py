"""Smoke test (<=10 distinct tasks): checks LLM output format, tool use, memory use and logging.

  python scripts/smoke_test.py --backend groq --n-build 5 --n-test 5
Uses an *oracle-built* memory bank for the test part (smoke only - never used for results).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rem.agent.runner import run_cell  # noqa: E402
from rem.baselines.conditions import NO_MEMORY, PLAIN  # noqa: E402
from rem.env.oracle import oracle_actions  # noqa: E402
from rem.env.tasks import generate_splits  # noqa: E402
from rem.llm.client import LLMClient  # noqa: E402
from rem.memory.schema import Episode, extract_claims, replay  # noqa: E402


def oracle_bank(tasks):
    out = []
    for t in tasks:
        acts = oracle_actions(t)
        obs, ok, _ = replay(t, acts, t.day)
        e = Episode(id=f"oracle-{t.task_id}", task_id=t.task_id, task_text=t.text, family=t.family, params=t.p,
                    env_version=t.version, timestamp_day=t.day, actions=acts, observations=obs,
                    outcome="success" if ok else "fail", confidence_raw=90, lesson="", true_outcome="success" if ok else "fail")
        e.claims = extract_claims(e.family, e.params, e.actions, e.observations)
        out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="groq")
    ap.add_argument("--n-build", type=int, default=5)
    ap.add_argument("--n-test", type=int, default=5)
    a = ap.parse_args()
    assert a.n_build + a.n_test <= 10
    llm = LLMClient(a.backend)
    sp = generate_splits()
    out = ROOT / "results" / "smoke"
    build = [t for t in sp["build"] if t.version == "v2"][: a.n_build]
    rb = run_cell(llm, PLAIN, "build", build, [], None, 0, "smoke-build", out / "build.jsonl", workers=1)
    test = sp["test"][: a.n_test]
    bank = [e for e in oracle_bank(sp["build"]) if e.env_version == "v2"]
    rn = run_cell(llm, NO_MEMORY, "fixed", test, None, None, 0, "smoke-nomem", out / "nomem.jsonl", workers=1)
    rp = run_cell(llm, PLAIN, "fixed", test, bank, None, 0, "smoke-plain", out / "plain.jsonl", workers=1)
    summary = {}
    for name, rs in [("build(privileged)", rb), ("test no_memory", rn), ("test plain(oracle bank)", rp)]:
        summary[name] = {
            "success": f"{sum(r['success'] for r in rs)}/{len(rs)}",
            "llm_calls": sum(r["llm_calls"] for r in rs), "prompt_tokens": sum(r["prompt_tokens"] for r in rs),
            "completion_tokens": sum(r["completion_tokens"] for r in rs),
            "failures": [(r["task_id"], r["checker_reason"]) for r in rs if not r["success"]],
        }
    summary["client_stats"] = llm.stats
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
