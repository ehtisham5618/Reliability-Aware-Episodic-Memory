"""Build the clean memory bank by running the LLM agent on the `build` split (days 0-180,
v1 then v2 API) with privileged API docs and evaluator feedback for lessons.

  python scripts/build_bank.py --backend vllm
Output: results/banks/clean_bank.jsonl (+ raw logs in results/raw/build/)."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rem.agent.graph import RunContext, build_graph, run_task  # noqa: E402
from rem.baselines.conditions import Condition  # noqa: E402
from rem.env.tasks import generate_splits  # noqa: E402
from rem.llm.client import LLMClient  # noqa: E402
from rem.memory.schema import ground_truth_reliable  # noqa: E402
from rem.memory.store import MemoryStore  # noqa: E402

BUILD = Condition("build", retrieval=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    llm = LLMClient(a.backend)
    tasks = generate_splits()["build"][: a.limit]
    store = MemoryStore([], name="build-bank")
    ctx = RunContext(llm=llm, store=store, condition=BUILD, mode="build", seed=0, run_tag="build", scorer=None)
    g = build_graph(ctx)
    raw = ROOT / "results" / "raw" / "build"
    raw.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(lambda t: run_task(g, ctx, t), tasks))
    with open(raw / "build.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    eps = sorted(store.episodes.values(), key=lambda e: e.task_id)
    out = ROOT / "results" / "banks" / "clean_bank.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in eps:
            e.parent_ids = []  # build episodes have no memory parents
            f.write(json.dumps(e.to_dict()) + "\n")
    stats = {
        "n": len(eps), "success_rate": sum(r["success"] for r in recs) / len(recs),
        "by_version": {v: sum(1 for e in eps if e.env_version == v) for v in ("v1", "v2")},
        "reliable_now_v2": sum(ground_truth_reliable(e, "v2") for e in eps),
        "outcome_success": sum(e.outcome == "success" for e in eps),
        "llm": llm.stats,
        "tokens": {"prompt": sum(r["prompt_tokens"] for r in recs), "completion": sum(r["completion_tokens"] for r in recs)},
    }
    (ROOT / "results" / "banks" / "clean_bank.stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    store.close()


if __name__ == "__main__":
    main()
