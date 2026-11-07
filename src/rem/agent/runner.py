"""Run one (condition, mode, corruption rate, seed) cell over a task list. Resumable: finished
cells are skipped; every task record is appended to a JSONL log as soon as it completes."""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..baselines.conditions import Condition
from ..env.tasks import Task
from ..memory.schema import Episode
from ..memory.store import MemoryStore
from ..scorer.model import Scorer
from .graph import RunContext, build_graph, run_task


def run_cell(llm, condition: Condition, mode: str, tasks: list[Task], bank: list[Episode] | None, scorer: Scorer | None,
             seed: int, run_tag: str, out_path: Path, workers: int = 8, store_out: Path | None = None,
             **ctx_kw) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["task_id"]] = r
    if len(done) == len(tasks) and (store_out is None or store_out.exists() or not condition.retrieval):
        return [done[t.task_id] for t in tasks]
    if mode == "growing" and done:  # growing runs must be replayed in order; cache makes it free
        out_path.unlink()
        done = {}

    store = MemoryStore([e.clone() for e in bank], name=f"s-{abs(hash(run_tag)) % 10**12}-{random.Random(run_tag).randint(0, 10**9)}") \
        if (bank is not None and condition.retrieval) else None
    ctx = RunContext(llm=llm, store=store, condition=condition, mode=mode, seed=seed, run_tag=run_tag, scorer=scorer, **ctx_kw)
    graph = build_graph(ctx)
    order = list(tasks)
    random.Random(f"order-{seed}").shuffle(order)  # same order for every condition with this seed
    records = dict(done)

    def _one(t):
        return run_task(graph, ctx, t)

    try:
        with open(out_path, "a", encoding="utf-8") as f:
            if mode == "fixed" or mode == "eval_only":
                todo = [t for t in order if t.task_id not in done]
                with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                    for rec in ex.map(_one, todo):
                        rec["order_index"] = order.index(next(t for t in order if t.task_id == rec["task_id"]))
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        records[rec["task_id"]] = rec
            else:  # growing / build: sequential, memory evolves
                for i, t in enumerate(order if mode == "growing" else tasks):
                    rec = _one(t)
                    rec["order_index"] = i
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    records[rec["task_id"]] = rec
        if store_out is not None and store is not None:
            store.to_jsonl(store_out)
    finally:
        if store is not None:
            store.close()
    return [records[t.task_id] for t in tasks]
