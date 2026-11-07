"""Run the experiment grid on the TEST split with FROZEN scorers.

  python scripts/run_experiments.py --block main      # 7 conditions x 5 rates x {fixed,growing} x 3 seeds
  python scripts/run_experiments.py --block ablation  # 7 ablations x {0.2,0.5} x fixed x 3 seeds
  python scripts/run_experiments.py --block crosscheck --backend groq   # small Groq cross-backend check
Resumable; every finished task is appended to results/raw/<block>/<cell>.jsonl."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rem.agent.runner import run_cell  # noqa: E402
from rem.baselines.conditions import ALL  # noqa: E402
from rem.corruption.inject import make_bank, save_bank  # noqa: E402
from rem.env.tasks import generate_splits  # noqa: E402
from rem.llm.client import LLMClient  # noqa: E402
from rem.memory.store import load_jsonl  # noqa: E402
from rem.scorer.model import Scorer, hand_scorer  # noqa: E402

ART = ROOT / "artifacts"


def load_frozen_scorers():
    frozen = json.loads((ART / "FROZEN.json").read_text())["scorers"]
    out = {}
    for name, digest in frozen.items():
        p = ART / f"scorer_{name}" / "scorer.json"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != digest:
            raise SystemExit(f"scorer {name} changed after freezing ({got} != {digest}); refusing to run on test")
        out[name] = Scorer.load(p)
    return out


def scorer_for(cond, scorers):
    if cond.scorer is None:
        return None
    key = {"hand": "hand", "trained": "trained", "nogate": "nogate"}.get(cond.scorer, cond.scorer)
    return scorers[key]


def cells(cfg, block):
    if block == "main":
        for mode in cfg["modes"]:
            for rate in cfg["rates"]:
                for seed in cfg["seeds"]:
                    for c in cfg["main_conditions"]:
                        yield c, mode, rate, seed
    elif block == "ablation":
        for mode in cfg["ablation_modes"]:
            for rate in cfg["ablation_rates"]:
                for seed in cfg["seeds"]:
                    for c in cfg["ablation_conditions"]:
                        yield c, mode, rate, seed
    elif block == "crosscheck":
        for mode in ("fixed", "growing"):
            for rate in (0.0, 0.3):
                for c in ("no_memory", "plain", "xiong", "ours_trained"):
                    yield c, mode, rate, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="main", choices=["main", "ablation", "crosscheck"])
    ap.add_argument("--backend", default=None)
    ap.add_argument("--cell-workers", type=int, default=24)
    ap.add_argument("--task-workers", type=int, default=8)
    ap.add_argument("--n-tasks", type=int, default=None, help="limit test tasks (crosscheck)")
    ap.add_argument("--only", default=None, help="comma list of conditions")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
    backend = a.backend or cfg["backend"]
    llm = LLMClient(backend, max_tokens=cfg["max_tokens"], reasoning_effort=cfg["reasoning_effort"],
                    max_concurrency=cfg["llm_concurrency"] if backend == "vllm" else 3)
    scorers = load_frozen_scorers()
    tasks = generate_splits()["test"]
    if a.block == "crosscheck":
        tasks = tasks[: a.n_tasks or 24]
    clean = load_jsonl(ROOT / "results" / "banks" / "clean_bank.jsonl")
    banks = {}
    for rate in sorted(set(cfg["rates"]) | set(cfg["ablation_rates"])):
        for seed in cfg["seeds"]:
            b, man = make_bank(clean, rate, seed)
            tag = f"bank_r{int(rate*100):02d}_s{seed}"
            save_bank(b, man, ROOT / "results" / "banks", tag)
            banks[(rate, seed)] = b
    todo = [c for c in cells(cfg, a.block) if not a.only or c[0] in a.only.split(",")]
    out_dir = ROOT / "results" / "raw" / (a.block if backend == cfg["backend"] or a.block == "crosscheck" else f"{a.block}_{backend}")
    print(f"{len(todo)} cells x {len(tasks)} tasks on {backend}", flush=True)
    t0 = time.time()

    def go(cell):
        cname, mode, rate, seed = cell
        cond = ALL[cname]
        tag = f"{cname}__{mode}__r{int(rate*100):02d}__s{seed}"
        recs = run_cell(llm, cond, mode, tasks, banks[(rate, seed)], scorer_for(cond, scorers), seed, tag,
                        out_dir / f"{tag}.jsonl", workers=a.task_workers,
                        store_out=(out_dir / "stores" / f"{tag}.jsonl") if mode == "growing" else None,
                        k=cfg["retrieval_k"], max_turns=cfg["max_turns"], max_attempts=cfg["max_attempts"])
        return tag, sum(r["success"] for r in recs) / len(recs)

    done = 0
    with ThreadPoolExecutor(a.cell_workers) as ex:
        futs = {ex.submit(go, c): c for c in todo}
        for f in as_completed(futs):
            tag, sr = f.result()
            done += 1
            print(f"[{done}/{len(todo)}] {tag} success={sr:.3f} | {time.time()-t0:.0f}s | llm={llm.stats}", flush=True)
    (out_dir / f"_llm_stats_{int(time.time())}.json").write_text(json.dumps(llm.stats))


if __name__ == "__main__":
    main()
