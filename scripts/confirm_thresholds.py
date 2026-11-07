"""LLM confirmation of the top-3 threshold settings on the VAL split (fixed bank, val seed,
30% corruption). Picks the setting with the highest val success; ties -> fewer tool calls."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rem.agent.runner import run_cell  # noqa: E402
from rem.baselines.conditions import OURS_TRAINED  # noqa: E402
from rem.corruption.inject import make_bank  # noqa: E402
from rem.llm.client import LLMClient  # noqa: E402
from rem.scorer.model import Scorer  # noqa: E402


def confirm(cfg, clean, val, scorer: Scorer, top):
    llm = LLMClient(cfg["backend"], max_tokens=cfg["max_tokens"], reasoning_effort=cfg["reasoning_effort"],
                    max_concurrency=cfg["llm_concurrency"])
    rate, seed = cfg["training"]["confirm_rate"], cfg["training"]["confirm_seed"]
    bank, _ = make_bank(clean, rate, seed)
    res = []
    for i, t in enumerate(top):
        s = Scorer(scorer.weights, t["thresholds"], scorer.calibrator, scorer.hard_gate, scorer.dropped, scorer.name)
        recs = run_cell(llm, OURS_TRAINED, "fixed", val, bank, s, seed, f"confirm{i}",
                        ROOT / "results" / "raw" / "val_confirm" / f"thr{i}.jsonl", workers=64)
        succ = sum(r["success"] for r in recs) / len(recs)
        tools = sum(r["tool_calls"] + r["probe_calls"] for r in recs) / len(recs)
        res.append({"thresholds": t["thresholds"], "val_success": succ, "tool_calls_per_task": tools,
                    "sim_utility": t["utility"]})
        print(res[-1])
    best = sorted(res, key=lambda r: (-r["val_success"], r["tool_calls_per_task"]))[0]
    return {"thresholds": best["thresholds"], "all": res}
