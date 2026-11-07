"""Train (fit) the reliability scorer on the validation split, then FREEZE the artifacts.

  python scripts/train_scorer.py [--confirm]        # --confirm: LLM check of top-3 thresholds on val

Writes artifacts/scorer_<name>/scorer.json (+ train_report.json) for: trained, hand,
ablation-<signal> (x6), nogate; and artifacts/FROZEN.json with their sha256 digests.
No test-split data is touched."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rem.memory.store import load_jsonl  # noqa: E402
from rem.env.tasks import generate_splits  # noqa: E402
from rem.scorer import train as T  # noqa: E402
from rem.scorer.model import Scorer, hand_scorer  # noqa: E402
from rem.scorer.signals import SIGNAL_GROUPS  # noqa: E402

ART = ROOT / "artifacts"


def save(s: Scorer, report: dict, name: str):
    d = ART / f"scorer_{name}"
    s.save(d)
    (d / "train_report.json").write_text(json.dumps(report, indent=1))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
    T.UTILITY.update(cfg["training"]["utility"])
    T.MAX_WRONG_REJECT = cfg["training"]["max_wrong_reject"]
    clean = load_jsonl(ROOT / "results" / "banks" / "clean_bank.jsonl")
    val = generate_splits()["val"]
    print(f"collecting val records: {len(val)} tasks x {len(cfg['val_seeds'])} seeds x {len(cfg['rates'])} rates")
    recs = T.collect(clean, val, seeds=tuple(cfg["val_seeds"]), rates=tuple(cfg["rates"]), k=cfg["retrieval_k"])
    tdir = ART / "training_data"
    tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / "val_records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{len(recs)} (memory, task) records; reliable fraction={sum(r['y'] for r in recs)/len(recs):.3f}")

    out = {}
    s, rep = T.train_scorer(recs, name="trained")
    out["trained"] = (s, rep)
    for g in SIGNAL_GROUPS:
        out[f"ablation-{g}"] = T.train_scorer(recs, dropped=[g], name=f"ablation-{g}")
    out["nogate"] = T.train_scorer(recs, gate=False, name="nogate")
    h = hand_scorer()
    out["hand"] = (h, {"note": "a-priori weights, no fitting", "val_policy": T.simulate_policy(recs, h)})
    for name, (sc, rp) in out.items():
        rp["val_policy_chosen"] = T.simulate_policy(recs, sc)
        save(sc, rp, name)
        print(f"{name:24s} thresholds={sc.thresholds} val={rp['val_policy_chosen']}")

    if a.confirm:
        from confirm_thresholds import confirm  # noqa: E402
        best = confirm(cfg, clean, val, out["trained"][0], out["trained"][1]["top_thresholds"])
        s = out["trained"][0]
        s.thresholds = best["thresholds"]
        out["trained"][1]["llm_confirmation"] = best["all"]
        save(s, out["trained"][1], "trained")

    frozen = {}
    for name in out:
        p = ART / f"scorer_{name}" / "scorer.json"
        frozen[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (ART / "FROZEN.json").write_text(json.dumps({"scorers": frozen, "note": "frozen before any test-split run"}, indent=1))
    print("FROZEN:", json.dumps(frozen, indent=1))


if __name__ == "__main__":
    main()
