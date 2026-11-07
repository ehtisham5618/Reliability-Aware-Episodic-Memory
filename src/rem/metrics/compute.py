"""All metrics are recomputed from raw logs + saved banks/stores (LR §7.2). Nothing is typed by hand."""
from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..env.oracle import oracle_actions
from ..env.tasks import generate_splits
from ..memory.schema import Episode, claim_is_true, extract_claims, ground_truth_reliable, replay

TASKS = {t.task_id: t for t in generate_splits()["test"]}
RNG = np.random.default_rng(12345)
N_BOOT = 10_000


# ---------------------------------------------------------------------------------- loading
def parse_tag(tag: str) -> dict:
    cond, mode, r, s = tag.split("__")
    return {"condition": cond, "mode": mode, "rate": int(r[1:]) / 100, "seed": int(s[1:])}


def load_records(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(raw_dir.glob("*.jsonl")):
        meta = parse_tag(f.stem)
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r.update(meta)
                r["tag"] = f.stem
                rows.append(r)
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def load_bank(banks_dir: str, rate: float, seed: int) -> dict:
    p = Path(banks_dir) / f"bank_r{int(rate*100):02d}_s{seed}.jsonl"
    return {d["id"]: Episode.from_dict(d) for d in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if d}


def load_store(path: Path) -> dict:
    if not path.exists():
        return {}
    return {d["id"]: Episode.from_dict(d) for d in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if d}


@lru_cache(maxsize=None)
def task_keys(task_id: str) -> frozenset:
    t = TASKS[task_id]
    acts = oracle_actions(t)
    obs, _, _ = replay(t, acts, t.day)
    return frozenset(c["key"] for c in extract_claims(t.family, t.p, acts, obs))


# ---------------------------------------------------------------------------------- statistics
def boot_ci(x, n=N_BOOT, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return (float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2)))


def paired_diff(a: pd.Series, b: pd.Series):
    """a, b indexed by (seed, task_id). Paired bootstrap CI of mean(a-b)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return boot_ci((j["a"] - j["b"]).values) + (len(j),)


def fmt(m, lo, hi, pct=True):
    if np.isnan(m):
        return "n/a"
    if pct:
        return f"{100*m:.1f} [{100*lo:.1f}, {100*hi:.1f}]"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------------------------- per-task metrics
def lineage(store: dict, corrupted: set) -> tuple[set, dict]:
    """Tainted ids (corrupted roots + all descendants) and depth from nearest corrupted root."""
    children = defaultdict(list)
    for e in store.values():
        for p in e.parent_ids:
            children[p].append(e.id)
    depth = {c: 0 for c in corrupted}
    frontier = list(corrupted)
    while frontier:
        nxt = []
        for p in frontier:
            for c in children.get(p, []):
                if c not in depth or depth[c] > depth[p] + 1:
                    depth[c] = depth[p] + 1
                    nxt.append(c)
        frontier = nxt
    return set(depth), depth


def enrich(df: pd.DataFrame, banks_dir: Path, stores_dir: Path | None) -> pd.DataFrame:
    """Adds per-task mechanism metrics."""
    out = []
    store_cache = {}
    for _, r in df.iterrows():
        bank = load_bank(str(banks_dir), r["rate"], r["seed"])
        corrupted = {i for i, e in bank.items() if e.corrupted}
        mem = bank
        tainted = corrupted
        if r["mode"] == "growing" and stores_dir is not None:
            if r["tag"] not in store_cache:
                st = load_store(stores_dir / f"{r['tag']}.jsonl")
                store_cache[r["tag"]] = (st, lineage(st, corrupted)[0] if st else corrupted)
            st, tainted = store_cache[r["tag"]]
            mem = st or bank
        cands = r["cands"] or []
        retrieved = [c["id"] for c in cands]
        used = r["used_ids"] or []
        keys = task_keys(r["task_id"])
        # repeated error: bad, applicable claims in retrieved memories -> did the agent repeat them?
        agent_claims = {}
        if r["attempts_log"]:
            t = TASKS[r["task_id"]]
            last = r["attempts_log"][-1]
            agent_claims = {c["key"]: c["value"] for c in extract_claims(t.family, t.p, last["actions"], last["observations"])}
        opp = rep = 0
        for mid in retrieved:
            e = mem.get(mid)
            if e is None:
                continue
            for c in e.claims:
                if c["key"] in keys and not claim_is_true(c, "v2"):
                    opp += 1
                    rep += int(c["key"] in agent_claims and agent_claims[c["key"]] == c["value"])
        safe = [c for c in cands if c.get("gt_reliable")]
        out.append({
            "exposed_retrieved": any(m in tainted for m in retrieved),
            "exposed_used": any(m in tainted for m in used),
            "n_retrieved": len(retrieved), "n_used": len(used),
            "n_used_reliable": int(sum(r["used_gt_reliable"] or [])),
            "n_safe_retrieved": len(safe),
            "n_safe_rejected": sum(1 for c in safe if c.get("decision") == "reject"),
            "p_at_k_family": np.mean([c.get("family_match", False) for c in cands]) if cands else np.nan,
            "p_at_k_entity": np.mean([c.get("family_match", False) and c.get("entity_overlap", False) for c in cands]) if cands else np.nan,
            "rep_opportunities": opp, "rep_repeats": rep,
            "tokens": r["prompt_tokens"] + r["completion_tokens"],
            "all_tool_calls": r["tool_calls"] + r["probe_calls"],
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def propagation(stores_dir: Path, banks_dir: Path) -> pd.DataFrame:
    """Growing mode: erroneous descendants of corrupted memories (count, depth) per run."""
    rows = []
    for f in sorted(stores_dir.glob("*.jsonl")):
        meta = parse_tag(f.stem)
        st = load_store(f)
        bank = load_bank(str(banks_dir), meta["rate"], meta["seed"])
        corrupted = {i for i, e in bank.items() if e.corrupted}
        tainted, depth = lineage(st, corrupted)
        new = [e for e in st.values() if e.id not in bank]  # written during this run (episodes + revisions)
        err_desc = [e for e in new if e.id in tainted and not ground_truth_reliable(e, "v2")]
        rows.append({**meta, "tag": f.stem, "n_written": len(new),
                     "n_tainted_written": sum(1 for e in new if e.id in tainted),
                     "n_erroneous_descendants": len(err_desc),
                     "max_depth": max([depth[e.id] for e in err_desc], default=0),
                     "mean_depth": float(np.mean([depth[e.id] for e in err_desc])) if err_desc else 0.0,
                     "n_erroneous_written": sum(1 for e in new if not ground_truth_reliable(e, "v2")),
                     "n_flagged_rejected": sum(1 for e in st.values() if e.status == "rejected"),
                     "n_revisions": sum(1 for e in new if e.revised_from),
                     "n_written_episodes": sum(1 for e in new if not e.revised_from),
                     "n_corrupted_flagged": sum(1 for i in corrupted if i in st and st[i].status in ("rejected", "revised", "deleted"))})
    return pd.DataFrame(rows)


def attributed_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Paired-run attribution: same condition/mode/seed/task, corrupted bank vs clean (0%) twin.
    An error is attributed to corruption iff the twin succeeded, this run failed, and a corrupted
    memory (or descendant) was used."""
    twin = df[df["rate"] == 0].set_index(["condition", "mode", "seed", "task_id"])["success"]
    rows = []
    for _, r in df[df["rate"] > 0].iterrows():
        k = (r["condition"], r["mode"], r["seed"], r["task_id"])
        if k not in twin.index:
            continue
        tw = bool(twin.loc[k])
        rows.append({"condition": r["condition"], "mode": r["mode"], "rate": r["rate"], "seed": r["seed"], "task_id": r["task_id"],
                     "attributed_error": int(tw and not r["success"] and r["exposed_used"]),
                     "twin_success": tw, "success": r["success"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- calibration
def calibration_rows(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    rows = []
    for _, r in df[df["condition"] == condition].iterrows():
        for c in r["cands"] or []:
            if "score_pre" not in c or "gt_reliable" not in c:
                continue
            s = c.get("score_post", c["score_pre"])
            rows.append({"score": s, "score_pre": c["score_pre"], "y": int(c["gt_reliable"]),
                         "conf": c["x"]["confidence"], "rate": r["rate"], "mode": r["mode"]})
    return pd.DataFrame(rows)


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)
