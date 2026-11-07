"""Corruption injection (LR §6.3, §7.1).

Five types: wrong_fact, wrong_action, false_success, context_mismatch, stale_version.
For a given seed the corrupted sets are *nested* (10% subset of 20% subset of ... 50%), so
rate curves are paired. A corrupted episode keeps its id (it replaces its clean twin in that
bank) and records hidden ``corrupted`` / ``corrupted_from`` fields for analysis only.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from ..env.tasks import Task, check, render
from ..env.tools import ToolEnv
from ..env.world import DEPARTMENTS, SERVICES, WORLD
from ..memory.schema import Episode, extract_claims, ground_truth_reliable

TYPES = ["wrong_fact", "wrong_action", "false_success", "context_mismatch", "stale_version"]
RATES = [0.0, 0.1, 0.2, 0.3, 0.5]


def _task_of(ep: Episode) -> Task:
    return Task(ep.task_id, "mem", ep.family, ep.task_text, tuple(sorted(ep.params.items())), ep.timestamp_day)


def _finalize(ep: Episode, ctype: str, orig: Episode, *, replay_obs: bool = False) -> Episode:
    task = _task_of(ep)
    # replay in the world the memory claims to describe (its API version)
    task = Task(task.task_id, task.split, task.family, task.text, task.params, _day_for(ep))
    env = ToolEnv(day=_day_for(ep))
    obs = [env.call(a["tool"], a.get("args") or {}) for a in ep.actions]
    if replay_obs:
        ep.observations = obs
    ok, _ = check(task, env)
    # true outcome is judged in the world the memory *claims* to describe
    ep.true_outcome = "success" if ok else "fail"
    ep.claims = extract_claims(ep.family, ep.params, ep.actions, ep.observations)
    ep.corrupted = ctype
    ep.corrupted_from = orig.id
    return ep


def _day_for(ep: Episode) -> int:
    return 50 if ep.env_version == "v1" else max(ep.timestamp_day, 100)


def _rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256(repr(parts).encode()).hexdigest()[:8], 16))


def _sub(text: str, pairs: list[tuple[str, str]]) -> str:
    for a, b in pairs:
        if a and a != b:
            text = re.sub(re.escape(str(a)), str(b), text, flags=re.IGNORECASE)
    return text


def _other(rng, pool, cur):
    return rng.choice([x for x in pool if x != cur])


# ---------------------------------------------------------------------------------- types
def wrong_fact(ep: Episode, rng: random.Random) -> Episode | None:
    e = ep.clone()
    pairs = []
    if e.family == "booking" and e.env_version == "v2":
        idx = [i for i, a in enumerate(e.actions) if a["tool"] == "book_room" and "room_id" in a["args"]]
        if not idx:
            return None
        old = e.actions[idx[-1]]["args"]["room_id"]
        new = _other(rng, sorted(WORLD.all_room_ids("v2")), old)
        pairs = [(old, new)]
    elif e.family == "config":
        old = WORLD.service_id[e.params["service"]]
        new = WORLD.service_id[_other(rng, SERVICES, e.params["service"])]
        pairs = [(old, new)]
    elif e.family == "expense":
        old = WORLD.dept_code[e.params["dept"]]
        new = WORLD.dept_code[_other(rng, DEPARTMENTS, e.params["dept"])]
        pairs = [(old, new)]
    elif e.family == "aggregate":
        caps = [(i, a) for i, a in enumerate(e.actions) if a["tool"] == "kb_lookup" and a["args"].get("field") == "capacity"]
        subs = [(i, a) for i, a in enumerate(e.actions) if a["tool"] == "submit_answer"]
        if not caps or not subs or len(e.observations) < len(e.actions):
            return None
        i, _ = caps[0]
        old_cap = e.observations[i]
        if not str(old_cap).isdigit():
            return None
        delta = rng.choice([-5, -3, 3, 5, 7])
        new_cap = str(max(1, int(old_cap) + delta))
        e.observations[i] = new_cap
        j = subs[-1][0]
        try:
            old_ans = float(e.actions[j]["args"]["answer"])
        except (TypeError, ValueError, KeyError):
            return None
        new_ans = old_ans - int(old_cap) + int(new_cap)
        e.actions[j]["args"]["answer"] = int(new_ans) if new_ans == int(new_ans) else new_ans
        e.lesson = _sub(e.lesson, [(str(int(old_ans)) if old_ans == int(old_ans) else str(old_ans), str(e.actions[j]["args"]["answer"]))])
        return _finalize(e, "wrong_fact", ep)
    else:
        return None
    js = json.dumps([e.actions, e.observations])
    if pairs[0][0] not in js:
        return None
    a, o = json.loads(_sub(js, pairs))
    e.actions, e.observations = a, o
    e.lesson = _sub(e.lesson, pairs)
    return _finalize(e, "wrong_fact", ep)


def wrong_action(ep: Episode, rng: random.Random) -> Episode | None:
    e = ep.clone()
    p = e.params
    if e.family == "booking" and e.env_version == "v2":
        good, bad = f"{p['d']:02d}/{p['m']:02d}/{p['y']}", f"{p['m']:02d}/{p['d']:02d}/{p['y']}"
        hit = False
        for a in e.actions:
            if a["tool"] == "book_room" and a["args"].get("date") == good:
                a["args"]["date"] = bad
                hit = True
        if not hit:
            return None
        e.lesson = _sub(e.lesson, [("DD/MM/YYYY", "MM/DD/YYYY"), ("DD/MM", "MM/DD"), ("day first", "month first"),
                                    ("day-first", "month-first"), (good, bad)])
    elif e.family == "config":
        mult = 1000 if e.env_version == "v2" else 1
        hit = False
        for a in e.actions:
            if a["tool"] == "config" and a["args"].get("op") == "set" and a["args"].get("key") in ("timeout_ms", "timeout"):
                try:
                    if float(a["args"]["value"]) == p["timeout_s"] * mult:
                        a["args"]["value"] = p["timeout_s"] if mult == 1000 else p["timeout_s"] * 1000
                        hit = True
                except (TypeError, ValueError):
                    pass
        if not hit:
            return None
        e.lesson = _sub(e.lesson, [("milliseconds", "seconds"), ("* 1000", ""), ("*1000", ""), ("x 1000", ""),
                                    ("×1000", ""), ("× 1000", ""), (str(p["timeout_s"] * 1000), str(p["timeout_s"]))]
                        if mult == 1000 else [("seconds", "milliseconds")])
    elif e.family == "expense":
        code = WORLD.dept_code[p["dept"]]
        hit = False
        for a in e.actions:
            if a["tool"] == "log_expense" and a["args"].get("dept") == code:
                a["args"]["dept"] = p["dept"]
                hit = True
        if not hit:
            return None
        e.lesson = _sub(e.lesson, [(code, p["dept"]), ("department code", "department name"), ("dept code", "dept name")])
    elif e.family == "aggregate":
        caps = [i for i, a in enumerate(e.actions) if a["tool"] == "kb_lookup" and a["args"].get("field") == "capacity"]
        subs = [i for i, a in enumerate(e.actions) if a["tool"] == "submit_answer"]
        if len(caps) < 2 or not subs or len(e.observations) < len(e.actions) or not str(e.observations[caps[-1]]).isdigit():
            return None
        drop = caps[-1]
        dropped = int(e.observations[drop])
        try:
            old_ans = float(e.actions[subs[-1]]["args"]["answer"])
        except (TypeError, ValueError, KeyError):
            return None
        e.actions[subs[-1]]["args"]["answer"] = int(old_ans - dropped)
        del e.actions[drop]
        del e.observations[drop]
        e.lesson = _sub(e.lesson, [(str(int(old_ans)), str(int(old_ans - dropped)))])
    else:
        return None
    return _finalize(e, "wrong_action", ep)


def false_success(ep: Episode, rng: random.Random) -> Episode | None:
    e = ep.clone()
    if ep.true_outcome == "fail" and ep.outcome == "fail":
        e.outcome = "success"  # natural failure relabelled as success
        e.corrupted, e.corrupted_from = "false_success", ep.id
        e.lesson = _sub(e.lesson, [("failed", "succeeded"), ("did not", "did")])
        return e
    commit = {"booking": "book_room", "config": "config", "expense": "log_expense", "aggregate": "submit_answer"}[e.family]
    idx = [i for i, a in enumerate(e.actions) if a["tool"] == commit]
    if not idx:
        return None
    if e.family == "config":
        idx = idx[-1:]  # drop only the final set -> task left half-done
    keep = [i for i in range(len(e.actions)) if i not in idx]
    e.actions = [e.actions[i] for i in keep]
    e.observations = [e.observations[i] for i in keep if i < len(e.observations)]
    e.outcome = "success"
    out = _finalize(e, "false_success", ep)
    return out if out.true_outcome == "fail" else None


def context_mismatch(ep: Episode, rng: random.Random) -> Episode | None:
    e = ep.clone()
    p = dict(e.params)
    if e.family == "booking":
        old = p["room"]
        new = _other(rng, sorted(WORLD.rooms), old)
        pairs = [(old, new), (p["office"], WORLD.rooms[new].office)]
        p["room"], p["office"] = new, WORLD.rooms[new].office
    elif e.family == "config":
        old = p["service"]
        new = _other(rng, SERVICES, old)
        pairs = [(old, new)]
        p["service"] = new
    elif e.family == "expense":
        old = p["dept"]
        new = _other(rng, DEPARTMENTS, old)
        pairs = [(old, new)]
        p["dept"] = new
    elif e.family == "aggregate":
        rooms = list(p["rooms"])
        old = rooms[0]
        new = _other(rng, [r for r in sorted(WORLD.rooms) if r not in rooms], old)
        rooms[0] = new
        p["rooms"] = tuple(sorted(rooms))
        pairs = [(old, new)]
    else:
        return None
    e.params = p
    e.task_text = render(e.family, p)
    e.lesson = _sub(e.lesson, pairs)  # lesson now *talks about* the new entity; actions still carry old values
    return _finalize(e, "context_mismatch", ep)


def stale_version(ep: Episode, rng: random.Random) -> Episode | None:
    if ep.env_version != "v2":
        return None
    from ..env.oracle import oracle_actions
    e = ep.clone()
    task = _task_of(e)
    v1_actions = oracle_actions(task, "v1")
    if e.family == "aggregate":
        if all(WORLD.capacity(r, "v1") == WORLD.capacity(r, "v2") for r in e.params["rooms"]):
            return None
    if e.family == "expense" and e.params["cur"] not in ("EUR", "GBP", "PKR"):
        pass  # AED/SAR rates also moved (>0.5%), still stale
    e.actions = v1_actions
    e.env_version = "v1"  # version tag reflects the old API; timestamp stays recent
    e.lesson = _sub(e.lesson, [("DD/MM/YYYY", "YYYY-MM-DD"), ("connection_limit", "max_conn"), ("timeout_ms", "timeout"),
                                ("milliseconds", "seconds"), ("room_id", "room name"), ("v2", "v1")])
    out = _finalize(e, "stale_version", ep, replay_obs=True)
    return out


_RAW = {"wrong_fact": wrong_fact, "wrong_action": wrong_action, "false_success": false_success,
        "context_mismatch": context_mismatch, "stale_version": stale_version}


def _guarded(fn):
    """A corruption only counts if it really makes the memory unreliable now (v2); otherwise
    (e.g. a swapped room that happens to have the same capacity) it is 'not applicable'."""
    def wrap(ep, rng):
        ce = fn(ep, rng)
        if ce is None or ground_truth_reliable(ce, "v2"):
            return None
        return ce
    wrap.__name__ = fn.__name__
    return wrap


FUNCS = {k: _guarded(f) for k, f in _RAW.items()}


# ---------------------------------------------------------------------------------- banks
def corruption_order(clean: list[Episode], seed: int, max_rate: float = 0.5) -> list[tuple[str, str, Episode]]:
    """Seeded ordered list of (episode_id, type, corrupted_episode) up to max_rate of the bank.
    Prefixes of this list give nested corrupted sets."""
    rng = random.Random(10_000 + seed)
    ids = [e.id for e in clean]
    rng.shuffle(ids)
    by_id = {e.id: e for e in clean}
    target = int(round(max_rate * len(clean)))
    out = []
    used = set()
    t = 0
    for eid in ids:
        if len(out) >= target:
            break
        # cycle types; fall back to the next applicable type
        for k in range(len(TYPES)):
            ctype = TYPES[(t + k) % len(TYPES)]
            ce = FUNCS[ctype](by_id[eid], _rng(seed, eid, ctype))
            if ce is not None:
                out.append((eid, ctype, ce))
                used.add(eid)
                t += 1
                break
    return out


def make_bank(clean: list[Episode], rate: float, seed: int) -> tuple[list[Episode], dict]:
    order = corruption_order(clean, seed)
    n = int(round(rate * len(clean)))
    chosen = order[:n]
    repl = {eid: ce for eid, _, ce in chosen}
    bank = [repl.get(e.id, e).clone() for e in clean]
    manifest = {"rate": rate, "seed": seed, "n_bank": len(clean), "n_corrupted": len(chosen),
                "corrupted_ids": [eid for eid, _, _ in chosen],
                "types": {eid: t for eid, t, _ in chosen}}
    return bank, manifest


def save_bank(bank: list[Episode], manifest: dict, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{tag}.jsonl", "w", encoding="utf-8") as f:
        for e in bank:
            f.write(json.dumps(e.to_dict()) + "\n")
    (out_dir / f"{tag}.manifest.json").write_text(json.dumps(manifest, indent=1))
