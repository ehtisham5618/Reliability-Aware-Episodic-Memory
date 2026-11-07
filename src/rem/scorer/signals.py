"""The six reliability signals (LR §6.2) and independent verification probes."""
from __future__ import annotations

import math

from ..env.tasks import Task
from ..env.tools import ToolEnv
from ..env.world import CURRENT_DAY, SERVICES, WORLD
from ..memory.evidence import EVIDENCE
from ..memory.schema import Episode

RECENCY_TAU = 60.0  # days
VERIFY_MAX_PROBES_PER_MEMORY = 2

FEATURES = ["outcome", "confidence", "recency", "context_sim", "version_match",
            "contradiction", "verif_pass", "verif_fail", "verif_unchecked"]
SIGNAL_GROUPS = {  # ablation unit -> feature columns
    "outcome": ["outcome"],
    "confidence": ["confidence"],
    "recency": ["recency"],
    "context": ["context_sim", "version_match"],
    "contradiction": ["contradiction"],
    "verification": ["verif_pass", "verif_fail", "verif_unchecked"],
}


def recency(ep: Episode, now: int = CURRENT_DAY) -> float:
    """Volatility-aware: age is penalised only if the memory makes a volatile claim."""
    if not any(c["volatile"] for c in ep.claims):
        return 1.0
    age = max(0, now - ep.timestamp_day)
    return math.exp(-age / RECENCY_TAU)


def context_sim(task: Task, ep: Episode, text_sim: float) -> float:
    t_ents = task.entities()
    m_ents = ep.claim_entities() | ep.entities()
    overlap = len(t_ents & {x.lower() for x in m_ents}) / max(1, len(t_ents))
    fam = 1.0 if task.family == ep.family else 0.0
    return round(0.4 * max(0.0, text_sim) + 0.3 * overlap + 0.3 * fam, 4)


def contradiction(ep: Episode, verified_claims: dict | None = None) -> tuple[int, list]:
    """1 if any claim conflicts with the versioned evidence store or with a claim from a
    memory that passed verification in this retrieval set. Returns (flag, conflicts)."""
    conflicts = []
    for c in ep.claims:
        doc = EVIDENCE.get(c["key"])
        if doc is not None and c["value"] != doc:
            conflicts.append({"key": c["key"], "memory": c["value"], "evidence": doc, "source": "docs"})
        elif verified_claims and c["key"] in verified_claims and c["value"] != verified_claims[c["key"]]:
            if c["type"] == "fx_rate" and abs(float(c["value"]) - float(verified_claims[c["key"]])) < 1e-4:
                continue
            conflicts.append({"key": c["key"], "memory": c["value"], "evidence": verified_claims[c["key"]], "source": "verified_memory"})
    return (1 if conflicts else 0), conflicts


# ---------------------------------------------------------------------------------- verification
def _probe(c: dict, env: ToolEnv):
    """Returns (status, true_value_or_None). status in pass|fail|unchecked. One tool call."""
    t, ent = c["type"], c.get("entity")
    fieldmap = {"room_id": "room_id", "capacity": "capacity", "region_code": "region_code",
                "service_id": "service_id", "dept_code": "code"}
    if t in fieldmap and ent:
        obs = env.call("kb_lookup", {"entity": ent, "field": fieldmap[t]}, probe=True)
        if obs.startswith("error"):
            return "unchecked", None
        val = int(obs) if t == "capacity" and obs.isdigit() else obs
        return ("pass" if val == c["value"] else "fail"), val
    if t == "fx_rate" and ent:
        obs = env.call("fx_convert", {"amount": 1000, "src": ent, "dst": "USD"}, probe=True)
        try:
            rate = round(float(obs) / 1000, 6)
        except ValueError:
            return "unchecked", None
        return ("pass" if abs(rate - float(c["value"])) <= abs(rate) * 0.002 else "fail"), rate
    if t == "book_room_args":
        obs = env.call("help", {"tool": "book_room"}, probe=True)
        try:
            inside = obs[obs.index("(") + 1: obs.index(")")]
            args = sorted(x.strip() for x in inside.split(","))
        except ValueError:
            return "unchecked", None
        return ("pass" if args == c["value"] else "fail"), args
    if t == "config_key":
        obs = env.call("config", {"op": "get", "service": WORLD.service_id[SERVICES[0]], "key": c["value"]}, probe=True)
        if obs.startswith("error"):
            # the probe tells us the key is wrong; the correct key is in help()
            h = env.call("help", {"tool": "config"}, probe=True)
            keys = h.split("keys:")[-1].strip(" .").split(", ") if "keys:" in h else []
            want = next((k for k in keys if ("conn" in k) == c["key"].endswith("conn")), None)
            return "fail", want
        return "pass", c["value"]
    return "unchecked", None


def verify(ep: Episode, day: int = CURRENT_DAY, max_probes: int = VERIFY_MAX_PROBES_PER_MEMORY) -> dict:
    """Independent verification with tool probes (undocumented claims first, then volatile).
    Result: {'status': pass|fail|unchecked, 'probes': n_tool_calls, 'corrections': {key: true}}"""
    env = ToolEnv(day=day)
    # probe what the evidence store cannot already settle first, then volatile claims
    cands = sorted([c for c in ep.claims if c["probeable"]],
                   key=lambda c: (c["key"] in EVIDENCE, not c["volatile"], c["key"]))
    if not cands:
        return {"status": "unchecked", "probes": 0, "corrections": {}, "checked": []}
    statuses, corrections, checked = [], {}, []
    for c in cands[:max_probes]:
        st, val = _probe(c, env)
        statuses.append(st)
        checked.append(c["key"])
        if st == "fail" and val is not None and not isinstance(val, list):
            corrections[c["key"]] = val  # only concrete scalar evidence can drive a revision
        if st == "fail":
            break
    n = len(env.calls)
    if "fail" in statuses:
        status = "fail"
    elif "pass" in statuses:
        status = "pass"
    else:
        status = "unchecked"
    return {"status": status, "probes": n, "corrections": corrections, "checked": checked}


def signal_vector(task: Task, ep: Episode, text_sim: float, verification: str = "unchecked",
                  calibrator=None, verified_claims: dict | None = None, now: int = CURRENT_DAY) -> dict:
    conf = ep.confidence_raw / 100.0
    if calibrator is not None:
        conf = float(calibrator(conf))
    contra, conflicts = contradiction(ep, verified_claims)
    return {
        "outcome": 1.0 if ep.outcome == "success" else 0.0,
        "confidence": round(conf, 4),
        "recency": round(recency(ep, now), 4),
        "context_sim": context_sim(task, ep, text_sim),
        "version_match": 1.0 if ep.env_version == task.version else 0.0,
        "contradiction": float(contra),
        "verif_pass": 1.0 if verification == "pass" else 0.0,
        "verif_fail": 1.0 if verification == "fail" else 0.0,
        "verif_unchecked": 1.0 if verification == "unchecked" else 0.0,
        "_conflicts": conflicts,
    }
