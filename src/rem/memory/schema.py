"""Episode schema, deterministic claim extraction, and ground-truth reliability labels.

Claims are derived from (episode task params, episode actions, observations) - never from the
LLM - so ground truth is computed purely from the environment's versioned facts/conventions.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field

from ..env.tasks import Task, check
from ..env.tools import HELP, ToolEnv
from ..env.world import CONFIG_KEYS, DEPARTMENTS, SERVICES, WORLD, volatile_fact

# claim types that have an independent tool probe
PROBEABLE = {"room_id", "region_code", "capacity", "service_id", "dept_code", "fx_rate", "book_room_args", "config_key"}


@dataclass
class Episode:
    id: str
    task_id: str
    task_text: str
    family: str
    params: dict
    env_version: str
    timestamp_day: int
    actions: list
    observations: list
    outcome: str  # agent-perceived: "success" | "fail"
    confidence_raw: float  # agent self-report 0-100
    lesson: str = ""
    evidence: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    parent_ids: list = field(default_factory=list)
    lesson_of: str | None = None  # lesson -> raw episode link (lessons are stored inline + linked)
    status: str = "active"  # active | rejected | revised
    revised_from: str | None = None
    provenance: dict = field(default_factory=dict)
    # ---- hidden analysis-only fields (never shown to agent or scorer) ----
    true_outcome: str | None = None
    corrupted: str | None = None  # corruption type, or None
    corrupted_from: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        return cls(**d)

    def clone(self, **kw) -> "Episode":
        e = copy.deepcopy(self)
        for k, v in kw.items():
            setattr(e, k, v)
        return e

    def tool_versions(self) -> dict:
        return {"book_room": self.env_version, "config": self.env_version, "fx_convert": self.env_version}

    def entities(self) -> set[str]:
        p = self.params
        out = {str(p[k]).lower() for k in ("room", "office", "service", "dept", "cur") if k in p}
        out |= {r.lower() for r in p.get("rooms", ())}
        return out

    def claim_entities(self) -> set[str]:
        return {str(c["entity"]).lower() for c in self.claims if c.get("entity")}

    def render(self, max_obs: int = 60) -> str:
        acts = "; ".join(f"{a['tool']}({json.dumps(a['args'], separators=(',', ':'))})" for a in self.actions)
        return (f"[{self.id}] task: {self.task_text} | api {self.env_version}, day {self.timestamp_day} | "
                f"actions: {acts} | outcome: {self.outcome} | lesson: {self.lesson}")


def _claim(ctype, key, value, entity=None):
    return {"type": ctype, "key": key, "value": value, "entity": entity,
            "volatile": volatile_fact(ctype, entity or key), "probeable": ctype in PROBEABLE}


def _date_format(date: str, p: dict) -> str:
    d, m, y = p.get("d"), p.get("m"), p.get("y")
    if date == f"{y}-{m:02d}-{d:02d}":
        return "YYYY-MM-DD"
    if date == f"{d:02d}/{m:02d}/{y}":
        return "DD/MM/YYYY"
    if date == f"{m:02d}/{d:02d}/{y}":
        return "MM/DD/YYYY"
    return "other"


_LESSON_DATE = re.compile(r"(YYYY-MM-DD|DD/MM/YYYY|MM/DD/YYYY)")
_LESSON_KEYS = {"max_conn": ("conn", "max_conn"), "connection_limit": ("conn", "connection_limit"),
                "timeout_ms": ("timeout", "timeout_ms")}


def lesson_claims(family: str, lesson: str) -> list:
    """Convention claims stated in the lesson text (what a lessons-only memory teaches).
    Only unambiguous tokens are extracted: date-format patterns and config key names."""
    out = []
    if not lesson:
        return out
    if family == "booking":
        fmts = set(_LESSON_DATE.findall(lesson))
        if len(fmts) == 1:  # a lesson naming several formats is not asserting one
            out.append(_claim("date_format", "lesson:date_format", fmts.pop(), None))
    if family == "config":
        for tok, (slot, val) in _LESSON_KEYS.items():
            if re.search(r"\b" + tok + r"\b", lesson):
                out.append(_claim("config_key", f"lesson:config_key:{slot}", val, None))
        quoted_timeout = re.search(r"`timeout`|'timeout'|\"timeout\"|key timeout\b", lesson)
        if quoted_timeout and not re.search(r"\btimeout_ms\b", lesson):
            out.append(_claim("config_key", "lesson:config_key:timeout", "timeout", None))
    for c in out:
        c["probeable"] = False
    return out


def extract_claims(family: str, params: dict, actions: list, observations: list, lesson: str | None = None) -> list:
    """What this memory asserts about the world, from its own params, actions, observations and
    (optionally) the convention claims stated in its lesson text."""
    p = params
    claims = []
    seen = set()

    def add(c):
        k = (c["type"], c["key"])
        if k not in seen:
            seen.add(k)
            claims.append(c)

    for a, obs in zip(actions, observations + [""] * (len(actions) - len(observations))):
        t, args = a.get("tool"), a.get("args") or {}
        ok = isinstance(obs, str) and not obs.startswith("error")
        if t == "kb_lookup" and ok:
            ent, fld = str(args.get("entity", "")), str(args.get("field", ""))
            e = ent.replace(" room", "").replace(" office", "").replace(" department", "").replace(" service", "").strip()
            cmap = {"room_id": "room_id", "capacity": "capacity", "region_code": "region_code",
                    "service_id": "service_id", "code": "dept_code"}
            if fld in cmap:
                val = int(obs) if fld == "capacity" and obs.isdigit() else obs
                ename = next((x for x in list(WORLD.rooms) + list(WORLD.office_region) + SERVICES + DEPARTMENTS
                              if x.lower() == e.lower()), e)
                add(_claim(cmap[fld], f"{cmap[fld]}:{ename}", val, ename))
        elif t == "book_room":
            add(_claim("book_room_args", "book_room_args", sorted(args), None))
            if "date" in args:
                add(_claim("date_format", "date_format", _date_format(str(args["date"]), p), None))
            if "room_id" in args and "room" in p:
                add(_claim("room_id", f"room_id:{p['room']}", str(args["room_id"]), p["room"]))
            if "region" in args and "office" in p:
                add(_claim("region_code", f"region_code:{p['office']}", str(args["region"]), p["office"]))
        elif t == "config" and args.get("op") == "set":
            key, val = str(args.get("key")), args.get("value")
            if "service" in p:
                add(_claim("service_id", f"service_id:{p['service']}", str(args.get("service")), p["service"]))
            try:
                v = float(val)
            except (TypeError, ValueError):
                v = None
            if v is not None and "conn" in p and v == p["conn"]:
                add(_claim("config_key", "config_key:conn", key, None))
            elif v is not None and "timeout_s" in p and v in (p["timeout_s"], p["timeout_s"] * 1000):
                add(_claim("config_key", "config_key:timeout", key, None))
                add(_claim("timeout_unit", "timeout_unit", "ms" if v == p["timeout_s"] * 1000 and p["timeout_s"] != 0 else "s", None))
        elif t == "log_expense":
            if "dept" in p:
                add(_claim("dept_code", f"dept_code:{p['dept']}", str(args.get("dept")), p["dept"]))
            if "cur" in p and "amount" in p:
                try:
                    rate = round(float(args.get("amount_usd")) / float(p["amount"]), 6)
                    add(_claim("fx_rate", f"fx_rate:{p['cur']}", rate, p["cur"]))
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
        elif t == "submit_answer" and "rooms" in p:
            try:
                add(_claim("capacity_sum", f"capacity_sum:{'+'.join(p['rooms'])}", float(args.get("answer")), None))
            except (TypeError, ValueError):
                pass
    for c in lesson_claims(family, lesson or ""):
        add(c)
    return claims


# ---------------------------------------------------------------------------------- truth
def _expected_args(tool: str, version: str) -> list:
    sig = HELP[version][tool]
    inside = sig[sig.index("(") + 1: sig.index(")")]
    return sorted(x.strip() for x in inside.split(","))


def claim_truth(c: dict, version: str):
    """Value of the claim under ``version`` (None if not determinable)."""
    t, ent = c["type"], c.get("entity")
    if t == "room_id":
        return WORLD.room_id(ent, version) if ent in WORLD.rooms else None
    if t == "capacity":
        return WORLD.capacity(ent, version) if ent in WORLD.rooms else None
    if t == "region_code":
        return WORLD.region_code(ent) if ent in WORLD.office_region else None
    if t == "service_id":
        return WORLD.service_id.get(ent)
    if t == "dept_code":
        return WORLD.dept_code.get(ent)
    if t == "fx_rate":
        return WORLD.fx(ent, version)
    if t == "book_room_args":
        return _expected_args("book_room", version)
    if t == "date_format":
        return "YYYY-MM-DD" if version == "v1" else "DD/MM/YYYY"
    if t == "config_key":
        return CONFIG_KEYS[version]["conn" if c["key"].endswith("conn") else "timeout"]
    if t == "timeout_unit":
        return CONFIG_KEYS[version]["timeout_unit"]
    if t == "capacity_sum":
        rooms = c["key"].split(":", 1)[1].split("+")
        return float(sum(WORLD.capacity(r, version) for r in rooms))
    return None


def claim_is_true(c: dict, version: str) -> bool:
    truth = claim_truth(c, version)
    if truth is None:
        return True
    if c["type"] == "fx_rate":
        # 0.2% relative tolerance absorbs 2-decimal rounding; v1->v2 rate moves are all >0.5%
        return abs(float(c["value"]) - truth) <= abs(truth) * 0.002 + 1e-9
    if c["type"] == "capacity_sum":
        return abs(float(c["value"]) - truth) < 1e-6
    return c["value"] == truth


def ground_truth_reliable(ep: Episode, version: str = "v2") -> bool:
    """Reliable for the current context <=> every claim holds under ``version`` AND the stored
    outcome label agrees with what actually happened. Never uses the agent's scorer."""
    if any(not claim_is_true(c, version) for c in ep.claims):
        return False
    if ep.true_outcome is not None and ep.outcome != ep.true_outcome:
        return False
    return True


def replay(task_like: Task, actions: list, day: int) -> tuple[list, bool, str]:
    """Deterministically replay actions in a fresh env; returns observations and checker result."""
    env = ToolEnv(day=day)
    obs = [env.call(a["tool"], a.get("args") or {}) for a in actions]
    ok, reason = check(task_like, env)
    return obs, ok, reason


_ID_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    return _ID_RE.sub("-", text.lower()).strip("-")
