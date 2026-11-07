"""Programmatic task generation, automatic checkers and data splits."""
from __future__ import annotations

import calendar
import hashlib
import json
import random
from dataclasses import asdict, dataclass

from .tools import ToolEnv
from .world import CONFIG_KEYS, CURRENT_DAY, DEPARTMENTS, FX_V1, SERVICES, WORLD, version_at

FAMILIES = ["booking", "config", "expense", "aggregate"]
SPLIT_SIZES = {"build": 150, "val": 60, "test": 60}
SPLIT_SEED = 7


@dataclass(frozen=True)
class Task:
    task_id: str
    split: str
    family: str
    text: str
    params: tuple  # sorted (key, value) pairs; hashable
    day: int

    @property
    def p(self) -> dict:
        return dict(self.params)

    @property
    def version(self) -> str:
        return version_at(self.day)

    def key(self) -> str:
        """Identity of the task instance ignoring split/day (used for leakage checks)."""
        return hashlib.sha256(json.dumps([self.family, list(self.params)]).encode()).hexdigest()[:16]

    def entities(self) -> set[str]:
        p = self.p
        out = set()
        for k in ("room", "office", "service", "dept", "cur"):
            if k in p:
                out.add(str(p[k]).lower())
        for r in p.get("rooms", ()):
            out.add(r.lower())
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["params"] = self.p
        return d


# ---------------------------------------------------------------------------------- generation
def _sample_params(family: str, rng: random.Random) -> dict:
    if family == "booking":
        room = rng.choice(sorted(WORLD.rooms))
        m = rng.randint(1, 12)
        d = rng.choice([x for x in range(1, 13) if x != m])  # ambiguous DD/MM vs MM/DD on purpose
        return {"room": room, "office": WORLD.rooms[room].office, "y": 2027, "m": m, "d": d,
                "slot": rng.choice(["morning", "afternoon"])}
    if family == "config":
        return {"service": rng.choice(SERVICES), "conn": rng.randrange(50, 510, 10), "timeout_s": rng.randint(5, 120)}
    if family == "expense":
        cur = rng.choice(sorted(FX_V1))
        amt = rng.randrange(2000, 95000, 50) if cur == "PKR" else rng.randrange(20, 950, 5)
        return {"dept": rng.choice(DEPARTMENTS), "amount": amt, "cur": cur}
    if family == "aggregate":
        k = rng.choice([2, 3])
        return {"rooms": tuple(sorted(rng.sample(sorted(WORLD.rooms), k)))}
    raise ValueError(family)


def render(family: str, p: dict) -> str:
    if family == "booking":
        return (f"Book the {p['room']} room in the {p['office']} office for "
                f"{calendar.month_name[p['m']]} {p['d']}, {p['y']}, {p['slot']} slot.")
    if family == "config":
        return (f"Set the {p['service']} service's connection limit to {p['conn']} and its request "
                f"timeout to {p['timeout_s']} seconds.")
    if family == "expense":
        return f"Log an expense of {p['amount']} {p['cur']} for the {p['dept']} department."
    if family == "aggregate":
        names = ", ".join(p["rooms"][:-1]) + f" and {p['rooms'][-1]}"
        return f"What is the total seating capacity of the {names} rooms? Submit the number."
    raise ValueError(family)


def generate_splits(seed: int = SPLIT_SEED, sizes: dict | None = None) -> dict[str, list[Task]]:
    sizes = sizes or SPLIT_SIZES
    rng = random.Random(seed)
    seen: set[str] = set()
    out: dict[str, list[Task]] = {}
    for split, n in sizes.items():
        tasks = []
        i = 0
        while len(tasks) < n:
            family = FAMILIES[i % len(FAMILIES)]
            i += 1
            p = _sample_params(family, rng)
            params = tuple(sorted(p.items()))
            k = hashlib.sha256(json.dumps([family, list(params)]).encode()).hexdigest()[:16]
            if k in seen:
                continue
            seen.add(k)
            day = rng.randint(0, 180) if split == "build" else CURRENT_DAY
            tasks.append(Task(f"{split}-{len(tasks):03d}", split, family, render(family, p), params, day))
        if split == "build":  # build episodes happen over time
            tasks.sort(key=lambda t: (t.day, t.task_id))
            tasks = [Task(f"build-{j:03d}", t.split, t.family, t.text, t.params, t.day) for j, t in enumerate(tasks)]
        out[split] = tasks
    return out


# ---------------------------------------------------------------------------------- ground truth
def expected(task: Task, version: str | None = None) -> dict:
    """Ground-truth target values for the task under ``version`` (default: task's own)."""
    v = version or task.version
    p = task.p
    if task.family == "booking":
        date = f"{p['y']}-{p['m']:02d}-{p['d']:02d}" if v == "v1" else f"{p['d']:02d}/{p['m']:02d}/{p['y']}"
        return {"room_id": WORLD.room_id(p["room"], v), "date": date, "slot": p["slot"],
                "region": None if v == "v1" else WORLD.region_code(p["office"])}
    if task.family == "config":
        k = CONFIG_KEYS[v]
        return {"service": WORLD.service_id[p["service"]], k["conn"]: p["conn"],
                k["timeout"]: p["timeout_s"] * (1000 if k["timeout_unit"] == "ms" else 1)}
    if task.family == "expense":
        return {"dept": WORLD.dept_code[p["dept"]], "amount_usd": round(p["amount"] * WORLD.fx(p["cur"], v), 2)}
    if task.family == "aggregate":
        return {"answer": sum(WORLD.capacity(r, v) for r in p["rooms"])}
    raise ValueError(task.family)


def _num(x):
    try:
        return float(str(x).strip().replace(",", ""))
    except ValueError:
        return None


def check(task: Task, env: ToolEnv) -> tuple[bool, str]:
    """Automatic pass/fail checker on final environment state. Returns (passed, reason)."""
    exp = expected(task, env.version)
    if task.family == "booking":
        if not env.bookings:
            return False, "no booking made"
        if len(env.bookings) > 1:
            return False, f"{len(env.bookings)} bookings made (double booking)"
        b = env.bookings[0]
        for k in ("room_id", "date", "slot", "region"):
            if b.get(k) != exp[k]:
                return False, f"wrong {k}: got {b.get(k)!r}, expected {exp[k]!r}"
        return True, "ok"
    if task.family == "config":
        cfg = env.configs.get(exp["service"])
        for k, val in exp.items():
            if k == "service":
                continue
            if cfg[k] != val:
                return False, f"wrong {k}: got {cfg[k]!r}, expected {val!r}"
        return True, "ok"
    if task.family == "expense":
        if len(env.ledger) != 1:
            return False, f"{len(env.ledger)} ledger entries (expected exactly 1)"
        e = env.ledger[0]
        if e["dept"] != exp["dept"]:
            return False, f"wrong dept: got {e['dept']!r}, expected department code {exp['dept']!r}"
        if abs(e["amount_usd"] - exp["amount_usd"]) > 0.011:
            return False, f"wrong amount_usd: got {e['amount_usd']}, expected {exp['amount_usd']}"
        return True, "ok"
    if task.family == "aggregate":
        if not env.answers:
            return False, "no answer submitted"
        got = _num(env.answers[-1])
        if got is None or abs(got - exp["answer"]) > 1e-6:
            return False, f"wrong answer: got {env.answers[-1]!r}, expected {exp['answer']}"
        return True, "ok"
    raise ValueError(task.family)


# ---------------------------------------------------------------------------------- privileged docs
PRIVILEGED_DOCS = {
    "v1": (
        "API v1 reference:\n"
        "- book_room(room, date, slot): room = room NAME; date = 'YYYY-MM-DD'; slot = 'morning'|'afternoon'.\n"
        "- config(op, service, key, value): service = service id (kb_lookup service service_id); keys max_conn and timeout (timeout in SECONDS).\n"
        "- log_expense(dept, amount_usd): dept must be the department CODE (kb_lookup department code); convert with fx_convert and round to 2 decimals.\n"
        "- For capacity questions, kb_lookup each room's capacity, add them, submit_answer(number)."
    ),
    "v2": (
        "API v2 reference:\n"
        "- book_room(room_id, date, slot, region): room_id from kb_lookup(room, room_id); date = 'DD/MM/YYYY' (DAY first); "
        "region = region_code of the room's office (kb_lookup office region_code); slot = 'morning'|'afternoon'.\n"
        "- config(op, service, key, value): service = service id (kb_lookup service service_id); keys connection_limit and timeout_ms "
        "(timeout in MILLISECONDS = seconds*1000).\n"
        "- log_expense(dept, amount_usd): dept must be the department CODE (kb_lookup department code); convert with fx_convert and round to 2 decimals.\n"
        "- For capacity questions, kb_lookup each room's capacity, add them, submit_answer(number)."
    ),
}
