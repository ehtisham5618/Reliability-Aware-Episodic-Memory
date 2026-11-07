"""Mock tools with versioned signatures. Errors are explicit for malformed calls but some
conventions fail *silently* (accepted by the tool, rejected by the checker) - this is what
makes experience valuable and corrupted experience harmful."""
from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass, field

from .world import CONFIG_KEYS, DEPARTMENTS, REGION_CODES, SERVICES, WORLD, FX_V1, version_at

TOOL_NAMES = ["kb_lookup", "calc", "fx_convert", "book_room", "config", "log_expense", "submit_answer", "help"]

# Short descriptions the test-time agent sees (version agnostic, deliberately terse).
TOOL_BRIEF = {
    "kb_lookup": "kb_lookup(entity, field): company knowledge base; entity is a name, e.g. kb_lookup(entity='Maple', field='capacity'). Covers rooms, offices, services, departments.",
    "calc": "calc(expression): arithmetic calculator.",
    "fx_convert": "fx_convert(amount, src, dst): currency conversion at today's rate.",
    "book_room": "book_room(...): book a meeting room. Signature changes between API versions.",
    "config": "config(...): read/write service configuration. Keys changed between API versions.",
    "log_expense": "log_expense(dept, amount_usd): record an expense in the ledger.",
    "submit_answer": "submit_answer(answer): submit the final answer for question tasks.",
    "help": "help(tool): exact signature of a tool for the current API version.",
}

HELP = {
    "v1": {
        "book_room": "book_room(room, date, slot) - room: room name; date: 'YYYY-MM-DD'; slot: 'morning'|'afternoon'.",
        "config": "config(op, service, key, value) - op: 'get'|'set'; service: service id; keys: max_conn, timeout.",
    },
    "v2": {
        "book_room": "book_room(room_id, date, slot, region) - room_id: room id; date: 'NN/NN/YYYY'; slot: 'morning'|'afternoon'; region: region code.",
        "config": "config(op, service, key, value) - op: 'get'|'set'; service: service id; keys: connection_limit, timeout_ms.",
    },
    "any": {
        "kb_lookup": "kb_lookup(entity, field) - room fields: room_id, capacity, office; office fields: region, region_code, rooms; service fields: service_id; department fields: code.",
        "calc": "calc(expression) - e.g. calc('12*3+4').",
        "fx_convert": "fx_convert(amount, src, dst) - dst must be 'USD'.",
        "log_expense": "log_expense(dept, amount_usd) - amount in USD.",
        "submit_answer": "submit_answer(answer).",
        "help": "help(tool).",
    },
}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv}


def safe_eval(expr: str) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "round":
            args = [ev(a) for a in n.args]
            return round(*args)
        raise ValueError("unsupported expression")
    return ev(ast.parse(str(expr), mode="eval"))


DATE_V1 = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATE_V2 = re.compile(r"^\d{2}/\d{2}/\d{4}$")


@dataclass
class ToolEnv:
    """Per-task tool environment. ``day`` fixes the API/fact version."""
    day: int
    bookings: list = field(default_factory=list)
    configs: dict = field(default_factory=dict)
    ledger: list = field(default_factory=list)
    answers: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    @property
    def version(self) -> str:
        return version_at(self.day)

    def __post_init__(self):
        v = self.version
        for s in SERVICES:
            self.configs[WORLD.service_id[s]] = {CONFIG_KEYS[v]["conn"]: 100, CONFIG_KEYS[v]["timeout"]: 30 if v == "v1" else 30000}

    # ------------------------------------------------------------------------------
    def call(self, tool: str, args: dict | None, *, dry_run: bool = False, probe: bool = False) -> str:
        args = dict(args or {})
        try:
            out = self._dispatch(tool, args, dry_run)
        except Exception as e:  # noqa: BLE001 - tool errors are observations
            out = f"error: {e}"
        self.calls.append({"tool": tool, "args": args, "obs": out, "dry_run": dry_run, "probe": probe})
        return out

    def _need(self, tool: str, args: dict, required: list[str], optional: tuple = ()):  # signature check
        for a in args:
            if a not in required and a not in optional:
                raise ValueError(f"{tool}() got an unexpected argument '{a}'")
        for r in required:
            if r not in args:
                raise ValueError(f"{tool}() missing required argument '{r}'")

    def _dispatch(self, tool: str, a: dict, dry_run: bool) -> str:
        v = self.version
        if tool == "help":
            t = a.get("tool") or a.get("name") or ""
            return HELP[v].get(t) or HELP["any"].get(t) or f"error: unknown tool '{t}'"
        if tool == "kb_lookup":
            self._need(tool, a, ["entity", "field"])
            return self._kb(str(a["entity"]).strip(), str(a["field"]).strip())
        if tool == "calc":
            self._need(tool, a, ["expression"])
            r = safe_eval(a["expression"])
            return str(round(r, 6))
        if tool == "fx_convert":
            self._need(tool, a, ["amount", "src", "dst"])
            src, dst = str(a["src"]).upper(), str(a["dst"]).upper()
            if dst != "USD" or src not in FX_V1:
                raise ValueError("unsupported currency pair")
            return str(round(float(a["amount"]) * WORLD.fx(src, v), 6))
        if tool == "book_room":
            return self._book(a, dry_run)
        if tool == "config":
            return self._config(a, dry_run)
        if tool == "log_expense":
            self._need(tool, a, ["dept", "amount_usd"])
            amt = float(a["amount_usd"])
            if not dry_run:
                self.ledger.append({"dept": str(a["dept"]).strip(), "amount_usd": amt})
            return "ok: expense logged" if not dry_run else "ok: valid call (dry run)"
        if tool == "submit_answer":
            self._need(tool, a, ["answer"])
            self.answers.append(a["answer"])
            return "ok: answer recorded"
        raise ValueError(f"unknown tool '{tool}'")

    def _kb(self, entity: str, fld: str) -> str:
        v = self.version
        e = entity.lower().replace(" room", "").replace(" office", "").replace(" department", "").replace(" service", "").strip()
        for name, room in WORLD.rooms.items():
            if e == name.lower():
                if fld == "room_id":
                    return room.room_id[v]
                if fld == "capacity":
                    return str(room.capacity[v])
                if fld == "office":
                    return room.office
                raise ValueError("unknown field for room (room_id, capacity, office)")
        for office in WORLD.office_region:
            if e == office.lower():
                if fld == "region":
                    return WORLD.office_region[office]
                if fld == "region_code":
                    return WORLD.region_code(office)
                if fld == "rooms":
                    return json.dumps(WORLD.rooms_in(office))
                raise ValueError("unknown field for office (region, region_code, rooms)")
        for s in SERVICES:
            if e == s:
                if fld == "service_id":
                    return WORLD.service_id[s]
                raise ValueError("unknown field for service (service_id)")
        for d in DEPARTMENTS:
            if e == d.lower():
                if fld == "code":
                    return WORLD.dept_code[d]
                raise ValueError("unknown field for department (code)")
        raise ValueError(f"unknown entity '{entity}'")

    def _book(self, a: dict, dry_run: bool) -> str:
        v = self.version
        slot = a.get("slot")
        if v == "v1":
            self._need("book_room", a, ["room", "date", "slot"])
            if str(a["room"]) not in WORLD.rooms:
                raise ValueError(f"no room named '{a['room']}'")
            if not DATE_V1.match(str(a["date"])):
                raise ValueError("invalid date format")
            rec = {"room_id": WORLD.room_id(str(a["room"]), "v1"), "date": str(a["date"]), "slot": slot, "region": None}
        else:
            self._need("book_room", a, ["room_id", "date", "slot", "region"])
            if str(a["room_id"]) not in WORLD.all_room_ids("v2"):
                raise ValueError(f"no room with id '{a['room_id']}'")
            if not DATE_V2.match(str(a["date"])):
                raise ValueError("invalid date format")
            if str(a["region"]) not in REGION_CODES.values():
                raise ValueError(f"unknown region code '{a['region']}'")
            rec = {"room_id": str(a["room_id"]), "date": str(a["date"]), "slot": slot, "region": str(a["region"])}
        if slot not in ("morning", "afternoon"):
            raise ValueError("slot must be 'morning' or 'afternoon'")
        if dry_run:
            return "ok: valid call (dry run)"
        self.bookings.append(rec)
        return f"ok: booked {rec['room_id']}"

    def _config(self, a: dict, dry_run: bool) -> str:
        v = self.version
        op = a.get("op")
        if op == "get":
            self._need("config", a, ["op", "service", "key"])
        else:
            self._need("config", a, ["op", "service", "key", "value"])
        svc = str(a["service"])
        if svc not in self.configs:
            raise ValueError(f"unknown service id '{svc}'")
        key = str(a["key"])
        valid = (CONFIG_KEYS[v]["conn"], CONFIG_KEYS[v]["timeout"])
        if key not in valid:
            raise ValueError(f"unknown key '{key}'")
        if op == "get":
            return str(self.configs[svc][key])
        if op != "set":
            raise ValueError("op must be 'get' or 'set'")
        val = int(float(a["value"]))
        if dry_run:
            return "ok: valid call (dry run)"
        self.configs[svc][key] = val
        return "ok: config updated"

    # ---- accounting --------------------------------------------------------------
    def n_calls(self, include_probes: bool = True) -> int:
        return sum(1 for c in self.calls if include_probes or not c["probe"])


# ---------------------------------------------------------------------------------- native tool schemas
def _fn(name, desc, props=None, req=None, open_=False):
    params = {"type": "object", "properties": props or {}}
    if req:
        params["required"] = req
    if open_:
        params["additionalProperties"] = True
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}


_S, _N = {"type": "string"}, {"type": "number"}
TOOL_SCHEMAS = [
    _fn("kb_lookup", "Company knowledge base lookup. entity is a name (e.g. 'Maple', 'Lahore', 'billing', 'Finance'); "
        "covers rooms, offices, services and departments.", {"entity": _S, "field": _S}, ["entity", "field"]),
    _fn("calc", "Arithmetic calculator.", {"expression": _S}, ["expression"]),
    _fn("fx_convert", "Currency conversion at today's rate.", {"amount": _N, "src": _S, "dst": _S}, ["amount", "src", "dst"]),
    _fn("book_room", "Book a meeting room. Its arguments changed between API versions; call help if unsure.", open_=True),
    _fn("config", "Read or write a service configuration value. Keys changed between API versions; call help if unsure.", open_=True),
    _fn("log_expense", "Record an expense in the ledger.", {"dept": _S, "amount_usd": _N}, ["dept", "amount_usd"]),
    _fn("submit_answer", "Submit the final answer for question tasks.", {"answer": _S}, ["answer"]),
    _fn("help", "Exact arguments of a tool for the current API version.", {"tool": _S}, ["tool"]),
    _fn("finish", "Call once when the task is complete (status 'done') or cannot be completed (status 'failed').",
        {"status": {"type": "string", "enum": ["done", "failed"]},
         "confidence": {"type": "integer", "description": "0-100: how sure you are the task was done correctly"}},
        ["status", "confidence"]),
]
