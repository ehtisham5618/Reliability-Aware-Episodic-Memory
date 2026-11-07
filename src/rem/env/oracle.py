"""Reference (oracle) action sequences. Used for tests and ground-truth replay only - never
shown to the agent."""
from __future__ import annotations

from .tasks import Task, expected
from .world import CONFIG_KEYS, WORLD


def oracle_actions(task: Task, version: str | None = None) -> list[dict]:
    v = version or task.version
    p = task.p
    exp = expected(task, v)
    if task.family == "booking":
        if v == "v1":
            return [{"tool": "book_room", "args": {"room": p["room"], "date": exp["date"], "slot": p["slot"]}}]
        return [
            {"tool": "kb_lookup", "args": {"entity": p["room"], "field": "room_id"}},
            {"tool": "kb_lookup", "args": {"entity": p["office"], "field": "region_code"}},
            {"tool": "book_room", "args": {"room_id": exp["room_id"], "date": exp["date"], "slot": p["slot"], "region": exp["region"]}},
        ]
    if task.family == "config":
        k = CONFIG_KEYS[v]
        svc = WORLD.service_id[p["service"]]
        return [
            {"tool": "kb_lookup", "args": {"entity": p["service"], "field": "service_id"}},
            {"tool": "config", "args": {"op": "set", "service": svc, "key": k["conn"], "value": exp[k["conn"]]}},
            {"tool": "config", "args": {"op": "set", "service": svc, "key": k["timeout"], "value": exp[k["timeout"]]}},
        ]
    if task.family == "expense":
        return [
            {"tool": "kb_lookup", "args": {"entity": p["dept"], "field": "code"}},
            {"tool": "fx_convert", "args": {"amount": p["amount"], "src": p["cur"], "dst": "USD"}},
            {"tool": "log_expense", "args": {"dept": exp["dept"], "amount_usd": exp["amount_usd"]}},
        ]
    if task.family == "aggregate":
        acts = [{"tool": "kb_lookup", "args": {"entity": r, "field": "capacity"}} for r in p["rooms"]]
        return acts + [{"tool": "submit_answer", "args": {"answer": exp["answer"]}}]
    raise ValueError(task.family)
