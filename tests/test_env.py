import pytest

from rem.env.oracle import oracle_actions
from rem.env.tasks import FAMILIES, SPLIT_SIZES, check, expected, generate_splits
from rem.env.tools import ToolEnv
from rem.env.world import CURRENT_DAY, WORLD, build_world

SPLITS = generate_splits()
ALL = [t for ts in SPLITS.values() for t in ts]


def run(task, actions, day=None):
    env = ToolEnv(day=task.day if day is None else day)
    for a in actions:
        env.call(a["tool"], a["args"])
    return env, check(task, env)


def test_world_is_deterministic():
    a, b = build_world(), build_world()
    assert {k: (r.room_id, r.capacity) for k, r in a.rooms.items()} == {k: (r.room_id, r.capacity) for k, r in b.rooms.items()}
    assert a.dept_code == b.dept_code


def test_versioned_facts_exist():
    assert any(r.room_id["v1"] != r.room_id["v2"] for r in WORLD.rooms.values())
    assert any(r.capacity["v1"] != r.capacity["v2"] for r in WORLD.rooms.values())


def test_split_sizes_and_balance():
    for s, n in SPLIT_SIZES.items():
        assert len(SPLITS[s]) == n
        fams = [t.family for t in SPLITS[s]]
        for f in FAMILIES:
            assert abs(fams.count(f) - n / 4) <= 1


def test_no_leakage_between_splits():
    keys = {s: {t.key() for t in ts} for s, ts in SPLITS.items()}
    assert not keys["build"] & keys["val"]
    assert not keys["build"] & keys["test"]
    assert not keys["val"] & keys["test"]


def test_splits_reproducible():
    again = generate_splits()
    assert [t.task_id + t.text for t in again["test"]] == [t.task_id + t.text for t in SPLITS["test"]]


def test_eval_splits_run_now_under_v2():
    assert all(t.day == CURRENT_DAY and t.version == "v2" for t in SPLITS["val"] + SPLITS["test"])
    assert {t.version for t in SPLITS["build"]} == {"v1", "v2"}


@pytest.mark.parametrize("task", ALL, ids=lambda t: t.task_id)
def test_oracle_passes_every_task(task):
    env, (ok, reason) = run(task, oracle_actions(task))
    assert ok, reason
    assert not any(c["obs"].startswith("error") for c in env.calls)


def _first(family, split="test"):
    return next(t for t in SPLITS[split] if t.family == family)


def test_booking_traps():
    t = _first("booking")
    exp = expected(t)
    good = {"room_id": exp["room_id"], "date": exp["date"], "slot": t.p["slot"], "region": exp["region"]}
    # silent: MM/DD swap is accepted by the tool but fails the checker
    swapped = dict(good, date=f"{t.p['m']:02d}/{t.p['d']:02d}/{t.p['y']}")
    env, (ok, _) = run(t, [{"tool": "book_room", "args": swapped}])
    assert env.calls[0]["obs"].startswith("ok") and not ok
    # explicit: v1 signature errors under v2
    env, (ok, _) = run(t, [{"tool": "book_room", "args": {"room": t.p["room"], "date": "2027-01-02", "slot": "morning"}}])
    assert env.calls[0]["obs"].startswith("error") and not ok
    # double booking fails
    env, (ok, reason) = run(t, [{"tool": "book_room", "args": good}] * 2)
    assert not ok and "double" in reason


def test_config_units_trap():
    t = _first("config")
    svc = WORLD.service_id[t.p["service"]]
    acts = [{"tool": "config", "args": {"op": "set", "service": svc, "key": "connection_limit", "value": t.p["conn"]}},
            {"tool": "config", "args": {"op": "set", "service": svc, "key": "timeout_ms", "value": t.p["timeout_s"]}}]
    env, (ok, reason) = run(t, acts)
    assert all(c["obs"].startswith("ok") for c in env.calls) and not ok and "timeout_ms" in reason
    env, _ = run(t, [{"tool": "config", "args": {"op": "set", "service": svc, "key": "max_conn", "value": 1}}])
    assert env.calls[0]["obs"].startswith("error")


def test_expense_dept_name_trap():
    t = _first("expense")
    exp = expected(t)
    env, (ok, reason) = run(t, [{"tool": "log_expense", "args": {"dept": t.p["dept"], "amount_usd": exp["amount_usd"]}}])
    assert env.calls[0]["obs"].startswith("ok") and not ok and "dept" in reason


def test_stale_v1_rate_fails_under_v2():
    t = _first("expense")
    stale = round(t.p["amount"] * WORLD.fx(t.p["cur"], "v1"), 2)
    _, (ok, _) = run(t, [{"tool": "log_expense", "args": {"dept": expected(t)["dept"], "amount_usd": stale}}])
    assert not ok


def test_dry_run_does_not_mutate():
    t = _first("booking")
    env = ToolEnv(day=t.day)
    a = oracle_actions(t)[-1]
    assert env.call(a["tool"], a["args"], dry_run=True).startswith("ok")
    assert env.bookings == []


def test_calc_is_safe():
    env = ToolEnv(day=200)
    assert env.call("calc", {"expression": "12*3+4"}) == "40"
    assert env.call("calc", {"expression": "__import__('os')"}).startswith("error")
