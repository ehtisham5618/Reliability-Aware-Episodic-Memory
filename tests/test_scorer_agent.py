import json

from rem.agent.graph import RunContext, build_graph, run_task
from rem.baselines.conditions import MAIN, NO_MEMORY, OURS_HAND, PLAIN, XIONG
from rem.corruption.inject import FUNCS, _rng, make_bank
from rem.env.tasks import generate_splits
from rem.llm.client import parse_json
from rem.llm.fake import FakeLLM
from rem.memory.schema import ground_truth_reliable
from rem.memory.store import MemoryStore
from rem.scorer.model import Scorer, decide_post, decide_pre, hand_scorer
from rem.scorer.signals import FEATURES, contradiction, recency, signal_vector, verify

from test_memory_corruption import CLEAN, SPLITS

V2 = [e for e in CLEAN if e.env_version == "v2"]
V1 = [e for e in CLEAN if e.env_version == "v1"]
TEST = SPLITS["test"]


def test_recency_only_penalises_volatile():
    e = V1[0].clone(claims=[c for c in V1[0].claims if not c["volatile"]] or [])
    assert recency(e) == 1.0
    stale = next(x for x in V1 if any(c["volatile"] for c in x.claims))
    assert recency(stale) < 0.2


def test_contradiction_catches_stale_signatures():
    stale = [e for e in V1 if e.family in ("booking", "config")]
    assert all(contradiction(e)[0] == 1 for e in stale)
    assert all(contradiction(e)[0] == 0 for e in V2)


def test_verification_three_states_and_corrections():
    good = next(e for e in V2 if e.family == "expense")
    assert verify(good)["status"] == "pass"
    bad = FUNCS["wrong_fact"](good, _rng(1, good.id, "wrong_fact"))
    v = verify(bad)
    assert v["status"] == "fail" and v["corrections"] and v["probes"] >= 1
    no_probe = good.clone(claims=[c for c in good.claims if not c["probeable"]])
    assert verify(no_probe)["status"] == "unchecked" and verify(no_probe)["probes"] == 0


def test_feature_vector_complete():
    x = signal_vector(TEST[0], V2[0], 0.8)
    assert all(f in x for f in FEATURES)
    assert x["verif_unchecked"] == 1.0


def test_policy_gate_and_revision():
    sc = hand_scorer()
    stale = next(e for e in V1 if e.family == "config")
    x = signal_vector(TEST[1], stale, 0.9)
    x.pop("_conflicts")
    assert decide_pre(sc, x) == "reject"  # hard contradiction gate
    sc_nogate = Scorer(sc.weights, sc.thresholds, None, hard_gate=False)
    assert not sc_nogate.gated(x)
    good = next(e for e in V2 if e.family == "expense")
    bad = FUNCS["wrong_fact"](good, _rng(1, good.id, "wrong_fact"))
    v = verify(bad)
    x2 = signal_vector(TEST[2], bad, 0.9, v["status"])
    x2.pop("_conflicts")
    x2["contradiction"] = 0.0
    assert decide_post(sc, x2, v) == "revise"


def test_revision_restores_reliability():
    from rem.agent.graph import _apply_corrections
    from rem.memory.evidence import EVIDENCE
    n_rev = n_gate = 0
    for good in [e for e in V2 if e.family in ("config", "expense")]:
        bad = FUNCS["wrong_fact"](good, _rng(3, good.id, "wrong_fact"))
        if bad is None:
            continue
        if contradiction(bad)[0]:  # documented fact -> caught by the gate instead
            n_gate += 1
            continue
        v = verify(bad)
        assert v["status"] == "fail" and v["corrections"], bad.claims
        r = _apply_corrections(bad, v["corrections"], bad.id + "~rev1")
        assert ground_truth_reliable(r) and r.revised_from == bad.id and r.parent_ids == [bad.id]
        n_rev += 1
    assert n_rev >= 3 and n_gate >= 3


def test_parse_json_lenient():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('blah {"a": {"b": 2}} tail') == {"a": {"b": 2}}
    assert parse_json("nope") is None


def _run(cond, mode, bank, tasks, scorer=None):
    store = MemoryStore([e.clone() for e in bank]) if cond.retrieval else None
    ctx = RunContext(llm=FakeLLM(), store=store, condition=cond, mode=mode, seed=0, run_tag=f"t-{cond.name}-{mode}",
                     scorer=scorer)
    g = build_graph(ctx)
    recs = [run_task(g, ctx, t) for t in tasks]
    return recs, store


def test_graph_runs_every_condition_fixed():
    tasks = TEST[:4]
    for cond in MAIN:
        recs, store = _run(cond, "fixed", CLEAN, tasks, scorer=hand_scorer() if cond.filter == "ours" else None)
        assert len(recs) == 4
        for r in recs:
            json.dumps(r)  # serialisable
            assert r["llm_calls"] >= 1
            if cond.retrieval:
                assert r["retrieved"]
        if store:
            assert len(store) == len(CLEAN)  # fixed bank: no writes
            store.close()


def test_growing_mode_writes_with_provenance():
    recs, store = _run(PLAIN, "growing", CLEAN, TEST[:6])
    written = [r["written"] for r in recs if r["written"]]
    assert len(written) == 6 and len(store) == len(CLEAN) + 6
    ep = store.get(written[-1])
    assert ep.parent_ids and ep.true_outcome in ("success", "fail")
    store.close()


def test_xiong_add_filter_and_no_memory():
    recs, store = _run(XIONG, "growing", CLEAN, TEST[:4])
    assert len(store) <= len(CLEAN) + 4
    store.close()
    recs, store = _run(NO_MEMORY, "fixed", CLEAN, TEST[:2])
    assert store is None and all(not r["retrieved"] for r in recs)


def test_ours_rejects_corrupted_more_than_plain():
    bank, man = make_bank(CLEAN, 0.5, seed=0)
    tasks = TEST[:16]
    p, s1 = _run(PLAIN, "fixed", bank, tasks)
    o, s2 = _run(OURS_HAND, "fixed", bank, tasks, scorer=hand_scorer())
    bad_plain = sum(not ok for r in p for ok in r["used_gt_reliable"])
    bad_ours = sum(not ok for r in o for ok in r["used_gt_reliable"])
    assert bad_ours < bad_plain
    s1.close(); s2.close()
