from collections import Counter

import pytest

from rem.corruption.inject import FUNCS, TYPES, _rng, corruption_order, make_bank
from rem.env.oracle import oracle_actions
from rem.env.tasks import generate_splits
from rem.memory.evidence import EVIDENCE
from rem.memory.schema import Episode, extract_claims, ground_truth_reliable, replay
from rem.memory.store import MemoryStore

SPLITS = generate_splits()


def oracle_episode(task) -> Episode:
    acts = oracle_actions(task)
    obs, ok, _ = replay(task, acts, task.day)
    ep = Episode(id=f"ep-{task.task_id}", task_id=task.task_id, task_text=task.text, family=task.family,
                 params=task.p, env_version=task.version, timestamp_day=task.day, actions=acts,
                 observations=obs, outcome="success" if ok else "fail", confidence_raw=90,
                 lesson="Use DD/MM/YYYY dates and department codes; timeout_ms is in milliseconds.",
                 true_outcome="success" if ok else "fail")
    ep.claims = extract_claims(ep.family, ep.params, ep.actions, ep.observations)
    return ep


CLEAN = [oracle_episode(t) for t in SPLITS["build"]]


def test_oracle_episodes_reliable_in_own_version():
    for e in CLEAN:
        assert ground_truth_reliable(e, e.env_version), e.id
        assert e.claims, e.id


def test_v1_booking_and_config_memories_are_stale_under_v2():
    v1 = [e for e in CLEAN if e.env_version == "v1" and e.family in ("booking", "config")]
    assert v1 and all(not ground_truth_reliable(e, "v2") for e in v1)


def test_v2_memories_reliable_now():
    v2 = [e for e in CLEAN if e.env_version == "v2"]
    assert all(ground_truth_reliable(e, "v2") for e in v2)


@pytest.mark.parametrize("ctype", TYPES)
def test_each_corruption_type_breaks_reliability(ctype):
    n_applied = 0
    for e in CLEAN:
        if e.env_version != "v2":
            continue
        ce = FUNCS[ctype](e, _rng(0, e.id, ctype))
        if ce is None:
            continue
        n_applied += 1
        assert ce.corrupted == ctype and ce.corrupted_from == e.id and ce.id == e.id
        assert not ground_truth_reliable(ce, "v2"), (ctype, e.id, ce.claims)
        assert e.corrupted is None  # original untouched
    assert n_applied >= 10


def test_banks_deterministic_nested_and_sized():
    b1, m1 = make_bank(CLEAN, 0.3, seed=1)
    b2, m2 = make_bank(CLEAN, 0.3, seed=1)
    assert m1 == m2 and [e.to_dict() for e in b1] == [e.to_dict() for e in b2]
    prev = set()
    for r in (0.0, 0.1, 0.2, 0.3, 0.5):
        _, m = make_bank(CLEAN, r, seed=1)
        assert m["n_corrupted"] == round(r * len(CLEAN))
        assert prev <= set(m["corrupted_ids"])
        prev = set(m["corrupted_ids"])
    _, m_other = make_bank(CLEAN, 0.3, seed=2)
    assert m_other["corrupted_ids"] != m1["corrupted_ids"]


def test_all_types_represented_at_50pct():
    order = corruption_order(CLEAN, seed=0)
    c = Counter(t for _, t, _ in order)
    assert set(c) == set(TYPES), c


def test_zero_rate_bank_equals_clean():
    b, m = make_bank(CLEAN, 0.0, seed=3)
    assert m["n_corrupted"] == 0
    assert [e.to_dict() for e in b] == [e.to_dict() for e in CLEAN]


def test_evidence_documents_signatures_but_not_silent_rules():
    assert EVIDENCE["config_key:timeout"] == "timeout_ms"
    assert "date_format" not in EVIDENCE and "timeout_unit" not in EVIDENCE
    facts = [k for k in EVIDENCE if ":" in k and not k.startswith("config_key")]
    assert 0.3 < len(facts) / (30 + 12 + 8 + 8) < 0.7


def test_store_retrieval_is_relevant_and_keeps_rejected():
    store = MemoryStore(CLEAN[:60])
    q = SPLITS["test"][0]
    hits = store.retrieve(q.text, k=4)
    assert len(hits) == 4 and hits[0][0].family == q.family
    store.set_status(hits[0][0].id, "rejected")
    assert hits[0][0].id in store.episodes  # flagged, not deleted
    store.set_status(hits[0][0].id, "deleted")
    assert hits[0][0].id not in [e.id for e, _ in store.retrieve(q.text, k=60)]
    store.close()


def test_retrieval_never_empty_under_concurrency():
    """Regression test: Chroma's in-process query intermittently returned [] under concurrency."""
    from concurrent.futures import ThreadPoolExecutor
    tasks = SPLITS["test"][:10]

    def cell(i):
        st = MemoryStore([e.clone() for e in CLEAN], name=f"conc-{i}")
        with ThreadPoolExecutor(4) as ex:
            out = list(ex.map(lambda t: len(st.retrieve(t.text, k=4)), tasks))
        st.close()
        return out

    with ThreadPoolExecutor(16) as ex:
        res = [n for out in ex.map(cell, range(32)) for n in out]
    assert all(n == 4 for n in res)
