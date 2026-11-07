"""Training = fitting the reliability scorer on the *validation* split (LR §6.2, plan §5).

1. collect (signal vector, ground-truth reliable?) pairs for memories retrieved by val tasks
   from val-seeded corrupted banks (deterministic; no LLM calls)
2. fit logistic-regression signal weights + Platt confidence calibrator (isotonic kept for comparison)
3. pick use/reject/caution thresholds maximising expected utility on val, subject to a cap on
   wrongly rejected safe memories
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from ..agent.graph import _apply_corrections
from ..corruption.inject import make_bank
from ..memory.schema import ground_truth_reliable
from ..memory.store import MemoryStore
from .model import Scorer, decide_post, decide_pre
from .signals import FEATURES, SIGNAL_GROUPS, signal_vector, verify

UTILITY = {"use_reliable": 1.0, "use_unreliable": -2.0, "probe_cost": 0.05}
MAX_WRONG_REJECT = 0.10


def collect(clean_bank, val_tasks, seeds=(100, 101, 102), rates=(0.0, 0.1, 0.2, 0.3, 0.5), k=4):
    """One record per (bank, task, retrieved memory) with pre- and post-verification signals."""
    recs = []
    for seed, rate in itertools.product(seeds, rates):
        bank, man = make_bank(clean_bank, rate, seed)
        store = MemoryStore(bank, name=f"train-{seed}-{int(rate*100)}")
        for t in val_tasks:
            for e, sim in store.retrieve(t.text, k=k):
                x0 = signal_vector(t, e, sim, "unchecked")
                ver = verify(e, day=t.day)
                x1 = signal_vector(t, e, sim, ver["status"])
                rev_ok = None
                if ver["status"] == "fail" and ver["corrections"]:
                    rev_ok = ground_truth_reliable(_apply_corrections(e, ver["corrections"], e.id + "~r"), t.version)
                recs.append({"seed": seed, "rate": rate, "task": t.task_id, "mem": e.id, "conf_raw": e.confidence_raw / 100,
                             "x0": {f: x0[f] for f in FEATURES}, "x1": {f: x1[f] for f in FEATURES},
                             "ver": ver["status"], "probes": ver["probes"], "has_corr": bool(ver["corrections"]),
                             "rev_ok": rev_ok, "y": int(ground_truth_reliable(e, t.version)), "corrupted": e.corrupted})
        store.close()
    return recs


def fit_platt(recs):
    seen = {}
    for r in recs:
        seen[(r["seed"], r["rate"], r["mem"])] = (r["conf_raw"], r["y"])
    X = np.array([[v[0]] for v in seen.values()])
    y = np.array([v[1] for v in seen.values()])
    lr = LogisticRegression(C=1e4).fit(X, y)
    iso = IsotonicRegression(out_of_bounds="clip").fit(X[:, 0], y)
    platt = {"type": "platt", "a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}
    p_platt = 1 / (1 + np.exp(-(platt["a"] * X[:, 0] + platt["b"])))
    report = {"n_memories": int(len(y)), "brier_raw": float(brier_score_loss(y, X[:, 0])),
              "brier_platt": float(brier_score_loss(y, p_platt)),
              "brier_isotonic": float(brier_score_loss(y, iso.predict(X[:, 0])))}
    iso_d = {"type": "isotonic", "x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}
    return platt, iso_d, report


def _rows(recs, calib, feats, gate):
    X, y, groups = [], [], []
    for r in recs:
        for key in ("x0", "x1"):  # unchecked and observed verification states, equally weighted
            x = dict(r[key])
            x["confidence"] = 1 / (1 + math.exp(-(calib["a"] * r["conf_raw"] + calib["b"])))
            if gate and x["contradiction"] >= 1:
                continue  # gated rows never reach the weighted score
            X.append([x[f] for f in feats])
            y.append(r["y"])
            groups.append(r["task"])
    return np.array(X), np.array(y), np.array(groups)


def fit_weights(recs, calib, dropped=(), gate=True, C=1.0):
    drop = {f for g in dropped for f in SIGNAL_GROUPS[g]}
    feats = [f for f in FEATURES if f not in drop and not (gate and f == "contradiction") and f != "verif_unchecked"]
    X, y, groups = _rows(recs, calib, feats, gate)
    cv_auc = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
        if len(set(y[te])) > 1:
            cv_auc.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    m = LogisticRegression(C=C, max_iter=2000).fit(X, y)
    w = {"bias": float(m.intercept_[0])}
    w.update({f: float(c) for f, c in zip(feats, m.coef_[0])})
    w["verif_unchecked"] = 0.0  # reference category
    return w, {"n_rows": int(len(y)), "pos_rate": float(y.mean()), "cv_auc_mean": float(np.mean(cv_auc)),
               "cv_auc_folds": [float(a) for a in cv_auc], "features": feats}


def simulate_policy(recs, scorer: Scorer):
    """Expected utility of the policy on val records (no LLM)."""
    util = probes = 0.0
    safe = wrong_rej = used = used_bad = 0
    for r in recs:
        cal = scorer.calibrate(r["conf_raw"])
        x0 = dict(r["x0"], confidence=cal)
        x1 = dict(r["x1"], confidence=cal)
        d = decide_pre(scorer, x0)
        if d == "verify":
            probes += r["probes"]
            d = decide_post(scorer, x1, {"status": r["ver"], "corrections": {"_": 1} if r["has_corr"] else {}})
        if d == "use":
            used += 1
            used_bad += 1 - r["y"]
            util += UTILITY["use_reliable"] if r["y"] else UTILITY["use_unreliable"]
        elif d == "revise":
            used += 1
            ok = bool(r["rev_ok"])
            used_bad += int(not ok)
            util += UTILITY["use_reliable"] if ok else UTILITY["use_unreliable"]
        if r["y"]:
            safe += 1
            wrong_rej += int(d == "reject")
    util -= UTILITY["probe_cost"] * probes
    return {"utility": util / len(recs), "wrong_reject_rate": wrong_rej / max(1, safe), "probes_per_mem": probes / len(recs),
            "used_frac": used / len(recs), "unreliable_used_frac": used_bad / max(1, used)}


def _precompute(recs, scorer: Scorer):
    """Arrays for a vectorised policy simulation (identical semantics to simulate_policy)."""
    s0, s1, gate0, gate1 = [], [], [], []
    for r in recs:
        cal = scorer.calibrate(r["conf_raw"])
        x0 = dict(r["x0"], confidence=cal)
        x1 = dict(r["x1"], confidence=cal)
        s0.append(scorer.score(x0)); s1.append(scorer.score(x1))
        gate0.append(scorer.gated(x0)); gate1.append(scorer.gated(x1))
    A = {k: np.array(v) for k, v in dict(s0=s0, s1=s1, gate0=gate0, gate1=gate1).items()}
    A["fail"] = np.array([r["ver"] == "fail" for r in recs])
    A["corr"] = np.array([r["has_corr"] for r in recs])
    A["rev_ok"] = np.array([bool(r["rev_ok"]) for r in recs])
    A["y"] = np.array([r["y"] for r in recs]).astype(bool)
    A["probes"] = np.array([r["probes"] for r in recs], dtype=float)
    return A


def _simulate_vec(A, tu, tr, tc, can_verify=True):
    use_pre = ~A["gate0"] & (A["s0"] >= tu)
    band = ~A["gate0"] & (A["s0"] >= tr) & (A["s0"] < tu)
    if can_verify:
        ver = band
        post_ok = ~A["gate1"] & ~A["fail"] & (A["s1"] >= tc)
        revise = ver & ~A["gate1"] & A["fail"] & A["corr"]
        use = use_pre | (ver & post_ok)
        probes = A["probes"][ver].sum()
    else:
        use = use_pre | (band & (A["s0"] >= tc))
        revise = np.zeros_like(use)
        probes = 0.0
    used_good = (use & A["y"]) | (revise & A["rev_ok"])
    used_bad = (use & ~A["y"]) | (revise & ~A["rev_ok"])
    n = len(A["y"])
    util = (UTILITY["use_reliable"] * used_good.sum() + UTILITY["use_unreliable"] * used_bad.sum()
            - UTILITY["probe_cost"] * probes) / n
    rejected = ~(use | revise)
    safe = A["y"].sum()
    return {"utility": float(util), "wrong_reject_rate": float((rejected & A["y"]).sum() / max(1, safe)),
            "probes_per_mem": float(probes / n), "used_frac": float((use | revise).sum() / n),
            "unreliable_used_frac": float(used_bad.sum() / max(1, (use | revise).sum()))}


def pick_thresholds(recs, scorer: Scorer, top=3):
    A = _precompute(recs, scorer)
    can_verify = "verification" not in scorer.dropped
    grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    results = []
    for tu in grid:
        for tr in [g for g in grid if g < tu]:
            for tc in [g for g in grid if tr <= g <= tu]:
                m = _simulate_vec(A, tu, tr, tc, can_verify)
                if m["wrong_reject_rate"] <= MAX_WRONG_REJECT:
                    results.append(({"use": tu, "reject": tr, "caution": tc}, m))
    if not results:
        raise RuntimeError("no threshold setting satisfies the wrong-rejection cap")
    results.sort(key=lambda z: (-z[1]["utility"], z[0]["use"], -z[0]["reject"]))
    return results[:top]


def train_scorer(recs, dropped=(), gate=True, name="trained"):
    platt, iso, cal_rep = fit_platt(recs)
    w, rep = fit_weights(recs, platt, dropped, gate)
    base = Scorer(w, {"use": 0.7, "reject": 0.3, "caution": 0.5}, platt, gate, list(dropped), name)
    top = pick_thresholds(recs, base)
    base.thresholds = top[0][0]
    return base, {"weights_fit": rep, "calibration": cal_rep, "isotonic": iso,
                  "top_thresholds": [{"thresholds": t, **m} for t, m in top]}
