"""Recompute every table and plot from raw logs (results/raw) - nothing is hand-entered.

  python scripts/analyze.py
Writes results/tables/*.csv|.md and results/plots/*.png."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rem.metrics import compute as M  # noqa: E402

RAW, BANKS = ROOT / "results" / "raw", ROOT / "results" / "banks"
TAB, PLOT = ROOT / "results" / "tables", ROOT / "results" / "plots"
MAIN = ["no_memory", "plain", "reflexion", "outcome_only", "xiong", "ours_hand", "ours_trained"]
LABEL = {"no_memory": "No memory", "plain": "Plain retrieval", "reflexion": "Reflexion-style", "outcome_only": "Outcome-only filter",
         "xiong": "Xiong-style add/delete", "ours_hand": "Ours (hand-weighted)", "ours_trained": "Ours (trained)"}
# validated categorical palette (fixed order, see dataviz palette validation in RESULTS.md)
COLOR = dict(zip(MAIN, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
MARKER = dict(zip(MAIN, ["o", "s", "^", "v", "D", "P", "X"]))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save_table(df: pd.DataFrame, name: str):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(df.to_markdown(index=False))


def main():
    PLOT.mkdir(parents=True, exist_ok=True)
    summary = {}
    df = M.load_records(RAW / "main")
    df = M.enrich(df, BANKS, RAW / "main" / "stores")
    df.drop(columns=["attempts_log"]).to_json(TAB / "per_task_main.jsonl", orient="records", lines=True)
    summary["main_task_runs"] = len(df)

    # ---------------- success table with bootstrap CIs (units = seed x task)
    rows = []
    for (mode, cond, rate), g in df.groupby(["mode", "condition", "rate"]):
        m, lo, hi = M.boot_ci(g["success"].astype(float))
        rows.append({"mode": mode, "condition": cond, "rate": rate, "n": len(g), "success": m, "ci_lo": lo, "ci_hi": hi})
    succ = pd.DataFrame(rows)
    save_table(succ, "success_long")
    for mode in ("fixed", "growing"):
        s = succ[succ["mode"] == mode]
        wide = s.assign(cell=[M.fmt(a, b, c) for a, b, c in zip(s.success, s.ci_lo, s.ci_hi)]) \
            .pivot(index="condition", columns="rate", values="cell").reindex([c for c in MAIN if c in set(s.condition)])
        wide.columns = [f"{int(r*100)}% corrupt" for r in wide.columns]
        save_table(wide.reset_index().rename(columns={"condition": "Condition"}), f"success_{mode}")

    # ---------------- paired differences: ours_trained minus each baseline
    rows = []
    key = ["seed", "task_id"]
    for (mode, rate), g in df.groupby(["mode", "rate"]):
        ours = g[g.condition == "ours_trained"].set_index(key)["success"].astype(float)
        for other in MAIN:
            if other == "ours_trained" or other not in set(g.condition):
                continue
            b = g[g.condition == other].set_index(key)["success"].astype(float)
            m, lo, hi, n = M.paired_diff(ours, b)
            rows.append({"mode": mode, "rate": rate, "vs": other, "diff_pp": 100 * m, "ci_lo_pp": 100 * lo, "ci_hi_pp": 100 * hi,
                         "n_pairs": n, "significant": bool(lo > 0 or hi < 0)})
    save_table(pd.DataFrame(rows), "paired_diff_ours_trained")

    # ---------------- mechanism metrics
    rows = []
    for (mode, cond, rate), g in df.groupby(["mode", "condition", "rate"]):
        used = g["n_used"].sum()
        safe = g["n_safe_retrieved"].sum()
        exp = g[g["exposed_retrieved"]]
        rows.append({"mode": mode, "condition": cond, "rate": rate,
                     "reliable_used_frac": g["n_used_reliable"].sum() / used if used else np.nan,
                     "unnecessary_rejection": g["n_safe_rejected"].sum() / safe if safe else np.nan,
                     "p_at_k_family": g["p_at_k_family"].mean(), "p_at_k_entity": g["p_at_k_entity"].mean(),
                     "repeated_error_rate": g["rep_repeats"].sum() / g["rep_opportunities"].sum() if g["rep_opportunities"].sum() else np.nan,
                     "repeat_opportunities": int(g["rep_opportunities"].sum()),
                     "exposed_tasks": len(exp), "recovery_rate": exp["success"].mean() if len(exp) else np.nan,
                     "mem_used_per_task": g["n_used"].mean()})
    mech = pd.DataFrame(rows)
    save_table(mech.round(4), "mechanism")

    att = M.attributed_errors(df)
    if len(att):
        a = att.groupby(["mode", "condition", "rate"])["attributed_error"].agg(["mean", "sum", "count"]).reset_index()
        save_table(a.round(4), "attributed_errors")

    # ---------------- propagation (growing)
    prop = M.propagation(RAW / "main" / "stores", BANKS)
    if len(prop):
        save_table(prop, "propagation_runs")
        pa = prop.groupby(["condition", "rate"]).agg(erroneous_descendants=("n_erroneous_descendants", "mean"),
                                                    max_depth=("max_depth", "mean"), mean_depth=("mean_depth", "mean"),
                                                    erroneous_written=("n_erroneous_written", "mean"),
                                                    corrupted_flagged=("n_corrupted_flagged", "mean")).reset_index()
        save_table(pa.round(3), "propagation")

    # ---------------- cost
    rows = []
    for (mode, cond, rate), g in df.groupby(["mode", "condition", "rate"]):
        rows.append({"mode": mode, "condition": cond, "rate": rate, "prompt_tokens": g.prompt_tokens.mean(),
                     "completion_tokens": g.completion_tokens.mean(), "llm_calls": g.llm_calls.mean(),
                     "tool_calls": g.tool_calls.mean(), "probe_calls": g.probe_calls.mean(),
                     "latency_median_s": g.llm_latency_s.median(), "latency_p95_s": g.llm_latency_s.quantile(0.95)})
    cost = pd.DataFrame(rows)
    save_table(cost.round(3), "cost")

    # ---------------- calibration of the scorer on test memories
    cal_rows, cal_curves = [], {}
    for cond in ("ours_hand", "ours_trained"):
        c = M.calibration_rows(df, cond)
        if len(c):
            cal_rows.append({"scorer": cond, "n": len(c), "brier_score": float(((c.score - c.y) ** 2).mean()),
                             "ece_10bin": M.ece(c.score, c.y), "brier_pre_verification": float(((c.score_pre - c.y) ** 2).mean()),
                             "ece_pre_verification": M.ece(c.score_pre, c.y), "base_rate": float(c.y.mean())})
            cal_curves[cond] = c
    if cal_rows:
        save_table(pd.DataFrame(cal_rows).round(4), "calibration")

    # ---------------- ablations
    abl_dir = RAW / "ablation"
    if abl_dir.exists() and any(abl_dir.glob("*.jsonl")):
        ab = M.load_records(abl_dir)
        ab = M.enrich(ab, BANKS, None)
        base = df[(df.condition == "ours_trained") & (df["mode"] == "fixed") & (df.rate.isin(sorted(ab.rate.unique())))]
        both = pd.concat([ab, base])
        rows = []
        for (cond, rate), g in both.groupby(["condition", "rate"]):
            m, lo, hi = M.boot_ci(g.success.astype(float))
            ref = base[base.rate == rate].set_index(["seed", "task_id"])["success"].astype(float)
            d = M.paired_diff(g.set_index(["seed", "task_id"])["success"].astype(float), ref) if cond != "ours_trained" else (0, 0, 0, len(g))
            used = g.n_used.sum()
            rows.append({"condition": cond, "rate": rate, "success": m, "ci_lo": lo, "ci_hi": hi,
                         "diff_vs_full_pp": 100 * d[0], "diff_ci_lo_pp": 100 * d[1], "diff_ci_hi_pp": 100 * d[2],
                         "reliable_used_frac": g.n_used_reliable.sum() / used if used else np.nan,
                         "probe_calls": g.probe_calls.mean()})
        abl = pd.DataFrame(rows)
        save_table(abl.round(4), "ablation")
        plot_ablation(abl)

    # ---------------- crosscheck (Groq)
    cc = RAW / "crosscheck"
    if cc.exists() and any(cc.glob("*.jsonl")):
        c = M.load_records(cc)
        rows = []
        for (mode, cond, rate), g in c.groupby(["mode", "condition", "rate"]):
            m, lo, hi = M.boot_ci(g.success.astype(float))
            rows.append({"backend": "groq", "mode": mode, "condition": cond, "rate": rate, "n": len(g), "success": m, "ci_lo": lo, "ci_hi": hi})
        save_table(pd.DataFrame(rows).round(4), "crosscheck_groq")

    plot_success(succ)
    if len(prop):
        plot_propagation(prop)
    plot_cost(df)
    if cal_curves:
        plot_calibration(cal_curves)
    (TAB / "summary.json").write_text(json.dumps(summary, indent=1))
    print("done:", summary)


# ---------------------------------------------------------------------------------- plots
def _endlabels(ax, items):
    """Direct labels at line ends, nudged apart to avoid collisions."""
    items = sorted(items, key=lambda z: z[1])
    ys, min_gap = [], 0.035
    for _, y, _, _ in items:
        y2 = max(y, ys[-1] + min_gap) if ys else y
        ys.append(y2)
    for (x, _, txt, col), y in zip(items, ys):
        ax.annotate(txt, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)


def plot_success(succ):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, facecolor=SURF)
    for ax, mode in zip(axes, ("fixed", "growing")):
        style(ax)
        ends = []
        for cond in MAIN:
            s = succ[(succ["mode"] == mode) & (succ.condition == cond)].sort_values("rate")
            if s.empty:
                continue
            x = s.rate * 100
            ax.fill_between(x, s.ci_lo, s.ci_hi, color=COLOR[cond], alpha=0.10, lw=0)
            ax.plot(x, s.success, color=COLOR[cond], lw=2, marker=MARKER[cond], ms=8, label=LABEL[cond],
                    markeredgecolor=SURF, markeredgewidth=1.5)
            ends.append((x.iloc[-1], s.success.iloc[-1], LABEL[cond], COLOR[cond]))
        _endlabels(ax, ends)
        ax.set_title(f"{mode.capitalize()}-bank mode", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Corrupted memories in bank (%)", color=INK2)
        ax.set_xticks([0, 10, 20, 30, 50])
        ax.set_xlim(-2, 72)
    axes[0].set_ylabel("Task success (95% bootstrap CI)", color=INK2)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Task success vs corruption rate", color=INK, x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig(PLOT / "success_vs_corruption.png", dpi=160)
    plt.close(fig)


def plot_propagation(prop):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURF)
    for ax, (col, title) in zip(axes, [("n_erroneous_descendants", "Erroneous descendants of corrupted memories (per run)"),
                                       ("max_depth", "Max propagation depth (per run)")]):
        style(ax)
        for cond in MAIN:
            if cond == "no_memory":
                continue
            p = prop[prop.condition == cond].groupby("rate")[col].agg(["mean", "min", "max"]).reset_index()
            if p.empty:
                continue
            ax.plot(p.rate * 100, p["mean"], color=COLOR[cond], lw=2, marker=MARKER[cond], ms=8, label=LABEL[cond],
                    markeredgecolor=SURF, markeredgewidth=1.5)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Corrupted memories in bank (%)", color=INK2)
        ax.set_xticks([0, 10, 20, 30, 50])
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Error propagation in growing-bank mode (mean over 3 seeds)", color=INK, x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig(PLOT / "propagation_vs_corruption.png", dpi=160)
    plt.close(fig)


def plot_ablation(abl):
    rates = sorted(abl.rate.unique())
    fig, axes = plt.subplots(1, len(rates), figsize=(6.5 * len(rates), 4.6), sharex=True, facecolor=SURF)
    axes = np.atleast_1d(axes)
    for ax, rate in zip(axes, rates):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, lw=0.8)
        a = abl[abl.rate == rate].sort_values("success")
        y = np.arange(len(a))
        cols = ["#6250d6" if c == "ours_trained" else "#9ec5f4" for c in a.condition]
        ax.barh(y, a.success, height=0.62, color=cols, edgecolor=SURF, linewidth=2)
        ax.errorbar(a.success, y, xerr=[a.success - a.ci_lo, a.ci_hi - a.success], fmt="none", ecolor=INK2, lw=1, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([("Full model (trained)" if c == "ours_trained" else c.replace("abl_no_", "– ")) for c in a.condition],
                           fontsize=9, color=INK)
        for yi, v in zip(y, a.success):
            ax.text(v + 0.01, yi + 0.25, f"{100*v:.1f}%", fontsize=8, color=INK2)
        ax.set_title(f"{int(rate*100)}% corrupted, fixed bank", color=INK, fontsize=10, loc="left")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("Task success (95% CI)", color=INK2)
    fig.suptitle("Signal ablations of Ours (trained): one signal removed at a time", color=INK, x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig(PLOT / "ablation.png", dpi=160)
    plt.close(fig)


def plot_cost(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURF)
    for ax, mode in zip(axes, ("fixed", "growing")):
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        g = df[df["mode"] == mode].groupby("condition").agg(tokens=("tokens", "mean"), success=("success", "mean"),
                                                            tools=("all_tool_calls", "mean")).reset_index()
        for _, r in g.iterrows():
            ax.scatter(r.tokens / 1000, r.success, s=90, color=COLOR[r.condition], marker=MARKER[r.condition],
                       edgecolor=SURF, linewidth=1.5, zorder=3, label=LABEL[r.condition])
            ax.annotate(f"{LABEL[r.condition]} ({r.tools:.1f} tool calls)", (r.tokens / 1000, r.success), xytext=(7, 4),
                        textcoords="offset points", fontsize=7.5, color=INK2)
        ax.set_xlabel("LLM tokens per task (thousands)", color=INK2)
        ax.set_title(f"{mode.capitalize()}-bank mode, all corruption rates pooled", color=INK, fontsize=10, loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Task success", color=INK2)
    fig.suptitle("Cost vs success", color=INK, x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig(PLOT / "cost_vs_success.png", dpi=160)
    plt.close(fig)


def plot_calibration(curves):
    fig, ax = plt.subplots(figsize=(5.8, 5.4), facecolor=SURF)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="Perfect calibration")
    edges = np.linspace(0, 1, 11)
    for cond, c in curves.items():
        b = np.clip(np.digitize(c.score, edges) - 1, 0, 9)
        agg = c.assign(b=b).groupby("b").agg(p=("score", "mean"), y=("y", "mean"), n=("y", "size")).reset_index()
        agg = agg[agg.n >= 5]
        ax.plot(agg.p, agg.y, color=COLOR[cond], lw=2, marker=MARKER[cond], ms=8, label=f"{LABEL[cond]} (n={len(c)})",
                markeredgecolor=SURF, markeredgewidth=1.5)
    ax.set_xlabel("Predicted reliability score", color=INK2)
    ax.set_ylabel("Observed fraction reliable (ground truth)", color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("Scorer calibration on test-split memories", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(PLOT / "scorer_calibration.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
