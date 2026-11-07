"""LangGraph agent loop:

retrieve -> score -> policy -> [verify] -> act (ReAct turns, loop) -> self_check
    -> [retry_reflect -> act]  (one Reflexion-style retry, same budget for all conditions)
    -> reflect (lesson) -> write -> END

Every decision is logged (retrieved ids, signal vectors, policy action per memory, verification
outcome, injected memory ids, tool calls, tokens, latency, checker result) so the provenance
graph and all metrics are rebuilt from logs only.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..baselines.conditions import (XIONG_ADD_MIN_CONF, XIONG_DELETE_MAX_FAILRATE, XIONG_DELETE_MIN_USES, Condition)
from ..env.tasks import PRIVILEGED_DOCS, Task, check
from ..env.tools import TOOL_SCHEMAS, ToolEnv
from ..llm.client import LLMClient
from ..memory.schema import Episode, extract_claims, ground_truth_reliable
from ..memory.store import MemoryStore
from ..scorer.model import Scorer, decide_post, decide_pre
from ..scorer.signals import signal_vector, verify
from . import prompts as P

COMMIT_TOOLS = {"booking": {"book_room"}, "config": {"config"}, "expense": {"log_expense"}, "aggregate": {"submit_answer"}}


@dataclass
class RunContext:
    llm: LLMClient
    store: MemoryStore | None
    condition: Condition
    mode: str  # fixed | growing | build
    seed: int
    run_tag: str
    scorer: Scorer | None = None
    k: int = 4
    max_turns: int = P.MAX_TURNS
    max_attempts: int = 2
    xiong_stats: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    rev_counter: int = 0


class S(TypedDict, total=False):
    task: Task
    retrieved: list  # [(id, sim)]
    cands: list  # per-memory dicts {id, sim, x, decision, ...}
    used: list  # [Episode] to inject
    revised: list  # [Episode] revisions created
    pitfalls: list  # [str]
    probe_calls: int
    attempt: int
    turn: int
    messages: list
    env: Any
    actions: list
    observations: list
    final: dict | None
    reflections: list
    attempts_log: list
    perceived: str
    confidence: float
    lesson: str
    llm: dict  # token/latency accounting
    written: str | None


# ---------------------------------------------------------------------------------- helpers
def _acct(state: S, r):
    a = state.setdefault("llm", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "cached": 0})
    a["calls"] += 1
    a["prompt_tokens"] += r.prompt_tokens
    a["completion_tokens"] += r.completion_tokens
    a["latency_s"] += r.latency_s
    a["cached"] += int(r.cached)


def _memory_block(ctx: RunContext, state: S) -> str:
    used, pits = state.get("used", []), state.get("pitfalls", [])
    if not used and not pits:
        return ""
    if ctx.condition.view == "lessons":
        lines = [f"- {e.lesson}" for e in used if e.lesson]
        return "Lessons from your past experience:\n" + "\n".join(lines) if lines else ""
    out = []
    if used:
        out.append("Relevant past experiences from your memory:")
        for e in used:
            tag = ""
            c = next((c for c in state.get("cands", []) if c["id"] in (e.id, e.revised_from)), None)
            if ctx.condition.view == "ours" and c:
                if c.get("decision") == "revise":
                    tag = f" [REVISED with fresh tool evidence: {c.get('corrections')}]"
                elif c.get("verification") == "pass":
                    tag = " [VERIFIED by tool check]"
            out.append(e.render() + tag)
    if pits:
        out.append("Known unreliable memories - do NOT repeat these:")
        out.extend(f"- {p}" for p in pits)
    return "\n".join(out)


def _textual_tool_calls(content: str, turn: int) -> list:
    """gpt-oss sometimes writes a tool call as JSON text instead of a native call; accept the
    common shapes {"tool"|"name"|"action": ..., "args"|"arguments"|"parameters": {...}}."""
    if not content or "{" not in content:
        return []
    out, dec, i = [], json.JSONDecoder(), 0
    while True:
        i = content.find("{", i)
        if i < 0:
            break
        try:
            obj, end = dec.raw_decode(content, i)
        except json.JSONDecodeError:
            i += 1
            continue
        i = end
        if isinstance(obj, dict):
            name = obj.get("tool") or obj.get("name") or obj.get("action")
            args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if isinstance(name, str) and isinstance(args, dict):
                out.append({"id": f"text-{turn}-{len(out)}", "name": name, "arguments": json.dumps(args)})
            elif obj.get("status") in ("done", "failed"):
                out.append({"id": f"text-{turn}-{len(out)}", "name": "finish",
                            "arguments": json.dumps({"status": obj["status"], "confidence": obj.get("confidence", 50)})})
    return out


def _apply_corrections(ep: Episode, corr: dict, new_id: str) -> Episode:
    old_vals = {c["key"]: c["value"] for c in ep.claims}
    pairs = [(str(old_vals[k]), str(v)) for k, v in corr.items() if k in old_vals and not isinstance(v, list)]
    blob = json.dumps([ep.actions, ep.observations])
    lesson = ep.lesson
    for a, b in pairs:
        blob = blob.replace(json.dumps(a)[1:-1], json.dumps(b)[1:-1])
        lesson = re.sub(re.escape(a), b, lesson)
    acts, obs = json.loads(blob)
    r = ep.clone(id=new_id, actions=acts, observations=obs, lesson=lesson, revised_from=ep.id, parent_ids=[ep.id],
                 status="active", corrupted=None, corrupted_from=None)
    r.claims = extract_claims(r.family, r.params, r.actions, r.observations, r.lesson)
    r.provenance = {"revision_of": ep.id, "corrections": corr}
    r.true_outcome = None
    return r


# ---------------------------------------------------------------------------------- nodes
def build_graph(ctx: RunContext):
    cond = ctx.condition

    def retrieve(state: S) -> S:
        if not cond.retrieval or ctx.store is None or len(ctx.store) == 0:
            return {"retrieved": [], "cands": [], "used": [], "pitfalls": [], "probe_calls": 0}
        hits = ctx.store.retrieve(state["task"].text, k=ctx.k)
        return {"retrieved": [(e.id, round(s, 4)) for e, s in hits], "probe_calls": 0}

    def score(state: S) -> S:
        task = state["task"]
        cands = []
        for eid, sim in state.get("retrieved", []):
            e = ctx.store.get(eid)
            c = {"id": eid, "sim": sim, "status_before": e.status}
            if cond.filter == "ours":
                x = signal_vector(task, e, sim, "unchecked", calibrator=ctx.scorer.calibrate if ctx.scorer.calibrator else None)
                c["conflicts"] = x.pop("_conflicts")
                c["x"] = x
                c["score_pre"] = round(ctx.scorer.score(x), 4)
            cands.append(c)
        return {"cands": cands}

    def policy(state: S) -> S:
        cands, used, pits = state["cands"], [], []
        for c in cands:
            e = ctx.store.get(c["id"])
            if cond.filter == "none":
                c["decision"] = "use"
            elif cond.filter == "outcome":
                c["decision"] = "use" if e.outcome == "success" else "reject"
            elif cond.filter == "xiong":
                ok = e.outcome == "success" and e.confidence_raw >= XIONG_ADD_MIN_CONF
                c["decision"] = "use" if ok else "reject"
            elif cond.filter == "ours":
                if e.status != "active":  # flagged earlier in this run (rejected / superseded by a revision)
                    c["decision"] = "reject"
                    c["reason"] = f"previously_{e.status}"
                else:
                    c["decision"] = decide_pre(ctx.scorer, c["x"])
                    if c["decision"] == "reject" and ctx.scorer.gated(c["x"]):
                        c["reason"] = "contradiction_gate"
            if c["decision"] == "use":
                used.append(e)
        if cond.filter == "ours" and cond.pitfalls:
            pits = _pitfalls(state, cands)
        return {"cands": cands, "used": used, "pitfalls": pits}

    def _pitfalls(state, cands):
        pits = []
        for c in cands:
            if c["decision"] != "reject":
                continue
            e = ctx.store.get(c["id"])
            if c.get("conflicts"):
                k = c["conflicts"][0]
                pits.append(f"[{e.id}] claims {k['key']}={k['memory']!r}, which contradicts current documentation ({k['evidence']!r}).")
            elif c.get("verification") == "fail":
                pits.append(f"[{e.id}] failed an independent tool check ({', '.join(c.get('checked', []))}).")
            elif e.outcome == "fail" and e.lesson:
                pits.append(f"[{e.id}] recorded failure: {e.lesson}")
        return pits[:2]

    def verify_node(state: S) -> S:
        task, cands = state["task"], state["cands"]
        probes = 0
        revised = list(state.get("revised", []))
        verified_claims = {}
        for c in cands:
            if c.get("decision") != "verify":
                continue
            e = ctx.store.get(c["id"])
            ver = verify(e, day=task.day)
            probes += ver["probes"]
            c["verification"], c["checked"], c["corrections"] = ver["status"], ver["checked"], ver["corrections"]
            x2 = signal_vector(task, e, c["sim"], ver["status"], calibrator=ctx.scorer.calibrate if ctx.scorer.calibrator else None,
                               verified_claims=verified_claims)
            c["conflicts"] = x2.pop("_conflicts") or c.get("conflicts")
            c["x_post"] = x2
            c["score_post"] = round(ctx.scorer.score(x2), 4)
            c["decision"] = decide_post(ctx.scorer, x2, ver)
            if ver["status"] == "pass":
                verified_claims.update({cl["key"]: cl["value"] for cl in e.claims})
            if c["decision"] == "revise":
                rid = f"{e.id}~rev@{task.task_id}"  # deterministic (thread-order independent) -> stable prompts/cache keys
                r = _apply_corrections(e, ver["corrections"], rid)
                r.provenance.update({"run": ctx.run_tag, "task": task.task_id})
                c["revised_id"] = rid
                revised.append(r)
        used = []
        for c in cands:
            if c["decision"] == "use":
                used.append(ctx.store.get(c["id"]))
            elif c["decision"] == "revise":
                used.append(next(r for r in revised if r.id == c["revised_id"]))
        pits = _pitfalls(state, cands) if cond.pitfalls else []
        return {"cands": cands, "used": used, "revised": revised, "probe_calls": probes, "pitfalls": pits}

    def start_attempt(state: S) -> S:
        task = state["task"]
        docs = PRIVILEGED_DOCS[task.version] if ctx.mode == "build" else None
        user = f"Task: {task.text}"
        mem = _memory_block(ctx, state)
        if mem:
            user += "\n\n" + mem
        if state.get("reflections"):
            user += "\n\nReflection from your previous attempt:\n" + state["reflections"][-1]
        msgs = [{"role": "system", "content": P.system_prompt(task.day, task.version, docs, ctx.max_turns)},
                {"role": "user", "content": user}]
        return {"messages": msgs, "env": ToolEnv(day=task.day), "turn": 0, "actions": [], "observations": [],
                "final": None, "attempt": state.get("attempt", 0) + 1}

    def act(state: S) -> S:
        """One ReAct turn: the model reasons (hidden reasoning channel) and emits native tool calls;
        the environment executes them and returns observations as tool messages."""
        msgs = list(state["messages"])
        r = ctx.llm.chat(msgs, seed=ctx.seed, tools=TOOL_SCHEMAS)
        _acct(state, r)
        env: ToolEnv = state["env"]
        acts, obs = list(state["actions"]), list(state["observations"])
        final = None
        left = ctx.max_turns - state["turn"] - 1
        tool_calls = r.tool_calls or _textual_tool_calls(r.content, state["turn"])
        if tool_calls:
            msgs.append({"role": "assistant", "content": r.content or "",
                         "tool_calls": [{"id": tc["id"], "type": "function",
                                         "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls]})
            for tc in tool_calls[:10]:
                # gpt-oss (harmony) sometimes leaks channel tags into the name, e.g. 'finish<|channel|>commentary'
                name = str(tc["name"]).split("<|")[0].strip()
                try:
                    args = json.loads(tc["arguments"] or "{}")
                    if not isinstance(args, dict):
                        raise ValueError
                except (json.JSONDecodeError, ValueError):
                    args, o = {}, "error: arguments must be a JSON object"
                else:
                    if name == "finish":
                        final = {"status": args.get("status"), "confidence": args.get("confidence")}
                        o = "ok"
                    else:
                        o = env.call(name, args)
                        acts.append({"tool": name, "args": args})
                        obs.append(o)
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": o})
            if final is None:
                msgs.append({"role": "user", "content": f"Turns left: {left}"})
        else:
            note = "Use the tools to make progress; call finish when the task is complete."
            if r.error:
                note = "Your previous reply could not be parsed. " + note
            if r.content:
                msgs.append({"role": "assistant", "content": r.content})
            msgs.append({"role": "user", "content": f"{note} Turns left: {left}"})
        if final is not None and final.get("status") not in ("done", "failed"):
            final = {"status": "failed", "confidence": final.get("confidence")}
        return {"messages": msgs, "actions": acts, "observations": obs, "turn": state["turn"] + 1, "final": final,
                "llm": state["llm"]}

    def after_act(state: S) -> str:
        if state.get("final") is not None or state["turn"] >= ctx.max_turns:
            return "self_check"
        return "act"

    def self_check(state: S) -> S:
        """Agent-observable success: agent declared done AND its last commit call returned ok."""
        task = state["task"]
        final = state.get("final") or {}
        commits = [o for a, o in zip(state["actions"], state["observations"]) if a["tool"] in COMMIT_TOOLS[task.family]]
        # declared done, OR ran out of turns without declaring failure, after a commit that returned ok
        ok = final.get("status") != "failed" and bool(commits) and not commits[-1].startswith("error")
        try:
            conf = float(final.get("confidence", 0 if not ok else 50))
        except (TypeError, ValueError):
            conf = 50.0
        passed, reason = check(task, state["env"])
        log = list(state.get("attempts_log", []))
        log.append({"attempt": state["attempt"], "turns": state["turn"], "perceived": "success" if ok else "fail",
                    "confidence": conf, "checker": passed, "checker_reason": reason,
                    "actions": state["actions"], "observations": state["observations"], "tool_calls": len(state["env"].calls)})
        return {"perceived": "success" if ok else "fail", "confidence": max(0.0, min(100.0, conf)), "attempts_log": log}

    def after_check(state: S) -> str:
        if state["perceived"] == "fail" and state["attempt"] < ctx.max_attempts:
            return "retry_reflect"
        return "reflect"

    def retry_reflect(state: S) -> S:
        task = state["task"]
        traj = P.fmt_traj(state["actions"], state["observations"])
        r = ctx.llm.chat([{"role": "user", "content": P.REFLECT_RETRY.format(task=task.text, traj=traj)}], seed=ctx.seed)
        _acct(state, r)
        return {"reflections": list(state.get("reflections", [])) + [r.content.strip()[:600]], "llm": state["llm"]}

    def reflect(state: S) -> S:
        if ctx.mode == "fixed":
            return {"lesson": ""}
        task = state["task"]
        acted = next((a for a in reversed(state["attempts_log"]) if a["actions"]), state["attempts_log"][-1])
        traj = P.fmt_traj(acted["actions"], acted["observations"])
        fb = ""
        outcome = state["perceived"]
        if ctx.mode == "build":  # build phase has an evaluator (Reflexion-style external feedback)
            last = state["attempts_log"][-1]
            outcome = "success" if last["checker"] else "fail"
            fb = f"\nEvaluator feedback: {'passed' if last['checker'] else 'FAILED - ' + last['checker_reason']}"
        r = ctx.llm.chat([{"role": "user", "content": P.LESSON.format(task=task.text, traj=traj, outcome=outcome, feedback=fb)}],
                         seed=ctx.seed)
        _acct(state, r)
        return {"lesson": r.content.strip()[:600], "llm": state["llm"]}

    def write(state: S) -> S:
        if ctx.mode == "fixed" or ctx.store is None:
            return {"written": None}
        task = state["task"]
        last = state["attempts_log"][-1]
        true_out = "success" if last["checker"] else "fail"
        outcome = true_out if ctx.mode == "build" else state["perceived"]
        acted = next((a for a in reversed(state["attempts_log"]) if a["actions"]), last)
        ep = Episode(id=f"{ctx.run_tag}:{task.task_id}", task_id=task.task_id, task_text=task.text, family=task.family,
                     params=task.p, env_version=task.version, timestamp_day=task.day, actions=acted["actions"],
                     observations=acted["observations"], outcome=outcome, confidence_raw=state["confidence"],
                     lesson=state.get("lesson", ""), parent_ids=[e.id for e in state.get("used", [])],
                     provenance={"run": ctx.run_tag, "condition": cond.name, "seed": ctx.seed}, true_outcome=true_out)
        ep.claims = extract_claims(ep.family, ep.params, ep.actions, ep.observations, ep.lesson)
        ep.evidence = [{"source": "tool", "call": a, "obs": o} for a, o in zip(ep.actions, ep.observations) if a["tool"] == "kb_lookup"]
        written = None
        with ctx.lock:
            # ours: persist policy flags + revisions (growing mode only)
            if cond.filter == "ours":
                for c in state.get("cands", []):
                    if c.get("decision") in ("reject", "revise") and ctx.store.get(c["id"]).status == "active":
                        ctx.store.set_status(c["id"], "rejected" if c["decision"] == "reject" else "revised")
                for r in state.get("revised", []):
                    if r.id not in ctx.store.episodes:
                        ctx.store.add(r)
            if cond.filter == "xiong":
                # utility-based deletion of used memories
                for e in state.get("used", []):
                    st = ctx.xiong_stats.setdefault(e.id, [0, 0])
                    st[0] += 1
                    st[1] += int(state["perceived"] == "fail")
                    if st[0] >= XIONG_DELETE_MIN_USES and st[1] / st[0] > XIONG_DELETE_MAX_FAILRATE:
                        ctx.store.set_status(e.id, "deleted")
                errs = any(o.startswith("error") for o in ep.observations)
                add = ep.outcome == "success" and ep.confidence_raw >= XIONG_ADD_MIN_CONF and not errs
            else:
                add = True
            if add:
                ctx.store.add(ep)
                written = ep.id
        return {"written": written}

    g = StateGraph(S)
    for name, fn in [("retrieve", retrieve), ("score", score), ("policy", policy), ("verify", verify_node),
                     ("start_attempt", start_attempt), ("act", act), ("self_check", self_check),
                     ("retry_reflect", retry_reflect), ("reflect", reflect), ("write", write)]:
        g.add_node(name, fn)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "score")
    g.add_edge("score", "policy")
    g.add_conditional_edges("policy", lambda s: "verify" if any(c.get("decision") == "verify" for c in s.get("cands", [])) else "start_attempt",
                            {"verify": "verify", "start_attempt": "start_attempt"})
    g.add_edge("verify", "start_attempt")
    g.add_edge("start_attempt", "act")
    g.add_conditional_edges("act", after_act, {"act": "act", "self_check": "self_check"})
    g.add_conditional_edges("self_check", after_check, {"retry_reflect": "retry_reflect", "reflect": "reflect"})
    g.add_edge("retry_reflect", "start_attempt")
    g.add_edge("reflect", "write")
    g.add_edge("write", END)
    return g.compile()


def run_task(graph, ctx: RunContext, task: Task) -> dict:
    t0 = time.time()
    st = graph.invoke({"task": task, "llm": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "cached": 0}},
                      {"recursion_limit": 60})
    last = st["attempts_log"][-1]
    used = st.get("used", [])
    cands = st.get("cands", [])
    for c in cands:  # ground-truth reliability of every retrieved memory (analysis only)
        e = ctx.store.get(c["id"]) if ctx.store and c["id"] in ctx.store.episodes else None
        if e is not None:
            c["gt_reliable"] = ground_truth_reliable(e, task.version)
            c["corrupted"] = e.corrupted
            c["family_match"] = e.family == task.family
            c["entity_overlap"] = bool(task.entities() & e.entities())
    tool_calls = sum(a["tool_calls"] for a in st["attempts_log"])
    return {
        "run": ctx.run_tag, "condition": ctx.condition.name, "mode": ctx.mode, "seed": ctx.seed,
        "task_id": task.task_id, "family": task.family,
        "success": bool(last["checker"]), "checker_reason": last["checker_reason"],
        "perceived": st["perceived"], "confidence": st["confidence"], "attempts": len(st["attempts_log"]),
        "retrieved": st.get("retrieved", []), "cands": cands,
        "used_ids": [e.id for e in used], "used_gt_reliable": [ground_truth_reliable(e, task.version) for e in used],
        "used_corrupted": [e.corrupted for e in used], "pitfalls": st.get("pitfalls", []),
        "revised_ids": [r.id for r in st.get("revised", [])],
        "written": st.get("written"), "lesson": st.get("lesson", ""),
        "llm_calls": st["llm"]["calls"], "prompt_tokens": st["llm"]["prompt_tokens"],
        "completion_tokens": st["llm"]["completion_tokens"], "llm_latency_s": round(st["llm"]["latency_s"], 3),
        "cached_calls": st["llm"]["cached"], "tool_calls": tool_calls, "probe_calls": st.get("probe_calls", 0),
        "attempts_log": st["attempts_log"], "wall_s": round(time.time() - t0, 3),
    }
