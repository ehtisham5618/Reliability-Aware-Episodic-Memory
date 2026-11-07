from __future__ import annotations

import json


MAX_TURNS = 8

SYSTEM = """You are an operations agent that completes back-office tasks using the provided tools.
Current day: {day}. Active API version: {version}.
{docs}
- You have at most {turns} turns; each of your replies is one turn. Call several tools in one turn when they do not depend on each other.
- Do not repeat a state-changing action (booking, config write, expense) that already succeeded.
- When the task is complete, call finish(status="done", confidence=0-100). If it cannot be completed, call finish(status="failed", confidence=0-100)."""

REFLECT_RETRY = """Your previous attempt at this task did not succeed.
Task: {task}
Trajectory:
{traj}
In 2-3 sentences, reflect on what went wrong and what you will do differently next time. Reply with plain text only."""

LESSON = """Task: {task}
Trajectory:
{traj}
Outcome: {outcome}{feedback}
Write the reusable lesson from this episode in 1-2 sentences for a future agent doing similar tasks: the exact conventions, formats, ids or values that worked (or that caused failure). Reply with plain text only."""


def system_prompt(day: int, version: str, docs: str | None = None, turns: int = MAX_TURNS) -> str:
    return SYSTEM.format(day=day, version=version, docs=("\n" + docs + "\n") if docs else "", turns=turns)


def fmt_traj(actions: list, observations: list) -> str:
    lines = []
    for a, o in zip(actions, observations):
        lines.append(f"- {a['tool']}({json.dumps(a.get('args') or {}, separators=(',', ':'))}) -> {o}")
    return "\n".join(lines) if lines else "(no actions)"


def fmt_observations(actions: list, observations: list, turns_left: int) -> str:
    return f"Observations:\n{fmt_traj(actions, observations)}\nTurns left: {turns_left}"
