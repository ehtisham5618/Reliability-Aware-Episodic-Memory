"""Deterministic scripted LLM for offline tests of the agent graph (never used for results).

Behaviour: if a retrieved memory for the same family is in the prompt, copy its actions with the
task's own entity values only where the memory's entity matches (i.e. it *trusts memory*);
otherwise act like the oracle. This lets tests check that corrupted memories propagate."""
from __future__ import annotations

import json
import re

from ..env.oracle import oracle_actions
from ..env.tasks import generate_splits
from .client import LLMResponse

_TASKS = {t.text: t for ts in generate_splits().values() for t in ts}


class FakeLLM:
    def __init__(self, follow_memory: bool = True):
        self.follow_memory = follow_memory
        self.calls = 0

    def chat(self, messages, seed=0, tools=None):
        self.calls += 1
        user = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        if len(messages) == 1:  # reflection / lesson prompt
            return LLMResponse("Use the documented conventions and look up ids first.", 50, 15, 0.01, False, "fake")
        if len(messages) > 2:  # after first turn: declare done
            return LLMResponse("", 100, 20, 0.01, False, "fake",
                               [{"id": "f1", "name": "finish", "arguments": json.dumps({"status": "done", "confidence": 80})}])
        m = re.search(r"Task: (.*)", user)
        task = _TASKS[m.group(1).strip()]
        acts = oracle_actions(task)
        if self.follow_memory and "Relevant past experiences" in user:
            mem_line = user.split("Relevant past experiences from your memory:\n", 1)[1].split("\n")[0]
            if task.text in mem_line or any(e in mem_line.lower() for e in task.entities()):
                am = re.search(r"actions: (.*?) \| outcome", mem_line)
                if am:
                    parsed = []
                    for call in re.finditer(r"(\w+)\((\{.*?\})\)(?:;|$)", am.group(1)):
                        try:
                            parsed.append({"tool": call.group(1), "args": json.loads(call.group(2))})
                        except json.JSONDecodeError:
                            pass
                    commit = [a for a in parsed if a["tool"] in ("book_room", "log_expense", "config", "submit_answer")]
                    if commit and mem_line.split("task: ")[1].split(" | ")[0] == task.text:
                        acts = parsed
        calls = [{"id": f"c{i}", "name": a["tool"], "arguments": json.dumps(a["args"])} for i, a in enumerate(acts)]
        return LLMResponse("", 200, 60, 0.02, False, "fake", calls)
