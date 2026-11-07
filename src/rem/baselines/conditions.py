"""Experimental conditions: baselines, our method (hand / trained), and ablations."""
from __future__ import annotations

from dataclasses import dataclass

from ..scorer.signals import SIGNAL_GROUPS


@dataclass(frozen=True)
class Condition:
    name: str
    retrieval: bool = True
    view: str = "raw"  # raw | lessons | ours
    filter: str = "none"  # none | outcome | xiong | ours
    scorer: str | None = None  # hand | trained | ablation-<group> | nogate
    pitfalls: bool = False


NO_MEMORY = Condition("no_memory", retrieval=False)
PLAIN = Condition("plain")
REFLEXION = Condition("reflexion", view="lessons")
OUTCOME = Condition("outcome_only", filter="outcome")
XIONG = Condition("xiong", filter="xiong")
OURS_HAND = Condition("ours_hand", view="ours", filter="ours", scorer="hand", pitfalls=True)
OURS_TRAINED = Condition("ours_trained", view="ours", filter="ours", scorer="trained", pitfalls=True)

MAIN = [NO_MEMORY, PLAIN, REFLEXION, OUTCOME, XIONG, OURS_HAND, OURS_TRAINED]
ABLATIONS = [Condition(f"abl_no_{g}", view="ours", filter="ours", scorer=f"ablation-{g}", pitfalls=True)
             for g in SIGNAL_GROUPS] + [Condition("abl_no_hardgate", view="ours", filter="ours", scorer="nogate", pitfalls=True)]
ALL = {c.name: c for c in MAIN + ABLATIONS}

# Xiong-style quality-aware add/delete (evaluator = agent-observable signals at test time)
XIONG_ADD_MIN_CONF = 70
XIONG_DELETE_MIN_USES = 2
XIONG_DELETE_MAX_FAILRATE = 0.5
