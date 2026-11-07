"""Interpretable weighted reliability score + use/verify/revise/reject policy (LR §6.2)."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .signals import FEATURES, SIGNAL_GROUPS

HAND_WEIGHTS = {  # documented a-priori weights (no data used)
    "bias": -2.0, "outcome": 1.5, "confidence": 1.0, "recency": 1.0, "context_sim": 1.5, "version_match": 1.5,
    "contradiction": -5.0, "verif_pass": 2.5, "verif_fail": -4.0, "verif_unchecked": 0.0,
}
HAND_THRESHOLDS = {"use": 0.70, "reject": 0.30, "caution": 0.50}


@dataclass
class Scorer:
    weights: dict
    thresholds: dict
    calibrator: dict | None = None  # {"type": "platt", "a":..., "b":...}
    hard_gate: bool = True
    dropped: list = field(default_factory=list)  # ablated signal groups
    name: str = "hand"

    # ---- calibration ---------------------------------------------------------
    def calibrate(self, conf01: float) -> float:
        c = self.calibrator
        if not c:
            return conf01
        if c["type"] == "platt":
            return 1 / (1 + math.exp(-(c["a"] * conf01 + c["b"])))
        if c["type"] == "isotonic":
            xs, ys = c["x"], c["y"]
            if conf01 <= xs[0]:
                return ys[0]
            for i in range(1, len(xs)):
                if conf01 <= xs[i]:
                    t = (conf01 - xs[i - 1]) / max(1e-12, xs[i] - xs[i - 1])
                    return ys[i - 1] + t * (ys[i] - ys[i - 1])
            return ys[-1]
        raise ValueError(c["type"])

    def active_features(self) -> list[str]:
        drop = {f for g in self.dropped for f in SIGNAL_GROUPS[g]}
        return [f for f in FEATURES if f not in drop]

    def score(self, x: dict) -> float:
        z = self.weights.get("bias", 0.0)
        for f in self.active_features():
            if f == "contradiction" and self.hard_gate:
                continue  # handled by the gate, not the weighted score
            z += self.weights.get(f, 0.0) * x[f]
        return 1 / (1 + math.exp(-z))

    def gated(self, x: dict) -> bool:
        return self.hard_gate and "contradiction" not in self.dropped and x["contradiction"] >= 1

    # ---- io --------------------------------------------------------------------
    def to_dict(self):
        return asdict(self)

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        (d / "scorer.json").write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "Scorer":
        p = path / "scorer.json" if path.is_dir() else path
        return cls(**json.loads(p.read_text()))

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def hand_scorer() -> Scorer:
    return Scorer(dict(HAND_WEIGHTS), dict(HAND_THRESHOLDS), None, True, [], "hand")


def decide_pre(scorer: Scorer, x: dict) -> str:
    """Decision before verification (verification feature = unchecked)."""
    if scorer.gated(x):
        return "reject"
    s = scorer.score(x)
    if s >= scorer.thresholds["use"]:
        return "use"
    if s >= scorer.thresholds["reject"]:
        return "verify" if "verification" not in scorer.dropped else ("use" if s >= scorer.thresholds["caution"] else "reject")
    return "reject"


def decide_post(scorer: Scorer, x_after: dict, ver: dict) -> str:
    """Decision after verification. fail + new evidence -> revise; fail w/o evidence -> reject;
    pass -> use if score clears 'use'; unchecked -> use only if score clears 'caution'."""
    if scorer.gated(x_after):
        return "reject"
    if ver["status"] == "fail":
        return "revise" if ver["corrections"] else "reject"
    s = scorer.score(x_after)
    if ver["status"] == "pass":
        return "use" if s >= scorer.thresholds["caution"] else "reject"
    return "use" if s >= scorer.thresholds["caution"] else "reject"
