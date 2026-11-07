"""Versioned evidence store (current API docs + changelog + partial fact sheet).

Used only by the contradiction signal. It documents all signature changes but only ~50% of
entity facts and *none* of the silent conventions (date order, timeout units), so
contradiction is a partial, not a perfect, detector."""
from __future__ import annotations

import random

from ..env.world import CONFIG_KEYS, DEPARTMENTS, SERVICES, WORLD
from .schema import _expected_args

EVIDENCE_SEED = 99


def build_evidence(version: str = "v2", coverage: float = 0.5, seed: int = EVIDENCE_SEED) -> dict:
    """Map claim key -> documented value."""
    rng = random.Random(seed)
    ev = {
        "book_room_args": _expected_args("book_room", version),
        "config_key:conn": CONFIG_KEYS[version]["conn"],
        "config_key:timeout": CONFIG_KEYS[version]["timeout"],
    }
    for name in sorted(WORLD.rooms):
        if rng.random() < coverage:
            ev[f"room_id:{name}"] = WORLD.room_id(name, version)
    for office in sorted(WORLD.office_region):
        if rng.random() < coverage:
            ev[f"region_code:{office}"] = WORLD.region_code(office)
    for d in DEPARTMENTS:
        if rng.random() < coverage:
            ev[f"dept_code:{d}"] = WORLD.dept_code[d]
    for s in SERVICES:
        if rng.random() < coverage:
            ev[f"service_id:{s}"] = WORLD.service_id[s]
    return ev


EVIDENCE = build_evidence()
