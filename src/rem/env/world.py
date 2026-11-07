"""Deterministic office-operations world with versioned facts and tool behaviour.

The world has a simulated clock (days). Tool/fact version switches from v1 to v2 at
``V2_START_DAY``. All entity data is generated from a fixed seed so every run sees the
same world.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

WORLD_SEED = 1234
V2_START_DAY = 100
CURRENT_DAY = 200  # val/test tasks run "now", under v2

OFFICES = [
    ("Lahore", "East"), ("Karachi", "South"), ("Islamabad", "North"), ("Peshawar", "North"),
    ("Quetta", "West"), ("Multan", "Central"), ("Faisalabad", "Central"), ("Sialkot", "East"),
    ("Hyderabad", "South"), ("Gilgit", "North"), ("Gwadar", "West"), ("Rawalpindi", "North"),
]
REGION_CODES = {"North": "RG-N4", "South": "RG-S2", "East": "RG-E7", "West": "RG-W3", "Central": "RG-C5"}
ROOM_NAMES = [
    "Cedar", "Maple", "Oak", "Pine", "Birch", "Willow", "Aspen", "Elm", "Juniper", "Spruce",
    "Laurel", "Magnolia", "Sequoia", "Banyan", "Olive", "Cypress", "Acacia", "Rowan", "Hazel",
    "Alder", "Poplar", "Linden", "Sycamore", "Chestnut", "Walnut", "Ash", "Fir", "Yew", "Teak", "Ebony",
]
SERVICES = ["billing", "auth", "search", "payments", "notifications", "analytics", "inventory", "reporting"]
DEPARTMENTS = ["Finance", "Marketing", "Engineering", "Sales", "HR", "Legal", "Operations", "Research"]
FX_V1 = {"PKR": 0.00355, "EUR": 1.08, "GBP": 1.27, "AED": 0.2723, "SAR": 0.2667}
FX_V2 = {"PKR": 0.00361, "EUR": 1.13, "GBP": 1.33, "AED": 0.2741, "SAR": 0.2652}

CONFIG_KEYS = {
    "v1": {"conn": "max_conn", "timeout": "timeout", "timeout_unit": "s"},
    "v2": {"conn": "connection_limit", "timeout": "timeout_ms", "timeout_unit": "ms"},
}


def version_at(day: int) -> str:
    return "v1" if day < V2_START_DAY else "v2"


@dataclass
class Room:
    name: str
    office: str
    room_id: dict  # version -> id
    capacity: dict  # version -> capacity


@dataclass
class WorldData:
    rooms: dict[str, Room] = field(default_factory=dict)
    office_region: dict[str, str] = field(default_factory=dict)
    service_id: dict[str, str] = field(default_factory=dict)
    dept_code: dict[str, str] = field(default_factory=dict)

    # ---- fact accessors (versioned) -------------------------------------------------
    def room_id(self, room: str, version: str) -> str:
        return self.rooms[room].room_id[version]

    def capacity(self, room: str, version: str) -> int:
        return self.rooms[room].capacity[version]

    def region_code(self, office: str) -> str:
        return REGION_CODES[self.office_region[office]]

    def rooms_in(self, office: str) -> list[str]:
        return sorted(r.name for r in self.rooms.values() if r.office == office)

    def fx(self, cur: str, version: str) -> float:
        return (FX_V1 if version == "v1" else FX_V2)[cur]

    def all_room_ids(self, version: str) -> set[str]:
        return {r.room_id[version] for r in self.rooms.values()}


def build_world(seed: int = WORLD_SEED) -> WorldData:
    rng = random.Random(seed)
    w = WorldData()
    w.office_region = dict(OFFICES)
    offices = [o for o, _ in OFFICES]
    used_ids: set[str] = set()

    def new_id() -> str:
        while True:
            rid = f"R-{rng.randint(100, 999)}"
            if rid not in used_ids:
                used_ids.add(rid)
                return rid

    for i, name in enumerate(ROOM_NAMES):
        office = offices[i % len(offices)]
        v1_id = new_id()
        cap_v1 = rng.randint(4, 40)
        renumber = rng.random() < 0.2
        recap = rng.random() < 0.2
        v2_id = new_id() if renumber else v1_id
        cap_v2 = max(4, cap_v1 + rng.choice([-6, -4, 4, 6, 8])) if recap else cap_v1
        w.rooms[name] = Room(name, office, {"v1": v1_id, "v2": v2_id}, {"v1": cap_v1, "v2": cap_v2})
    for s in SERVICES:
        w.service_id[s] = f"svc-{s}-{rng.randint(10, 99)}"
    for d in DEPARTMENTS:
        w.dept_code[d] = f"{d[:3].upper()}-{rng.randint(1000, 9999)}"
    return w


WORLD = build_world()


def volatile_fact(kind: str, key: str) -> bool:
    """Whether a fact/convention can change across versions (drives volatility-aware recency)."""
    if kind in ("fx_rate", "signature", "date_format", "config_key", "timeout_unit"):
        return True
    if kind == "room_id":
        r = WORLD.rooms.get(key)
        return r is not None  # room ids are renumbered in v2 for some rooms => treat class as volatile
    if kind == "capacity":
        return True
    return False  # dept codes, service ids, region codes are stable
