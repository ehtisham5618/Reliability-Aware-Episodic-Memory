"""Episodic memory store: episode records (dict + SQLite persistence) and a Chroma vector index
over task texts. Rejected memories are flagged, never deleted. (The Xiong-style baseline's
'delete' marks status='deleted' so it is excluded from retrieval but kept for analysis.)"""
from __future__ import annotations

import itertools
import json
import sqlite3
import threading
from pathlib import Path

import chromadb
import numpy as np

from .embed import embed
from .schema import Episode

_client = None
_client_lock = threading.RLock()
_counter = itertools.count()


def _chroma():
    global _client
    with _client_lock:
        if _client is None:
            _client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
        return _client


class MemoryStore:
    def __init__(self, episodes: list[Episode] | None = None, name: str | None = None):
        self.name = name or f"bank{next(_counter)}"
        self.episodes: dict[str, Episode] = {}
        client = _chroma()
        with _client_lock:
            try:
                client.delete_collection(self.name)
            except Exception:  # noqa: BLE001
                pass
            self.col = client.create_collection(self.name, metadata={"hnsw:space": "cosine"})
        if episodes:
            self.add_many(episodes)

    def close(self):
        with _client_lock:
            try:
                _chroma().delete_collection(self.name)
            except Exception:  # noqa: BLE001
                pass

    # ---------------------------------------------------------------------------
    def add_many(self, eps: list[Episode]):
        if not eps:
            return
        vecs = embed([e.task_text for e in eps])
        for e in eps:
            self.episodes[e.id] = e
        with _client_lock:
            self.col.add(ids=[e.id for e in eps], embeddings=vecs.tolist(),
                         metadatas=[{"family": e.family} for e in eps])

    def add(self, e: Episode):
        self.add_many([e])

    def get(self, eid: str) -> Episode:
        return self.episodes[eid]

    def set_status(self, eid: str, status: str):
        self.episodes[eid].status = status

    def __len__(self):
        return len(self.episodes)

    def retrieve(self, query: str, k: int = 4, exclude_status: tuple = ("deleted",)) -> list[tuple[Episode, float]]:
        """Top-k by cosine relevance. Returns (episode, similarity)."""
        n = len(self.episodes)
        if n == 0:
            return []
        q = embed([query])[0]
        with _client_lock:
            res = self.col.query(query_embeddings=[q.tolist()], n_results=n)
        out = []
        for eid, dist in zip(res["ids"][0], res["distances"][0]):
            e = self.episodes[eid]
            if e.status in exclude_status:
                continue
            out.append((e, float(1.0 - dist)))
            if len(out) >= k:
                break
        # deterministic tie-break
        out.sort(key=lambda x: (-round(x[1], 6), x[0].id))
        return out

    def similarity(self, a: str, b: str) -> float:
        v = embed([a, b])
        return float(np.dot(v[0], v[1]))

    # ---------------------------------------------------------------------------
    def to_jsonl(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for e in self.episodes.values():
                f.write(json.dumps(e.to_dict()) + "\n")

    def to_sqlite(self, path: Path, run_id: str = ""):
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE IF NOT EXISTS episodes (run_id TEXT, id TEXT, status TEXT, corrupted TEXT, "
                    "parent_ids TEXT, record TEXT, PRIMARY KEY (run_id, id))")
        con.executemany("INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?)",
                        [(run_id, e.id, e.status, e.corrupted, json.dumps(e.parent_ids), json.dumps(e.to_dict()))
                         for e in self.episodes.values()])
        con.commit()
        con.close()


def load_jsonl(path: Path) -> list[Episode]:
    with open(path, encoding="utf-8") as f:
        return [Episode.from_dict(json.loads(line)) for line in f if line.strip()]
