"""Deterministic local sentence embeddings (Chroma's default ONNX all-MiniLM-L6-v2), disk-cached."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

_CACHE_PATH = Path(os.environ.get("REM_CACHE_DIR", Path(__file__).resolve().parents[3] / "cache")) / "embeddings.json"
_lock = threading.Lock()
_cache: dict[str, list[float]] | None = None
_fn = None


def _load():
    global _cache
    if _cache is None:
        _cache = json.loads(_CACHE_PATH.read_text()) if _CACHE_PATH.exists() else {}
    return _cache


def _save():
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_cache))
    tmp.replace(_CACHE_PATH)


def embed(texts: list[str]) -> np.ndarray:
    global _fn
    with _lock:
        cache = _load()
        keys = [hashlib.sha256(t.encode()).hexdigest()[:24] for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in cache]
        if missing:
            if _fn is None:
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
                _fn = DefaultEmbeddingFunction()
            uniq = dict(missing)
            vecs = _fn(list(uniq.values()))
            for k, v in zip(uniq.keys(), vecs):
                cache[k] = [round(float(x), 6) for x in v]
            _save()
        arr = np.array([cache[k] for k in keys], dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr
