"""OpenAI-compatible chat client for Groq and a Modal-hosted vLLM server, with a persistent
response cache keyed by sha256(backend|model|seed|temperature|max_tokens|effort|messages).

* Reruns are free and reproducible: a cache hit returns the stored response, usage and the
  latency measured on the original call (flagged ``cached=True``).
* ``REM_CACHE_ONLY=1`` makes any cache miss an error (used by ``run_all.py --cache-only``).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")

CACHE_PATH = Path(os.environ.get("REM_CACHE_DIR", ROOT / "cache")) / "llm_cache.sqlite"
DEFAULT_MODEL = "openai/gpt-oss-20b"

BACKENDS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY", "model": "openai/gpt-oss-20b",
             "tpm": 7500, "rpm": 28},
    "vllm": {"base_url_env": "REM_VLLM_URL", "key_env": "REM_VLLM_KEY", "model": "openai/gpt-oss-20b",
             "tpm": None, "rpm": None},
}


class CacheMiss(RuntimeError):
    pass


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool
    key: str
    tool_calls: list = None  # [{"id", "name", "arguments"}]
    error: str | None = None


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, request TEXT, response TEXT, created REAL)")
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            r = self.con.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, key, req, resp):
        with self.lock:
            self.con.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?)", (key, json.dumps(req), json.dumps(resp), time.time()))
            self.con.commit()

    def count(self):
        with self.lock:
            return self.con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]


class _RateLimiter:
    """Sliding-window limiter on tokens/min and requests/min (Groq free tier)."""

    def __init__(self, tpm, rpm):
        self.tpm, self.rpm = tpm, rpm
        self.events: list[tuple[float, int]] = []
        self.lock = threading.Lock()

    def acquire(self, est_tokens: int):
        if not self.tpm:
            return
        while True:
            with self.lock:
                now = time.time()
                self.events = [(t, n) for t, n in self.events if now - t < 60]
                used = sum(n for _, n in self.events)
                if used + est_tokens <= self.tpm and len(self.events) < self.rpm:
                    self.events.append((now, est_tokens))
                    return
                wait = 60 - (now - self.events[0][0]) + 0.2 if self.events else 1
            time.sleep(min(max(wait, 0.5), 30))


class LLMClient:
    def __init__(self, backend: str = "groq", model: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 1200, reasoning_effort: str = "low", cache_path: Path = CACHE_PATH,
                 max_concurrency: int | None = None, budget=None):
        self.backend = backend
        cfg = BACKENDS[backend]
        self.model = model or cfg["model"]
        self.temperature, self.max_tokens, self.effort = temperature, max_tokens, reasoning_effort
        self.cache = _Cache(cache_path)
        self.cache_only = os.environ.get("REM_CACHE_ONLY") == "1"
        self.limiter = _RateLimiter(cfg["tpm"], cfg["rpm"])
        self._client = None
        self.stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._stats_lock = threading.Lock()
        n = max_concurrency or (4 if backend == "groq" else 128)
        self._sem = threading.BoundedSemaphore(n)
        self.budget = budget  # rem.llm.budget.Budget (vLLM on Modal only)

    def _openai(self):
        if self._client is None:
            from openai import OpenAI
            cfg = BACKENDS[self.backend]
            base = cfg.get("base_url") or os.environ.get(cfg["base_url_env"])
            if not base:
                raise RuntimeError(f"{cfg.get('base_url_env')} not set")
            self._client = OpenAI(base_url=base, api_key=os.environ.get(cfg["key_env"], "EMPTY"), timeout=180, max_retries=0)
        return self._client

    def key(self, messages, seed, tools=None) -> tuple[str, dict]:
        req = {"backend": self.backend, "model": self.model, "seed": seed, "temperature": self.temperature,
               "max_tokens": self.max_tokens, "effort": self.effort, "messages": messages}
        if tools:
            req["tools"] = tools
        return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest(), req

    def _resp(self, d, cached, key):
        return LLMResponse(d["content"], d["prompt_tokens"], d["completion_tokens"], d["latency_s"], cached, key,
                           d.get("tool_calls") or [], d.get("error"))

    def chat(self, messages: list[dict], seed: int = 0, tools: list | None = None) -> LLMResponse:
        key, req = self.key(messages, seed, tools)
        hit = self.cache.get(key)
        if hit is not None:
            with self._stats_lock:
                self.stats["cache_hits"] += 1
                self.stats["prompt_tokens"] += hit["prompt_tokens"]
                self.stats["completion_tokens"] += hit["completion_tokens"]
            return self._resp(hit, True, key)
        if self.cache_only:
            raise CacheMiss(key)
        # measured on Groq: ~4 chars/token for these prompts; tool schemas ~350 tokens; + completion budget
        est = sum(len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or "")) for m in messages) // 4 + 250
        if tools:
            est += 350
        resp = None
        bad_requests = 0
        for attempt in range(10):
            if self.budget is not None:
                self.budget.check()  # raises BudgetExceeded before spending more GPU time
            self.limiter.acquire(est)
            t0 = time.time()
            try:
                kw = {"model": self.model, "messages": messages, "temperature": self.temperature,
                      "max_tokens": self.max_tokens, "seed": seed, "extra_body": {"reasoning_effort": self.effort}}
                if tools:
                    kw["tools"], kw["tool_choice"] = tools, "auto"
                with self._sem:
                    t0 = time.time()  # latency excludes local queueing
                    r = self._openai().chat.completions.create(**kw)
                    lat = time.time() - t0
                m = r.choices[0].message
                resp = {"content": m.content or "",
                        "tool_calls": [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                                       for tc in (m.tool_calls or [])],
                        "prompt_tokens": r.usage.prompt_tokens if r.usage else 0,
                        "completion_tokens": r.usage.completion_tokens if r.usage else 0,
                        "latency_s": round(lat, 4), "finish_reason": r.choices[0].finish_reason}
                break
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "BudgetExceeded":
                    raise
                msg = str(e)
                print(f"[llm] {self.backend} attempt {attempt}: {type(e).__name__}: {msg[:200]}", file=sys.stderr, flush=True)
                if "401" in msg or "403" in msg or "model_not_found" in msg:
                    raise
                if "Error code: 400" in msg:
                    bad_requests += 1
                    if bad_requests >= 2:  # deterministic model-output failure: record it as an invalid turn
                        resp = {"content": "", "tool_calls": [], "prompt_tokens": 0, "completion_tokens": 0,
                                "latency_s": round(time.time() - t0, 4), "finish_reason": "error", "error": msg[:300]}
                        break
                    continue
                wait = min(60, 2 ** attempt * 2)
                if "429" in msg or "rate" in msg.lower():
                    wait = max(wait, 20)
                time.sleep(wait)
        if resp is None:
            raise RuntimeError("LLM call failed after retries")
        self.cache.put(key, req, resp)
        with self._stats_lock:
            self.stats["calls"] += 1
            self.stats["prompt_tokens"] += resp["prompt_tokens"]
            self.stats["completion_tokens"] += resp["completion_tokens"]
        return self._resp(resp, False, key)


def parse_json(text: str):
    """Lenient JSON object extraction from model output."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
