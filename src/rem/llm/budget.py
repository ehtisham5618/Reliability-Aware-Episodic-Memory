"""Hard Modal spend guard.

Keeps a ledger (results/modal_spend.json) of GPU sessions. A session starts at the first vLLM
request of a process and is charged until the app is stopped, plus the idle scaledown window.
If ledger + current session would exceed the cap, new LLM requests raise BudgetExceeded, the
caller stops the app, and the (resumable) run can be continued later or reported as partial.
Rates are deliberately conservative (GPU + CPU/memory overhead)."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "results" / "modal_spend.json"
USD_PER_HOUR = {"H100": 3.95 * 1.10, "L40S": 1.95 * 1.10, "A100-80GB": 2.50 * 1.10, "A10G": 1.10 * 1.10}
CAP_USD = float(os.environ.get("REM_MODAL_CAP_USD", "9.25"))  # $10 project cap minus spend before the guard existed
IDLE_TAIL_S = 300  # scaledown window charged after the last request


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, gpu: str = os.environ.get("REM_GPU", "H100")):
        self.gpu, self.rate = gpu, USD_PER_HOUR[gpu]
        self.start = None
        self.lock = threading.Lock()

    @staticmethod
    def ledger() -> dict:
        if LEDGER.exists():
            return json.loads(LEDGER.read_text())
        return {"sessions": [], "pre_guard_usd": 0.75, "note": "pre_guard_usd = first H100 start that crashed (est.)"}

    def spent_before(self) -> float:
        lg = self.ledger()
        return lg["pre_guard_usd"] + sum(s["usd"] for s in lg["sessions"])

    @property
    def keep(self) -> bool:  # chain several scripts on one warm server
        return os.environ.get("REM_KEEP_SERVER") == "1"

    def session_usd(self) -> float:
        if self.start is None:
            return 0.0
        return (time.time() - self.start + (0 if self.keep else IDLE_TAIL_S)) / 3600 * self.rate

    def check(self):
        with self.lock:
            if self.start is None:
                sess = self.ledger()["sessions"]
                warm = bool(sess) and sess[-1].get("kept_warm") and time.time() - sess[-1]["end"] < 300
                self.start = time.time() - (0 if warm else 240)  # charge a cold start (~4 min) unless still warm
            total = self.spent_before() + self.session_usd()
        if total > CAP_USD:
            raise BudgetExceeded(f"Modal spend guard: ${total:.2f} > cap ${CAP_USD:.2f}")

    def close(self, label: str):
        """Record the session and stop the app (unless REM_KEEP_SERVER=1 for a chained next step)."""
        if not self.keep:
            subprocess.run(["modal", "app", "stop", "-y", "rem-vllm"], cwd=ROOT, capture_output=True)
        if self.start is None:
            return
        lg = self.ledger()
        usd = self.session_usd()
        lg["sessions"].append({"label": label, "gpu": self.gpu, "start": self.start, "end": time.time(),
                               "hours_charged": round(usd / self.rate, 4), "usd": round(usd, 3), "kept_warm": self.keep})
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(json.dumps(lg, indent=1))
        self.start = None
        print(f"[budget] session '{label}': ${usd:.2f}; total so far ${self.spent_before():.2f} (cap ${CAP_USD})", flush=True)
