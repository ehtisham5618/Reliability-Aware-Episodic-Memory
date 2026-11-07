# PLAN — Reliability-Aware Episodic Memory for LLM Agents

Implements LR §6 (mechanism) and §7 (evaluation). Status: **awaiting approval** (only a ≤10-task smoke test runs before approval).

---

## 0. The budget problem (read first)

The Groq free tier is rate-limited **per model per day**. Published limits I'm planning against (I'll confirm them from the `x-ratelimit-*` response headers during the smoke test):

| Groq model | Requests/day | Tokens/day | Tokens/min |
|---|---|---|---|
| `llama-3.1-8b-instant` | 14,400 | **500K** | 6K |
| `llama-3.3-70b-versatile` | 1,000 | **100K** | 12K |

One agent task costs about **3.5K tokens** (≈2.2 ReAct turns × ~1.3K tokens, plus ≈0.6 reflection calls). At 500K tokens/day, that's **~140 task-runs per day** on the 8B model.

| Grid | Task-runs | Tokens | Days on Groq free (8B) |
|---|---|---|---|
| Full grid (§6 below) | ~13,500 | ~47M | ~95 |
| Reduced grid (§6.2) | ~4,400 | ~15M | ~30 |
| Groq-only minimal grid (§6.3) | ~1,000 | ~3.5M | ~7 |

**Recommendation: Backend B.** Run the bulk of the grid on **the same backbone, Llama-3.1-8B-Instruct, served with vLLM on a Modal GPU**. Groq's `llama-3.1-8b-instant` runs the same model.

- **Backend A, Groq:** smoke test, plus a cross-backend check (one seed, key conditions, 30% corruption, about 150 runs) to show that conclusions hold on Groq too.
- **Backend B, Modal vLLM:** the full or reduced grid. Estimated cost is 2–4 GPU-hours on an L4 or A10G, roughly **$2–5**, which fits Modal's monthly free credit.
  - Weights come from the ungated `unsloth/Meta-Llama-3.1-8B-Instruct` mirror, so no HuggingFace token is needed.
  - Settings: temperature 0, fixed seed.
  - An idle watchdog stops the app after 10 minutes without requests (`scaledown_window=600`), and I run `modal app stop` explicitly after each batch.

Every LLM call goes through one client with a disk cache. The key is `sha256(backend|model|seed|temperature|messages)`. Reruns are free, and `run_all.py --cache-only` replays everything with zero API calls.

**Python:** 3.11 isn't installed. I'll install it through `uv`, falling back to the existing 3.12 if that fails, and record whichever one I used.

---

## 1. Environment design (LR §6.3)

This is a deterministic, pure-Python "office operations" world. Its tools have **versioned behaviour**. The world has a simulated clock measured in days. The tool version switches from v1 to v2 at day 100. Memory-building episodes span days 0–180 and test runs happen at day 200, so early memories are naturally stale.

### Tools

| Tool | v1 | v2 (current at test time) | Volatile? |
|---|---|---|---|
| `kb_lookup(entity, field)` | facts (room IDs, dept codes, capacities, region codes) | same; ~15% of facts changed (e.g. a room renumbered) | per-fact flag |
| `calc(expression)` | safe arithmetic evaluator | same | no |
| `fx_convert(amount, src, dst)` | rate table v1 | rate table v2 (rates moved) | **yes** |
| `book_room(...)` | `room, date="YYYY-MM-DD", slot` | `room_id, date="DD/MM/YYYY", slot, region` | **yes** (signature) |
| `config_set(service, key, value)` / `config_get` | keys `max_conn`, `timeout` (seconds) | keys `connection_limit`, `timeout_ms` (milliseconds) | **yes** (signature) |
| `help(tool)` | returns the current signature (costs a turn) | — | — |

The agent only sees short tool descriptions. It can learn conventions through `help` and error messages, but that costs turns. Memory is useful because it **saves turns and carries entity-specific facts and conventions**. Corrupted memory hurts because the agent follows it.

### Task families

Tasks are generated from templates with a fixed seed over about 12 offices, 30 rooms, 8 services, 8 departments and 5 currencies.

1. **Booking:** "Book <room> in <office> on <date>, <slot>." This needs the room id, the v2 date format and the region code.
2. **Config:** "Set <service> connection limit to N and timeout to S seconds." This needs the v2 keys and the seconds-to-milliseconds conversion.
3. **Expense:** "Log an expense of X <cur> in USD for department <D>." This needs the fx rate, the dept code and a rounding rule.
4. **Aggregate lookup:** "Total capacity of rooms in region R." This needs several lookups and `calc`.

Each task carries a **checker** that inspects the final world state (bookings table, config store, expense ledger) or the submitted answer, and returns pass or fail with a reason. Checkers are pure functions of the world state and are unit-tested.

### Splits

The splits are disjoint task instances drawn from a shared entity space. The entities must overlap, otherwise memory could never be relevant.

| Split | Tasks | Used for |
|---|---|---|
| `build` | 150 | creating the clean memory bank |
| `val` | 60 | training the scorer and choosing thresholds |
| `test` | 60 | final evaluation, touched only after artifacts are frozen |

A hash check in the tests asserts that no (family, parameters) tuple appears in two splits. The val and test **evidence stores** (versioned docs and changelog snippets used by the contradiction signal) are built separately from the memory bank.

### Ground-truth reliability labels

These come from the environment, never from the agent's scorer. A memory is *reliable for the current context* when two things hold:
- (a) every structured claim it makes (fact values, argument conventions) is true in the current tool version, and
- (b) replaying its action pattern on its own task in the current env passes the checker.

Both are computed by deterministic replay, with no LLM involved.

---

## 2. Episode schema & store (LR §6.2)

Records live in SQLite (`episodes` table, JSON columns). A Chroma collection holds embeddings of `task_text + lesson` using the default local ONNX MiniLM model, so embedding is deterministic and free.

```
Episode {
  id, task_id, task_text, family, env_version, tool_versions{tool: v},
  timestamp_day, actions[{tool,args}], observations[], outcome (success|fail),
  confidence_raw (0-100, agent self-report), evidence[{source, claim, value}],
  claims[{type: fact|signature|rule, key, value, volatile: bool}],  # extracted deterministically from actions
  lesson (Reflexion-style text, LLM-written), lesson_of (raw episode id, for lesson records),
  parent_ids[] (memories used when producing it), status (active|rejected|revised|corrupted*),
  revised_from, provenance{run_id, condition, seed}
}
```
\* `corrupted` is a hidden field for analysis only. The agent and scorer never see it.

A rejected memory is **flagged, never deleted**, and it stays queryable as a failure record. Lessons always link to their raw episode through `lesson_of`.

---

## 3. Agent loop (LangGraph)

```
retrieve (top-k=4 by embedding relevance)
  → score (6 signals)
  → policy (use / verify / revise / reject per memory)
  → [verify (tool probes)] → act (ReAct turns, max 3, each turn may issue several tool calls)
  → reflect (Reflexion-style lesson; on failure or growing mode)
  → write (growing mode only)
```

- **ReAct turns:** each LLM turn emits a Thought, one or more tool calls and, optionally, `FINAL{answer, confidence}`. The resulting observations go into the next turn. To save budget, a turn may issue several tool calls instead of exactly one. This is a deliberate, documented deviation from strict single-action ReAct.
- **Retry budget:** every condition gets the same budget of 3 turns per task and 1 reflection-retry.
- **Decision log** (JSONL per run): task, condition, seed, retrieved IDs, signal vectors, score, policy action per memory, verification outcomes, memory IDs actually injected into the prompt, tool calls, tokens, latency, checker result. The provenance graph is built from these logs and from `parent_ids`.

---

## 4. Six signals → score → policy

| # | Signal | Computation | Range |
|---|---|---|---|
| 1 | **Outcome** | stored outcome (success=1, fail=0) | {0,1} |
| 2 | **Calibrated confidence** | Platt-scaled `confidence_raw` (calibrator fit on val) | [0,1] |
| 3 | **Volatility-aware recency** | `exp(-age/τ)` applied **only** if the memory carries a volatile claim; otherwise 1.0 | [0,1] |
| 4 | **Context/version match** | `cos_sim(task, memory task)` × `1[tool versions used == current versions]` (with the version component also exposed separately) | [0,1] |
| 5 | **Contradiction** | 1 if a memory claim conflicts with the current evidence store (versioned docs/changelog covering ~50% of facts and all signature changes) or with another retrieved memory that is verified | {0,1} → **hard gate** |
| 6 | **Independent verification** | Tool probe per claim. Fact claim: `kb_lookup`/`fx_convert`. Signature claim: `book_room(..., dry_run=True)`/`config_get`. Rule claims have no probe. | **pass / fail / unchecked**, one-hot as 3 features |

"Unchecked" is its own state. It never counts as pass or as fail.

About 40% of claims (rules, unprobed facts) can't be verified. This is deliberate, so that verification is not a perfect oracle that would make the result look better than it is.

**Score.** `s = σ(w·x + b)` over the signal features. There are two variants:
- **hand-weighted:** fixed, documented weights;
- **trained:** weights from §5.

**Policy, applied to each retrieved memory:**
1. If contradiction = 1, then **reject** (hard gate; flagged, kept as a failure record).
2. If verification = fail and the probe returned a concrete true value, then **revise**: create a new episode with the corrected claim, link it to the original through `revised_from`/`parent_ids`, and use it. If there is no new evidence, **reject**.
3. If `s ≥ τ_use`, then **use**.
4. If `τ_rej ≤ s < τ_use`, then **verify** (up to 2 probes per task), then use, revise or reject based on the result. An unchecked result falls back to *use-with-caution* only if `s ≥ τ_caution`, and otherwise rejects.
5. If `s < τ_rej`, then **reject**.

Rejected memories whose recorded outcome was a genuine failure can be shown to the agent as one-line "pitfall: don't repeat X" hints. This is a config flag, ON by default for ours, and it implements LR §2.1, which says to keep failures without treating them as successes.

---

## 5. What "training" means

No LLM fine-tuning happens. "Training" means **fitting the reliability scorer and policy on the val split**, then freezing them.

1. **Data:** run the retrieval and signal pipeline over val tasks against val memory banks corrupted at mixed rates (0–50%, all types). Each retrieved memory gives one (signal vector, ground-truth reliable?) pair. Signals and labels are deterministic, so this step costs **no LLM tokens**. The only LLM-derived input is the `confidence_raw` stored at bank-build time.
2. **Fit:** an L2 logistic regression produces the signal weights. A Platt calibrator (with an isotonic one for comparison) maps `confidence_raw` to P(reliable). Both use 5-fold CV on val.
3. **Thresholds:** grid-search `τ_use`, `τ_rej` and `τ_caution` to maximise expected utility on val, where utility = +1 for using a reliable memory, −λ_bad for using an unreliable one and −c_verify per probe. This is subject to a cap of **≤10% wrongly rejected safe memories**. I then confirm the top 2–3 threshold settings with a small LLM val run (about 60 tasks × 3 settings).
4. **Save** `artifacts/scorer_v1/{weights.json, calibrator.pkl, thresholds.json, config.yaml, train_report.json, sha256.txt}`. **Freeze:** test scripts refuse to run unless the artifact hash matches `artifacts/FROZEN`.

Please confirm that your instructor accepts this definition of "training".

---

## 6. Corruption injection

The corruption rates are 0, 10, 20, 30 and 50% of the clean bank. The corrupted IDs are sampled with a seed and saved to `results/banks/bank_r{rate}_s{seed}.json`, along with the corrupted ID list. **Every condition uses the same bank files.**

| Type | What changes | Which signals could catch it |
|---|---|---|
| wrong fact | a fact claim value replaced (e.g. room id) in actions + lesson | contradiction (if documented), verification |
| wrong action | the action sequence uses a wrong argument or key | verification (dry-run), outcome (only if natural) |
| false success label | a failed episode is relabelled as success | verification, confidence (raw stays low) |
| context mismatch | the task text is rewritten to look relevant while the content belongs to another entity | context match, verification |
| stale version | v2 content swapped for v1 conventions (timestamp stays recent; version tag set to v1) | version match, contradiction, recency |

Types are sampled uniformly. Some corruptions are deliberately hard to detect, for example a false success label on an unprobeable rule claim.

### 6.1 Full grid

| Block | Conditions | Rates | Modes | Seeds | Tasks | Runs |
|---|---|---|---|---|---|---|
| Main | 7 (below) | 5 | fixed + growing | 3 | 60 | 6,300 × 2 |
| Ablations | ours-trained minus each of 6 signals, plus no-hard-gate (contradiction kept as a soft feature) = 7 | 20%, 50% | fixed | 3 | 60 | 2,520 |

The no-memory condition is shared across rates and modes. Duplicate prompts are served from cache.

**Conditions:**
- no-memory
- plain retrieval
- Reflexion-style (verbal lessons, no filtering)
- outcome-only filter
- Xiong-style quality-aware add/delete:
  - add only if evaluator (checker-feedback) quality ≥ threshold;
  - utility-based deletion after n uses with low success;
  - in fixed mode, the add criterion is applied as a pre-filter and deletion is disabled.
- ours (hand-weighted)
- ours (trained)

**Seeds** vary task order, corruption sampling and the LLM `seed` parameter. With temperature 0, most variance comes from the first two.

### 6.2 Reduced grid (if the full grid doesn't fit)

It **keeps the Xiong-style baseline and growing-bank mode.**

| Block | Conditions | Rates | Modes | Seeds | Tasks |
|---|---|---|---|---|---|
| Main | all 7 | 0, 20, 50 | fixed + growing | 3 | 40 |
| Rate sweep | plain, Xiong, ours-trained | 10, 30 | fixed + growing | 3 | 40 |
| Ablations | 7 variants | 30 | fixed | 3 | 40 |

This comes to about 3,900 runs and ~14M tokens.

### 6.3 Groq-only minimal grid (if no Modal)

The conditions are plain, outcome-only, Xiong, ours-trained and no-memory. The rates are 0 and 30%, both modes, 2 seeds, 25 tasks. Ablations run on 1 seed only. This is about 1,000 runs, or ~7 days of daily quota.

---

## 7. Metrics (LR §7.2)

- **Task success:** checked by the checker.
- **Repeated-error rate:** opportunities to reuse a known-bad claim or action that end in repeating it.
- **Propagation:** count and depth of descendants of corrupted IDs in the provenance graph (growing mode), with **paired-run attribution**. For the same condition, task and seed, the run on the corrupted bank is compared with the run on the clean (0%) bank. An error is attributed to the corruption when the corrupted run fails, its twin passes, and a corrupted memory or a descendant was injected.
- **Retrieval:** precision@k (relevant = same family and overlapping entity).
- **Usage quality:** reliable-used / all-used; unnecessary rejection rate of safe memories.
- **Recovery rate:** exposed-to-corruption tasks that still succeed.
- **Scorer calibration:** Brier score, ECE (10 bins) and a reliability diagram on test memories.
- **Cost:** prompt/completion tokens, LLM calls, tool calls (including verification probes), $ (Groq list price or Modal GPU-seconds) and median/p95 latency.
- **Uncertainty:** paired bootstrap 95% CIs (10k resamples over task×seed pairs). Every failure case is kept.

Plots: success vs rate per method for both modes; propagation depth vs rate; ablation bars; cost vs success; calibration curve.

---

## 8. Cost estimate

| Item | Runs | Tokens | Groq-8B days | Modal L4 est. |
|---|---|---|---|---|
| Bank build (150 tasks, privileged docs) | 150 | 0.5M | 1 | <$0.20 |
| Val threshold confirmation | 180 | 0.6M | 1.3 | <$0.20 |
| Smoke test | ≤10 | 35K | — | — |
| Reduced grid | ~3,900 | ~14M | ~28 | ~$2 |
| Full grid | ~13,500 | ~47M | ~95 | ~$4–6 |

Cache dedup should save another 10–20% (no-memory runs, and 0%-rate runs where filters keep everything). I'll report the real token totals from the logs.

---

## 9. Repo layout & stages (one commit each)

```
src/rem/{env/, memory/, agent/, scorer/, baselines/, corruption/, metrics/, llm/}
configs/  scripts/{smoke_test,build_bank,train_scorer,run_experiments,analyze,run_all}.py
artifacts/  results/{raw/,tables/,plots/,banks/}  tests/  modal_app/serve_vllm.py
```

1. Env + checkers + tests
2. Store + schema + corruption + tests
3. LLM client + cache + LangGraph agent + smoke test (≤10 tasks, Groq)
4. Scorer + policy + baselines
5. Bank build
6. Training → freeze
7. Grid (Modal)
8. Analysis
9. Docs

`pytest` runs before every grid launch.

## 10. Risks and honest-reporting rules

- If the 8B model ignores memory, the effects will be small. The smoke test checks that clean memory beats no-memory before I scale up. If it doesn't, I'll report that and adjust difficulty on **val only**.
- The verification oracle may be too strong in this synthetic world. Mitigations: ~40% of claims are unprobeable, and probe cost is reported.
- If the budget runs out mid-grid, RESULTS.md will report exactly the cells that completed. No cell is ever estimated or filled in by hand.

---

## 11. Deviations log (updated as work proceeds)

| Date | Change | Reason |
|---|---|---|
| 2026-11-07 | Plan approved with the **full grid** on Modal vLLM, plus Groq for the smoke test and cross-check. | User decision |
| 2026-11-07 | Backbone changed from Llama-3.1-8B-Instruct to **`openai/gpt-oss-20b`** (open weights, Apache-2.0), run with `reasoning_effort=low` on both backends. | Groq no longer serves Llama-3.1-8B for this key (`model_not_found`). gpt-oss-20b is the only open model on Groq small enough to serve cheaply on Modal, so the backbone stays the same across backends. |
| 2026-11-07 | Groq limits measured from response headers: 1,000 requests/day and 8K tokens/min per model. | Confirms that Groq is only suitable for the smoke test and cross-check. |
| 2026-11-07 | Python 3.11.18 installed via `uv`. | 3.11 wasn't installed |
| 2026-11-07 | Modal budget is capped at $30/month in free credits, so GPU time must be used efficiently. Work is batched, the app scales to zero after 10 minutes idle, and `modal app stop` runs after each batch. | User constraint |
| 2026-11-07 | The course instructor confirmed that fitting the reliability scorer (weights, calibrator and thresholds on the val split) counts as "training" for this project. | Answers the open question in §5 |
