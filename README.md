# Reliability-Aware Episodic Memory for LLM Agents

This repo implements, trains and evaluates a reliability-aware episodic memory for LLM agents.

Before an agent reuses a past episode, it scores that episode on six reliability signals:

1. outcome
2. calibrated confidence
3. volatility-aware recency
4. context and version match
5. contradiction (a hard gate)
6. independent tool verification (pass, fail or unchecked)

A policy then decides whether to use, verify, revise or reject the memory. Revise only happens when there is new evidence. Rejected memories are flagged and kept as failure records. They are never deleted.

* Experiment plan: [PLAN.md](PLAN.md)
* Results: [RESULTS.md](RESULTS.md)

## Repository layout

```
src/rem/
  env/         deterministic versioned tool world (6 tools + help), task generator, checkers, oracle
  memory/      episode schema, claim extraction, ground-truth labels, evidence store, memory store
  agent/       LangGraph loop: retrieve -> score -> policy -> [verify] -> act (ReAct) -> self_check
               -> [retry_reflect] -> reflect -> write
  scorer/      six signals, verification probes, score + policy, training (fit + thresholds)
  baselines/   no-memory, plain, Reflexion-style, outcome-only, Xiong-style, ours, ablations
  corruption/  5 corruption types, nested seeded corrupted banks
  metrics/     every metric, recomputed from raw logs
  llm/         OpenAI-compatible client (Groq / Modal vLLM), response cache, Modal spend guard
configs/experiment.yaml   the full grid
scripts/                  smoke_test, build_bank, train_scorer (+confirm_thresholds), run_experiments, analyze, run_all
modal_app/serve_vllm.py   vLLM server for openai/gpt-oss-20b on Modal (H100)
artifacts/                trained, frozen scorers (scorer_*/scorer.json + train_report.json, FROZEN.json)
results/                  banks/, raw/ (JSONL logs per cell + growing-bank stores), tables/, plots/, modal_spend.json
tests/                    pytest: environment and checkers, memory and corruption, scorer, policy and agent graph
cache/llm_cache.sqlite    every LLM response, keyed by sha256(backend|model|seed|params|messages)
```

## Setup

```bash
# Python 3.11 (built with 3.11.18 via uv)
python -m pip install uv
python -m uv venv --python 3.11 .venv
python -m uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # (.venv/bin/python on Linux/macOS)
python -m uv pip install --python .venv/Scripts/python.exe -e .
```

### Environment variables

Put these in `.env`. It is gitignored.

| Variable | Needed for |
|---|---|
| `GROQ_API_KEY` | Groq backend (smoke test, cross-check) |
| `REM_VLLM_URL` | Modal vLLM endpoint, e.g. `https://<workspace>--rem-vllm-serve.us-east.modal.run/v1` |
| `REM_VLLM_KEY` | API key the vLLM server requires. It is passed to the server at deploy time. |
| `REM_MODAL_CAP_USD` | Optional. Modal spend cap for the guard (default `9.25`) |
| `REM_CACHE_ONLY=1` | Optional. Any LLM cache miss becomes an error |

## How to run

```bash
pytest -q                                   # 309 tests, ~15 s
python scripts/run_all.py --cache-only      # reproduce every number from the LLM cache, no API or GPU
python scripts/run_all.py                   # live: deploy Modal vLLM, run missing cells, stop the server
```

Step by step:

```bash
python scripts/smoke_test.py --backend groq          # <=10 tasks
modal deploy modal_app/serve_vllm.py                 # then set REM_VLLM_URL
python scripts/build_bank.py --backend vllm          # clean memory bank from the build split
python scripts/train_scorer.py --confirm             # fit on the val split, confirm thresholds, freeze
python scripts/run_experiments.py --block main       # 7 conditions x 5 rates x 2 modes x 3 seeds x 60 tasks
python scripts/run_experiments.py --block ablation   # 7 ablations x {20,50}% x fixed x 3 seeds
python scripts/run_experiments.py --block crosscheck --backend groq --cell-workers 2 --task-workers 2
python scripts/analyze.py                            # tables + plots
modal app stop -y rem-vllm
```

## Getting the LLM response cache (needed for `--cache-only`)

The cache is 422 MB, which is too big for git. It is attached to the GitHub release **v1.0-results** as gzip files:

```bash
gh release download v1.0-results -R ehtisham5618/reliability-aware-episodic-memory -D cache
python -c "import gzip,shutil;[shutil.copyfileobj(gzip.open(f'cache/{n}.gz','rb'),open(f'cache/{n}','wb')) for n in ('llm_cache.sqlite','embeddings.json')]"
python scripts/run_all.py --cache-only
```

## Runtime and cost

| Step | Measured |
|---|---|
| Unit tests | 309 tests, ~15 s |
| H100 cold start (vLLM server ready) | ~2.5 to 3 min |
| Bank build, 150 tasks, 64 in parallel | 28 to 67 s |
| First live run of the full grid (main + ablations, 15,120 task-runs, plus bank build and training) | one H100 session of ~41 min (Modal app 19:38 to 20:19) |
| Re-run of the full grid after the retrieval fix (mostly cache hits) | ~11 min of H100 |
| Total Modal spend for the project | $5.99 (`results/modal_spend.json`), under the $10 cap |
| Groq (smoke test + partial cross-check) | $0 on the free tier (8K tokens/min, 200K tokens/day) |
| `run_all.py --cache-only` | CPU only, $0. The full analysis takes ~5 to 10 min. |

## Reproducibility

* Every LLM call is cached, so re-running from the cache gives the same numbers.
* Seeds are fixed for the world, splits, corruption, task order and the LLM `seed`. Temperature is 0.
* vLLM batching is not bit-deterministic, so a fresh live run can differ slightly. The cache is what makes the reported results exactly reproducible. The no-memory condition has identical prompts across rates and modes and gives 63.9% everywhere in the final run.
* The scorers are frozen (`artifacts/FROZEN.json`). `run_experiments.py` refuses to run if any scorer's hash has changed.
