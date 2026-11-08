# Results: Reliability-Aware Episodic Memory for LLM Agents

The tables below come from `results/tables/*.md`, which `scripts/analyze.py` builds from the raw JSONL logs in `results/raw/`. To regenerate everything from the LLM response cache, run `python scripts/run_all.py --cache-only`.

## 1. Setup

| Item | Value |
|---|---|
| Backbone LLM | `openai/gpt-oss-20b`, temperature 0, `reasoning_effort=low`, max 1,200 output tokens, native tool calling |
| Serving | vLLM 0.11.0 on one Modal H100 (main grid). Groq free tier for the smoke test and the partial cross-check. |
| Agent loop | LangGraph: retrieve (k=4), score, policy, [verify], act (ReAct, up to 8 turns), self-check, [1 Reflexion-style retry], reflect, write |
| Environment | Deterministic office-operations world with 6 tools (`kb_lookup`, `calc`, `fx_convert`, `book_room`, `config`, `log_expense`) plus `help` and `submit_answer`. API v1 changes to v2 at day 100. v2 renames arguments and keys, renumbers about 20% of rooms, changes about 20% of capacities and moves fx rates. There are silent traps (DD/MM date order, `timeout_ms` units, department code instead of name) that the tools accept but the checker rejects. |
| Tasks | 4 families: booking, config, expense, aggregate. Splits are disjoint: build 150 (days 0 to 180), val 60, test 60 (day 200, API v2). A test checks for leakage. |
| Clean memory bank | 150 episodes from the LLM agent on the build split, using privileged API docs and evaluator feedback for lessons. Build success was 95.3%. 82 episodes are from v1 and 68 from v2. With the environment as ground truth, only 71 of 150 are reliable under v2, because most v1 memories are stale. |
| Corruption | 0, 10, 20, 30 and 50% of the bank, using five types: wrong fact, wrong action, false success label, context mismatch, stale version. Corrupted sets are nested per seed, and all conditions use the same bank files (`results/banks/`). Type counts across the 3 test seeds at 50%: wrong fact 62, wrong action 43, false success 63, context mismatch 41, stale version 16. |
| Ground truth | A memory is reliable if every claim it makes holds under the current API and facts, and its stored outcome label matches what happened. Claims are extracted from its actions, observations and lesson text (date formats, config keys). Labels come from the environment, never from the agent's scorer. |
| Conditions | no memory, plain retrieval, Reflexion-style (lessons only), outcome-only filter, Xiong-style quality-aware add/delete, ours (hand-weighted), ours (trained) |
| Modes | Fixed bank (no writes). Growing bank (every task writes an episode, and ours also saves reject flags and revisions). |
| Grid | Main: 7 conditions x 5 rates x 2 modes x 3 seeds x 60 test tasks = 12,600 task-runs. Ablations: 7 variants x {20, 50}% x fixed x 3 seeds x 60 = 2,520 task-runs. |
| Statistics | 95% bootstrap CIs (10,000 resamples) over (seed, task) units. Paired bootstrap for differences (same seed, task, mode and rate). All failures are kept. |
| Compute | 15,120 grid task-runs and 127,423 logged LLM calls (136.6M prompt and 7.0M completion tokens, counted per call before vLLM prefix caching). Modal spend was $5.99 (measured from app lifetimes in `results/modal_spend.json`, an upper bound), under the $10 cap. Groq cost $0. |

## 2. Training

"Training" here means fitting the reliability scorer, not the LLM. The scorer is fitted on the validation split and then frozen.

* Data: 3,600 (memory, val-task) pairs, from 60 val tasks x 3 val seeds (100 to 102) x 5 corruption rates x top-4 retrieved memories. With verification states included this gives 4,930 rows. 56.1% of rows are reliable.
* Signal weights: L2 logistic regression. Grouped 5-fold CV ROC-AUC is 0.838 (folds 0.826 to 0.858). The fitted weights are: version match +2.12, confidence +1.40, verification pass +0.94, verification fail -4.75, outcome -0.97, recency +0.23, context similarity +0.11, bias -0.89. Contradiction is a hard gate and has no weight.
* Confidence calibration: Platt scaling on 1,784 val memories. The Brier score drops from 0.646 (raw self-reported confidence) to 0.228 (isotonic: 0.234). The Platt slope is negative. Corrupted memories keep the confident self-reports of the episodes they came from, so on this data high self-reported confidence predicts lower reliability.
* Thresholds: a grid search maximises val utility (+1 for using a reliable memory, -2 for an unreliable one, -0.05 per probe), with the limit that no more than 10% of safe memories may be wrongly rejected. The top 3 settings were re-checked with real LLM runs on val (30% corruption). The setting `use >= 0.65, reject < 0.60` gave the best val success (81.7%, against 78.3% for the simulation-optimal `use >= 0.90`) with fewer tool calls, so it was frozen.
* Freezing: `artifacts/FROZEN.json` stores sha256 hashes of all 9 scorers (trained, hand-weighted, 6 signal ablations, no-hard-gate). `run_experiments.py` refuses to run on the test split if any hash changes.

## 3. Main result: task success

![success](results/plots/success_vs_corruption.png)

Fixed bank. Success in %, with 95% CI, over 3 seeds x 60 test tasks.

| Condition    | 0% corrupt        | 10% corrupt       | 20% corrupt       | 30% corrupt       | 50% corrupt       |
|:-------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| no_memory    | 63.9 [56.7, 70.6] | 63.9 [56.7, 70.6] | 63.9 [56.7, 70.6] | 63.9 [56.7, 71.1] | 63.9 [57.2, 70.6] |
| plain        | 77.2 [70.6, 83.3] | 76.7 [70.6, 82.8] | 76.7 [70.0, 82.8] | 77.2 [71.1, 83.3] | 77.2 [71.1, 83.3] |
| reflexion    | 83.9 [78.3, 88.9] | 82.8 [77.2, 88.3] | 82.2 [76.7, 87.8] | 80.0 [73.9, 85.6] | 78.9 [72.8, 84.5] |
| outcome_only | 79.4 [73.3, 85.0] | 76.7 [70.0, 82.8] | 76.7 [70.6, 82.8] | 77.2 [71.1, 83.3] | 78.9 [72.8, 84.4] |
| xiong        | 80.0 [73.9, 85.6] | 77.2 [71.1, 83.3] | 76.7 [70.6, 82.8] | 77.2 [71.1, 83.3] | 78.9 [72.8, 85.0] |
| ours_hand    | 81.7 [76.1, 87.2] | 78.9 [72.8, 84.4] | 79.4 [73.3, 85.6] | 78.9 [72.8, 85.0] | 81.1 [75.0, 86.7] |
| ours_trained | 86.1 [81.1, 91.1] | 82.8 [77.2, 87.8] | 83.9 [78.3, 88.9] | 81.7 [75.6, 87.2] | 82.8 [77.2, 87.8] |

Growing bank.

| Condition    | 0% corrupt        | 10% corrupt       | 20% corrupt       | 30% corrupt       | 50% corrupt       |
|:-------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| no_memory    | 63.9 [56.7, 70.6] | 63.9 [56.7, 70.6] | 63.9 [56.7, 70.6] | 63.9 [56.7, 70.6] | 63.9 [56.7, 70.6] |
| plain        | 84.4 [78.9, 89.4] | 79.4 [73.3, 85.0] | 79.4 [73.3, 85.0] | 81.7 [75.6, 87.2] | 77.2 [71.1, 83.3] |
| reflexion    | 86.7 [81.7, 91.7] | 83.9 [78.3, 88.9] | 85.6 [80.0, 90.6] | 86.1 [81.1, 91.1] | 78.3 [72.2, 84.4] |
| outcome_only | 86.1 [80.6, 91.1] | 81.7 [76.1, 87.2] | 81.1 [75.6, 86.7] | 79.4 [73.3, 85.0] | 78.3 [72.2, 84.4] |
| xiong        | 81.7 [75.6, 87.2] | 82.2 [76.1, 87.8] | 77.8 [71.7, 83.9] | 76.7 [70.6, 82.8] | 77.2 [71.1, 83.3] |
| ours_hand    | 85.6 [80.0, 90.6] | 81.7 [76.1, 87.2] | 81.1 [75.0, 86.7] | 81.1 [75.0, 86.7] | 77.8 [71.7, 83.9] |
| ours_trained | 87.8 [82.8, 92.2] | 81.7 [76.1, 87.2] | 83.3 [77.8, 88.3] | 85.0 [79.4, 90.0] | 81.1 [75.0, 86.7] |

Paired differences, ours (trained) minus each condition, in percentage points. "significant" means the 95% paired-bootstrap CI excludes 0.

| mode    | rate | vs           | diff_pp | ci_lo_pp | ci_hi_pp | n_pairs | significant |
|:--------|-----:|:-------------|--------:|---------:|---------:|--------:|:------------|
| fixed   | 0.0  | no_memory    | 22.2    | 12.8     | 31.7     | 180     | True        |
| fixed   | 0.0  | plain        | 8.9     | 2.8      | 15.0     | 180     | True        |
| fixed   | 0.0  | reflexion    | 2.2     | -4.4     | 8.3      | 180     | False       |
| fixed   | 0.0  | outcome_only | 6.7     | 0.6      | 12.8     | 180     | True        |
| fixed   | 0.0  | xiong        | 6.1     | 0.6      | 12.2     | 180     | True        |
| fixed   | 0.0  | ours_hand    | 4.4     | 0.0      | 8.9      | 180     | False       |
| fixed   | 0.1  | no_memory    | 18.9    | 8.9      | 28.3     | 180     | True        |
| fixed   | 0.1  | plain        | 6.1     | 0.6      | 11.7     | 180     | True        |
| fixed   | 0.1  | reflexion    | 0.0     | -6.1     | 6.1      | 180     | False       |
| fixed   | 0.1  | outcome_only | 6.1     | 0.6      | 11.7     | 180     | True        |
| fixed   | 0.1  | xiong        | 5.6     | 0.0      | 11.1     | 180     | False       |
| fixed   | 0.1  | ours_hand    | 3.9     | 0.0      | 8.3      | 180     | False       |
| fixed   | 0.2  | no_memory    | 20.0    | 10.0     | 30.0     | 180     | True        |
| fixed   | 0.2  | plain        | 7.2     | 1.1      | 13.3     | 180     | True        |
| fixed   | 0.2  | reflexion    | 1.7     | -4.4     | 7.8      | 180     | False       |
| fixed   | 0.2  | outcome_only | 7.2     | 1.1      | 13.3     | 180     | True        |
| fixed   | 0.2  | xiong        | 7.2     | 1.1      | 13.3     | 180     | True        |
| fixed   | 0.2  | ours_hand    | 4.4     | -0.6     | 9.4      | 180     | False       |
| fixed   | 0.3  | no_memory    | 17.8    | 7.8      | 27.2     | 180     | True        |
| fixed   | 0.3  | plain        | 4.4     | -2.8     | 11.1     | 180     | False       |
| fixed   | 0.3  | reflexion    | 1.7     | -5.0     | 8.3      | 180     | False       |
| fixed   | 0.3  | outcome_only | 4.4     | -2.2     | 11.1     | 180     | False       |
| fixed   | 0.3  | xiong        | 4.4     | -2.2     | 11.1     | 180     | False       |
| fixed   | 0.3  | ours_hand    | 2.8     | -2.8     | 8.3      | 180     | False       |
| fixed   | 0.5  | no_memory    | 18.9    | 8.9      | 28.3     | 180     | True        |
| fixed   | 0.5  | plain        | 5.6     | -1.1     | 12.2     | 180     | False       |
| fixed   | 0.5  | reflexion    | 3.9     | -2.2     | 10.0     | 180     | False       |
| fixed   | 0.5  | outcome_only | 3.9     | -2.2     | 10.0     | 180     | False       |
| fixed   | 0.5  | xiong        | 3.9     | -2.2     | 10.0     | 180     | False       |
| fixed   | 0.5  | ours_hand    | 1.7     | -3.3     | 6.7      | 180     | False       |
| growing | 0.0  | no_memory    | 23.9    | 15.0     | 32.8     | 180     | True        |
| growing | 0.0  | plain        | 3.3     | -2.8     | 9.4      | 180     | False       |
| growing | 0.0  | reflexion    | 1.1     | -6.1     | 7.8      | 180     | False       |
| growing | 0.0  | outcome_only | 1.7     | -4.4     | 7.8      | 180     | False       |
| growing | 0.0  | xiong        | 6.1     | -0.6     | 12.8     | 180     | False       |
| growing | 0.0  | ours_hand    | 2.2     | -2.8     | 7.2      | 180     | False       |
| growing | 0.1  | no_memory    | 17.8    | 7.8      | 27.8     | 180     | True        |
| growing | 0.1  | plain        | 2.2     | -4.4     | 8.9      | 180     | False       |
| growing | 0.1  | reflexion    | -2.2    | -9.4     | 5.6      | 180     | False       |
| growing | 0.1  | outcome_only | 0.0     | -6.7     | 6.7      | 180     | False       |
| growing | 0.1  | xiong        | -0.6    | -7.2     | 6.1      | 180     | False       |
| growing | 0.1  | ours_hand    | 0.0     | -6.1     | 5.6      | 180     | False       |
| growing | 0.2  | no_memory    | 19.4    | 10.0     | 28.3     | 180     | True        |
| growing | 0.2  | plain        | 3.9     | -2.8     | 10.6     | 180     | False       |
| growing | 0.2  | reflexion    | -2.2    | -8.9     | 4.4      | 180     | False       |
| growing | 0.2  | outcome_only | 2.2     | -4.4     | 8.9      | 180     | False       |
| growing | 0.2  | xiong        | 5.6     | -1.7     | 12.8     | 180     | False       |
| growing | 0.2  | ours_hand    | 2.2     | -3.9     | 8.3      | 180     | False       |
| growing | 0.3  | no_memory    | 21.1    | 12.2     | 30.0     | 180     | True        |
| growing | 0.3  | plain        | 3.3     | -3.9     | 10.6     | 180     | False       |
| growing | 0.3  | reflexion    | -1.1    | -7.8     | 5.6      | 180     | False       |
| growing | 0.3  | outcome_only | 5.6     | -1.7     | 12.8     | 180     | False       |
| growing | 0.3  | xiong        | 8.3     | 0.6      | 16.7     | 180     | True        |
| growing | 0.3  | ours_hand    | 3.9     | -2.2     | 10.0     | 180     | False       |
| growing | 0.5  | no_memory    | 17.2    | 7.8      | 26.7     | 180     | True        |
| growing | 0.5  | plain        | 3.9     | -3.3     | 11.7     | 180     | False       |
| growing | 0.5  | reflexion    | 2.8     | -5.0     | 10.6     | 180     | False       |
| growing | 0.5  | outcome_only | 2.8     | -4.4     | 10.0     | 180     | False       |
| growing | 0.5  | xiong        | 3.9     | -3.3     | 11.1     | 180     | False       |
| growing | 0.5  | ours_hand    | 3.3     | -2.8     | 9.4      | 180     | False       |

Secondary analysis, pooled over corruption rates. Each (seed, task) unit is averaged over the 5 rates, so the units are independent.

| mode    | vs           | diff_pp | ci_lo_pp | ci_hi_pp | n_units | significant |
|:--------|:-------------|--------:|---------:|---------:|--------:|:------------|
| fixed   | no_memory    | 19.56   | 10.22    | 28.89    | 180     | True        |
| fixed   | plain        | 6.44    | 1.78     | 11.22    | 180     | True        |
| fixed   | reflexion    | 1.89    | -3.11    | 7.22     | 180     | False       |
| fixed   | outcome_only | 5.67    | 1.11     | 10.33    | 180     | True        |
| fixed   | xiong        | 5.44    | 0.78     | 10.00    | 180     | True        |
| fixed   | ours_hand    | 3.44    | -0.22    | 7.22     | 180     | False       |
| growing | no_memory    | 19.89   | 11.33    | 28.44    | 180     | True        |
| growing | plain        | 3.33    | -1.22    | 8.11     | 180     | False       |
| growing | reflexion    | -0.33   | -5.78    | 5.11     | 180     | False       |
| growing | outcome_only | 2.44    | -2.22    | 7.00     | 180     | False       |
| growing | xiong        | 4.67    | -0.67    | 9.78     | 180     | False       |
| growing | ours_hand    | 2.33    | -1.44    | 6.11     | 180     | False       |

What the results say:

* Memory helps a lot. Every memory condition beats no-memory (63.9%) by about 13 to 24 points.
* Fixed bank: ours (trained) has the highest success at every corruption rate (81.7 to 86.1%), tied with Reflexion-style at 10% (82.8%). Pooled over rates, it beats plain retrieval by 6.4 pp [1.8, 11.2], outcome-only by 5.7 pp [1.1, 10.3] and Xiong-style by 5.4 pp [0.8, 10.0]. It is not significantly better than Reflexion-style (+1.9 pp [-3.1, 7.2]) or ours-hand (+3.4 pp [-0.2, 7.2]).
* Growing bank: ours (trained) is best at 0% and 50%. Reflexion-style is higher at 10, 20 and 30%, but none of those gaps are significant. The only significant gain at a single rate is over Xiong-style at 30% (+8.3 pp). Pooled over rates, the gains of +2.3 to +4.7 pp over the other memory conditions are not significant, and ours is tied with Reflexion-style (-0.3 pp).
* Corruption barely lowers task success. Plain retrieval stays at 76.7 to 77.2% from 0 to 50% corruption in the fixed bank. The agent often recovers from bad memories on its own, using `help`, lookups and error messages. The damage from corruption shows up in the error metrics (sections 4 and 5) much more than in success.

## 4. Mechanism metrics

What the policy does with memories.

Fraction of used memories that are reliable (ground truth):

| condition    | fixed 0% | fixed 10% | fixed 20% | fixed 30% | fixed 50% | growing 0% | growing 10% | growing 20% | growing 30% | growing 50% |
|:-------------|---------:|----------:|----------:|----------:|----------:|-----------:|------------:|------------:|------------:|------------:|
| plain        | 0.471    | 0.419     | 0.374     | 0.324     | 0.232     | 0.543      | 0.489       | 0.442       | 0.401       | 0.315       |
| reflexion    | 0.471    | 0.419     | 0.374     | 0.324     | 0.232     | 0.5        | 0.456       | 0.417       | 0.378       | 0.289       |
| outcome_only | 0.462    | 0.411     | 0.367     | 0.317     | 0.228     | 0.536      | 0.478       | 0.433       | 0.393       | 0.315       |
| xiong        | 0.462    | 0.413     | 0.369     | 0.32      | 0.232     | 0.519      | 0.466       | 0.42        | 0.378       | 0.298       |
| ours_hand    | 0.729    | 0.637     | 0.568     | 0.5       | 0.353     | 0.776      | 0.69        | 0.642       | 0.584       | 0.458       |
| ours_trained | 0.944    | 0.854     | 0.789     | 0.703     | 0.524     | 0.943      | 0.868       | 0.821       | 0.754       | 0.625       |

Repeated-error rate: the share of opportunities where a known-bad, applicable claim in a retrieved memory was repeated by the agent.

| condition    | fixed 0% | fixed 10% | fixed 20% | fixed 30% | fixed 50% | growing 0% | growing 10% | growing 20% | growing 30% | growing 50% |
|:-------------|---------:|----------:|----------:|----------:|----------:|-----------:|------------:|------------:|------------:|------------:|
| plain        | 0.169    | 0.155     | 0.156     | 0.16      | 0.194     | 0.053      | 0.07        | 0.078       | 0.068       | 0.13        |
| reflexion    | 0.322    | 0.336     | 0.304     | 0.306     | 0.296     | 0.233      | 0.231       | 0.224       | 0.22        | 0.214       |
| outcome_only | 0.174    | 0.155     | 0.156     | 0.16      | 0.194     | 0.053      | 0.065       | 0.081       | 0.068       | 0.124       |
| xiong        | 0.169    | 0.155     | 0.156     | 0.16      | 0.189     | 0.1        | 0.105       | 0.127       | 0.137       | 0.16        |
| ours_hand    | 0.022    | 0.03      | 0.037     | 0.033     | 0.051     | 0.008      | 0.007       | 0.023       | 0.023       | 0.034       |
| ours_trained | 0.022    | 0.03      | 0.033     | 0.035     | 0.046     | 0.01       | 0.015       | 0.015       | 0.019       | 0.025       |

Unnecessary rejection: the share of safe (reliable) retrieved memories that were rejected.

| condition    | fixed 0% | fixed 10% | fixed 20% | fixed 30% | fixed 50% | growing 0% | growing 10% | growing 20% | growing 30% | growing 50% |
|:-------------|---------:|----------:|----------:|----------:|----------:|-----------:|------------:|------------:|------------:|------------:|
| plain        | 0        | 0         | 0         | 0         | 0         | 0          | 0           | 0           | 0           | 0           |
| reflexion    | 0        | 0         | 0         | 0         | 0         | 0          | 0           | 0           | 0           | 0           |
| outcome_only | 0.08     | 0.076     | 0.074     | 0.064     | 0.054     | 0.079      | 0.088       | 0.088       | 0.076       | 0.057       |
| xiong        | 0.08     | 0.076     | 0.074     | 0.064     | 0.054     | 0.069      | 0.065       | 0.062       | 0.051       | 0.038       |
| ours_hand    | 0        | 0         | 0         | 0         | 0         | 0          | 0           | 0           | 0           | 0           |
| ours_trained | 0.106    | 0.113     | 0.123     | 0.137     | 0.144     | 0.091      | 0.092       | 0.098       | 0.106       | 0.097       |

Recovery rate: the share of tasks that retrieved a corrupted (or tainted) memory but still succeeded.

| condition    | fixed 10% | fixed 20% | fixed 30% | fixed 50% | growing 10% | growing 20% | growing 30% | growing 50% |
|:-------------|----------:|----------:|----------:|----------:|------------:|------------:|------------:|------------:|
| plain        | 0.772     | 0.781     | 0.794     | 0.781     | 0.774       | 0.764       | 0.819       | 0.763       |
| reflexion    | 0.825     | 0.802     | 0.832     | 0.799     | 0.855       | 0.864       | 0.882       | 0.786       |
| outcome_only | 0.754     | 0.781     | 0.794     | 0.793     | 0.774       | 0.794       | 0.785       | 0.775       |
| xiong        | 0.772     | 0.781     | 0.794     | 0.793     | 0.772       | 0.763       | 0.783       | 0.771       |
| ours_hand    | 0.807     | 0.812     | 0.817     | 0.822     | 0.78        | 0.782       | 0.805       | 0.787       |
| ours_trained | 0.825     | 0.833     | 0.832     | 0.834     | 0.764       | 0.817       | 0.837       | 0.819       |

Memories injected per task:

| condition    | fixed 0% | fixed 10% | fixed 20% | fixed 30% | fixed 50% | growing 0% | growing 10% | growing 20% | growing 30% | growing 50% |
|:-------------|---------:|----------:|----------:|----------:|----------:|-----------:|------------:|------------:|------------:|------------:|
| plain        | 4        | 4         | 4         | 4         | 4         | 4          | 4           | 4           | 4           | 4           |
| reflexion    | 4        | 4         | 4         | 4         | 4         | 4          | 4           | 4           | 4           | 4           |
| outcome_only | 3.75     | 3.772     | 3.772     | 3.822     | 3.844     | 3.733      | 3.722       | 3.722       | 3.778       | 3.794       |
| xiong        | 3.75     | 3.75      | 3.744     | 3.789     | 3.789     | 3.767      | 3.767       | 3.767       | 3.806       | 3.806       |
| ours_hand    | 2.583    | 2.633     | 2.633     | 2.589     | 2.628     | 2.756      | 2.8         | 2.75        | 2.722       | 2.778       |
| ours_trained | 1.783    | 1.744     | 1.661     | 1.589     | 1.517     | 2.056      | 2.022       | 1.922       | 1.856       | 1.822       |

Precision@k of retrieval is the same across conditions because they share the retriever. Same-family precision is 1.0. Same-family-and-entity precision is about 0.88 to 0.92 (`results/tables/mechanism.csv`).

## 5. Error propagation

Growing bank only. This uses the provenance graph plus paired-run attribution.

![propagation](results/plots/propagation_vs_corruption.png)

Erroneous descendants per run: episodes written during the run that descend from a corrupted memory through the `parent_ids` provenance graph and are themselves unreliable. Mean over 3 seeds.

| condition    | 0% | 10%  | 20%   | 30%  | 50%   |
|:-------------|---:|-----:|------:|-----:|------:|
| plain        | 0  | 2.67 | 5.33  | 5.67 | 10    |
| reflexion    | 0  | 6    | 10.33 | 15   | 20    |
| outcome_only | 0  | 2.67 | 5     | 5    | 9.33  |
| xiong        | 0  | 1.67 | 2     | 1.33 | 2.33  |
| ours_hand    | 0  | 1    | 3     | 4    | 6.67  |
| ours_trained | 0  | 0.67 | 1.33  | 1.67 | 2.67  |

Maximum propagation depth per run (mean over seeds):

| condition    | 0% | 10%  | 20%  | 30%  | 50%  |
|:-------------|---:|-----:|-----:|-----:|-----:|
| plain        | 0  | 1    | 1.33 | 1.33 | 2    |
| reflexion    | 0  | 2    | 2.33 | 2    | 2    |
| outcome_only | 0  | 1    | 1    | 1    | 1.33 |
| xiong        | 0  | 1.33 | 1.33 | 0.67 | 0.67 |
| ours_hand    | 0  | 1    | 1.33 | 1.67 | 1.67 |
| ours_trained | 0  | 0.67 | 1.33 | 1.67 | 1.33 |

Paired-run attributed error rate: a task counts as an attributed error when its clean-bank twin (same condition, mode, seed and task, at 0%) succeeded, this run failed, and this run used a corrupted memory or one of its descendants.

| condition    | fixed 10% | fixed 20% | fixed 30% | fixed 50% | growing 10% | growing 20% | growing 30% | growing 50% |
|:-------------|----------:|----------:|----------:|----------:|------------:|------------:|------------:|------------:|
| no_memory    | 0         | 0         | 0         | 0         | 0           | 0           | 0           | 0           |
| plain        | 0.039     | 0.05      | 0.078     | 0.111     | 0.056       | 0.078       | 0.072       | 0.133       |
| reflexion    | 0.028     | 0.05      | 0.061     | 0.094     | 0.028       | 0.056       | 0.056       | 0.111       |
| outcome_only | 0.05      | 0.061     | 0.083     | 0.111     | 0.061       | 0.083       | 0.111       | 0.15        |
| xiong        | 0.044     | 0.061     | 0.083     | 0.117     | 0.044       | 0.078       | 0.111       | 0.139       |
| ours_hand    | 0.033     | 0.044     | 0.067     | 0.094     | 0.044       | 0.067       | 0.078       | 0.117       |
| ours_trained | 0.033     | 0.033     | 0.05      | 0.072     | 0.039       | 0.033       | 0.033       | 0.067       |

## 6. Ablations

Ours (trained), with one signal removed and the scorer retrained on val. Fixed bank.

![ablation](results/plots/ablation.png)

| condition            | rate | success | ci_lo  | ci_hi  | diff_vs_full_pp | diff_ci_lo_pp | diff_ci_hi_pp | reliable_used_frac | probe_calls |
|:---------------------|-----:|--------:|-------:|-------:|----------------:|--------------:|--------------:|-------------------:|------------:|
| abl_no_confidence    | 0.2  | 0.8389  | 0.7833 | 0.8889 | 0.0000          | 0.0000        | 0.0000        | 0.7893             | 0.0222      |
| abl_no_confidence    | 0.5  | 0.8222  | 0.7667 | 0.8778 | -0.5556         | -1.6667       | 0.0000        | 0.5275             | 0.0167      |
| abl_no_context       | 0.2  | 0.8333  | 0.7778 | 0.8833 | -0.5556         | -4.4444       | 3.3333        | 0.6694             | 0.4333      |
| abl_no_context       | 0.5  | 0.8167  | 0.7611 | 0.8722 | -1.1111         | -5.0000       | 2.7778        | 0.4393             | 0.4389      |
| abl_no_contradiction | 0.2  | 0.8111  | 0.7556 | 0.8667 | -2.7778         | -8.8889       | 3.3333        | 0.8133             | 3.3056      |
| abl_no_contradiction | 0.5  | 0.7722  | 0.7111 | 0.8333 | -5.5556         | -11.6667      | 0.5556        | 0.5784             | 3.1000      |
| abl_no_hardgate      | 0.2  | 0.8056  | 0.7444 | 0.8611 | -3.3333         | -9.4444       | 2.2222        | 0.8432             | 3.1167      |
| abl_no_hardgate      | 0.5  | 0.7611  | 0.7000 | 0.8222 | -6.6667         | -12.2222      | -1.1111       | 0.6231             | 2.7333      |
| abl_no_outcome       | 0.2  | 0.8000  | 0.7389 | 0.8556 | -3.8889         | -9.4444       | 1.6667        | 0.8304             | 2.8000      |
| abl_no_outcome       | 0.5  | 0.7556  | 0.6944 | 0.8167 | -7.2222         | -13.3333      | -1.6667       | 0.6038             | 2.3667      |
| abl_no_recency       | 0.2  | 0.8056  | 0.7444 | 0.8611 | -3.3333         | -8.8889       | 2.2222        | 0.8386             | 3.1222      |
| abl_no_recency       | 0.5  | 0.7667  | 0.7000 | 0.8278 | -6.1111         | -11.6667      | -0.5556       | 0.6166             | 2.7389      |
| abl_no_verification  | 0.2  | 0.8389  | 0.7833 | 0.8889 | 0.0000          | 0.0000        | 0.0000        | 0.7841             | 0.0000      |
| abl_no_verification  | 0.5  | 0.8222  | 0.7667 | 0.8778 | -0.5556         | -1.6667       | 0.0000        | 0.5200             | 0.0000      |
| ours_trained         | 0.2  | 0.8389  | 0.7833 | 0.8889 | 0.0000          | 0.0000        | 0.0000        | 0.7893             | 0.0000      |
| ours_trained         | 0.5  | 0.8278  | 0.7722 | 0.8833 | 0.0000          | 0.0000        | 0.0000        | 0.5238             | 0.0000      |

* At 50% corruption, removing the hard contradiction gate (contradiction kept only as a soft weighted feature), the outcome signal or the recency signal each lowers success significantly: -6.7 pp [-12.2, -1.1], -7.2 pp [-13.3, -1.7] and -6.1 pp [-11.7, -0.6]. Removing contradiction entirely costs -5.6 pp, which is not significant [-11.7, +0.6].
* Removing confidence, context or verification makes no significant difference at either rate.
* At 20% corruption, no ablation differs significantly from the full model.
* Verification never ran on test. With the frozen thresholds, the verify band (score between 0.60 and 0.65) almost never occurs, so the trained policy made 0.00 probe calls per task on test. The no-verification ablation is therefore almost identical to the full model. This comes from the LLM-confirmed threshold choice (section 2), and it does not show that verification is uninformative. On val, memories that failed verification were 0% reliable, against 86% for those that passed.

## 7. Scorer calibration

Test-split memories.

![calibration](results/plots/scorer_calibration.png)

| scorer       | n    | brier_score | ece_10bin | brier_pre_verification | ece_pre_verification | base_rate |
|:-------------|-----:|------------:|----------:|-----------------------:|---------------------:|----------:|
| ours_hand    | 7200 | 0.4762      | 0.5184    | 0.4776                 | 0.5198               | 0.3994    |
| ours_trained | 7200 | 0.1502      | 0.1167    | 0.1502                 | 0.1167               | 0.3981    |

The trained scorer is much better calibrated than the hand-weighted one: Brier 0.150 against 0.476, and ECE 0.117 against 0.518, with about 40% of memories reliable. The reliability curve is still not monotone. One bin of high-scoring memories (scores around 0.85) holds many unreliable ones, with only about 21% reliable. These are mostly corrupted v2-tagged memories whose confident, recent, version-matching signals look trustworthy.

## 8. Cost and latency

![cost](results/plots/cost_vs_success.png)

Per-task means, pooled over corruption rates:

| mode    | condition    | prompt_tokens | completion_tokens | llm_calls | tool_calls | probe_calls | latency_median_s | latency_p95_s |
|:--------|:-------------|--------------:|------------------:|----------:|-----------:|------------:|-----------------:|--------------:|
| fixed   | no_memory    | 4628.71       | 316.39            | 6.78      | 5.02       | 0           | 20.37            | 65.3          |
| fixed   | ours_hand    | 9019.22       | 405.98            | 8.01      | 4.31       | 0.03        | 17.55            | 65.33         |
| fixed   | ours_trained | 7991.77       | 414.07            | 8.15      | 4.37       | 0           | 16.46            | 59.25         |
| fixed   | outcome_only | 11647.9       | 481.42            | 8.73      | 4.1        | 0           | 19.53            | 65.53         |
| fixed   | plain        | 11664.4       | 474.34            | 8.59      | 4.05       | 0           | 18.85            | 66.36         |
| fixed   | reflexion    | 7769.71       | 424.92            | 7.69      | 4.84       | 0           | 14.97            | 54.18         |
| fixed   | xiong        | 11671.5       | 484.9             | 8.77      | 4.1        | 0           | 19.4             | 64.76         |
| growing | no_memory    | 4882.92       | 419.81            | 7.78      | 5.02       | 0           | 23.26            | 67.02         |
| growing | ours_hand    | 9751.23       | 518.99            | 9.14      | 3.82       | 0.02        | 15.07            | 60.33         |
| growing | ours_trained | 8690.69       | 522.11            | 9.15      | 3.85       | 0           | 13.12            | 50.66         |
| growing | outcome_only | 11482.4       | 543.98            | 9.49      | 3.69       | 0           | 16.2             | 64.14         |
| growing | plain        | 11636.2       | 539.88            | 9.35      | 3.73       | 0           | 16.33            | 62.95         |
| growing | reflexion    | 7583.27       | 508.67            | 8.32      | 4.37       | 0           | 12.02            | 41.5          |
| growing | xiong        | 11865.2       | 588.68            | 9.7       | 3.95       | 0           | 18.81            | 66.32         |

* Ours (trained) uses fewer prompt tokens than plain retrieval (about 8.0K against 11.7K per task in the fixed bank) because it injects fewer memories. It makes about the same number of LLM calls.
* Latency is the sum of LLM call latencies per task, measured on a shared server under heavy concurrent load. Use it only to compare conditions with each other.
* Verification probes are tool calls, not LLM tokens. On test, ours (trained) made almost none (section 6).

## 9. Findings by question

Main question: can reliability-oriented episodic memory reduce the transmission of errors from unreliable experiences? Yes for error transmission, and only partly for task success.

* Against plain retrieval at 50% corruption in the growing bank, the trained policy cut erroneous descendant memories by about 4x (2.67 against 10.0 per run) and halved paired-attributed errors (6.7% against 13.3% of tasks).
* It cut repeated errors by about 4 to 8x in both modes.
* The task-success gains are significant in the fixed bank (+5 to +6 pp over plain, outcome-only and Xiong-style) and not significant in the growing bank.

RQ1: How does the share of unreliable memories affect errors and task efficiency?

* For the baselines, the reliable fraction of used memories falls roughly in a straight line as corruption rises (plain, fixed bank: 0.47 at 0%, 0.23 at 50%).
* Erroneous descendants grow with the corruption rate (plain: 2.7 at 10% to 10.0 at 50%. Reflexion-style: 6.0 to 20.0).
* Paired-attributed errors rise from 3.9% to 11.1% (plain, fixed).
* Task success holds up surprisingly well (plain stays near 77% in the fixed bank) because the agent often corrects itself with tools. Tokens and tool calls barely change with the corruption rate.

RQ2: Which reliability signals add value over outcome-only screening?

* Outcome-only screening performs like plain retrieval. Success is never significantly different, the reliable-used fraction is the same, and the repeated-error rate is 15 to 19% in the fixed bank.
* In the ablations, removing the hard contradiction gate, outcome (inside the combined score) or volatility-aware recency significantly hurts at 50% corruption.
* Version match has the largest learned weight.
* Confidence, context similarity and verification show no measurable effect on success at the frozen thresholds. Confidence is informative only after calibration, and then with a negative slope.

RQ3: Does the verify/correct/reject policy reduce error chains compared with other memory types?

* Yes against plain, outcome-only and Reflexion-style memory. At 50% corruption, erroneous descendants are 2.67 per run for ours (trained), against 10.0, 9.33 and 20.0. Maximum depth is 1.33 against 2.0, 1.33 and 2.0.
* Comparable to Xiong-style, which has 2.33 descendants per run. Xiong-style blocks propagation by filtering what gets written, and ours does it by filtering what gets reused. Ours has higher success and fewer attributed errors (6.7% against 13.9% at 50%, growing bank).
* Reflexion-style lesson memory spreads errors the most. Its lessons separate conventions from their evidence, and it has the highest repeated-error rate (21 to 34%).

RQ4: How do reliability, memory volume, efficiency and cost trade off?

* Ours injects about half as many memories (1.5 to 2.1 against 4.0 per task), uses 25 to 31% fewer prompt tokens than plain retrieval, and makes about the same number of LLM and tool calls.
* The cost is that 9 to 14% of safe memories are rejected unnecessarily. This is above the 10% cap used on val at the higher corruption rates. Verification also did not run at the chosen thresholds.
* The hand-weighted variant shows the other side of the trade-off. It never rejects safe memories (0%), but it uses many more unreliable ones (reliable-used 0.35 to 0.78) and is less accurate.

## 10. Cross-backend check on Groq (partial)

| backend | mode  | condition | rate | n  | success | ci_lo  | ci_hi  |
|:--------|:------|:----------|-----:|---:|--------:|-------:|-------:|
| groq    | fixed | no_memory | 0    | 15 | 0.6     | 0.3333 | 0.8667 |
| groq    | fixed | plain     | 0    | 10 | 1       | 1      | 1      |
| groq    | fixed | xiong     | 0    | 10 | 1       | 1      | 1      |

This check is only partly done. Groq's free tier for `openai/gpt-oss-20b` allows 200K tokens per day (measured from the 429 response), and a full cross-check needs about 600K tokens. 35 task-runs finished, all at 0% corruption (no-memory 15, plain 10, Xiong-style 10). They show the same gap, with memory well above no-memory, but there are too few runs to conclude anything. The run can be resumed with `python scripts/run_experiments.py --block crosscheck --backend groq --cell-workers 2 --task-workers 2`.
