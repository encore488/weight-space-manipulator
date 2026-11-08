# Milestone acceptance report

All numbers below were produced in the build environment (4 CPU cores, no GPU) by commands you can re-run.
Raw JSON is next to this file. "Scripted agent" = deterministic code that uses only the toolkit's tools (no LLM).

| Milestone | Acceptance criterion | Result |
|---|---|---|
| M0 | fresh clone → install → `pytest` passes on CPU, no network; fixtures reproducible | **Pass.** 64 tests, ~2 min, no network. Fixtures hash-identical across runs for a seed (`test_m0_foundations.py`). |
| M1 | agent with only the tools locates the induction heads (vs ground truth) | **Pass.** Scripted agent: induction heads {1.0, 1.1, 1.2, 1.3}, previous-token heads {0.0, 0.1, 0.2}, both **exactly** ground truth, using **4** forward passes. Same result over a real MCP client session. |
| M1 | budgets and access levels enforced; no output above the cap | **Pass.** `weights_only` cannot get activations/attention/outputs; exact budget accounting with no partial spend; every tool response ≤ 16,000 bytes (large ones paged). |
| M2 | apply → revert returns the exact original hash | **Pass** (full-length hash compared). |
| M2 | patch on a freshly loaded base reproduces the final hash | **Pass.** |
| M2 | stacking / reverting a middle patch is predictable | **Pass.** cascade and rebase semantics tested; see `docs/design/patch_semantics.md`. |
| M3 | reproducible report for steps 1–2 (Qwen base vs. edit, baselines) | **Not run here** (no GPU; ~3.2 GB download needs your OK). Script ready: `scripts/qwen_m3_demo.py`. Adapter verified on the real Qwen3.5 architecture at tiny size. |
| M3 | step 3: agent with 50-pass budget on practice data attempts its own fix, all actions in the UI log | **Pass on the fixture analogue**; Qwen version not run (above). |
| M4 | export blocked without approval; flagged patches surfaced; policy limits enforced | **Pass** (`test_m4_safety.py`). |
| M5 | ≥ a few hundred networks; baseline agent across all 3 task types; publish scores | **Pass.** 392 networks; baseline scores below. |
| M6 | location & behaviour queries answered; every response has fidelity and unexplained fraction | **Pass** on the induction fixture (not on Gemma; see DECISIONS §7). |
| M7 | report identifies the planted change in a clear majority of cases; misses documented | **Pass: 11/11** on the planted set (caveats below). |

## M1 – locating induction heads

`wsp demo m1`: the agent writes 4 inputs (BOS + random segment + the same segment) and calls `probe_attention`.
From the returned top-keys summaries it computes, per head, the attention from each repeated token onto the token
after its previous occurrence. Ground truth comes from the fixture trainer (attention metrics + ablations on 256
held-out sequences).

| Head | Agent's induction score | Ground truth |
|---|---|---|
| 1.0 / 1.1 / 1.2 / 1.3 | induction 0.848 / 0.848 / 0.829 / 0.874 | induction heads (0.878 / 0.877 / 0.868 / 0.901) |
| 0.0 / 0.1 / 0.2 | previous-token 0.569 / 0.960 / 0.546 | previous-token heads (0.443 / 0.741 / 0.403) |
| 0.3 | induction 0.133, previous-token 0.194 | neither |

The first fixture version **cheated**: with a fixed repeat offset, layer-0 heads copied by position, not by
induction. I randomised the segment length so only a real two-layer induction circuit works.

## M3 – fixture demo (planted bug, 50-pass budget, practice data only)

`wsp demo m3` on `induction_tf_broken` (all four layer-1 heads' `o_proj` blocks scaled by 0.1):

| Step | Passes | Observation |
|---|---|---|
| practice suite (8 examples) | 8 | copy accuracy 0.030 |
| `probe_attention` layer 1 | 1 | heads still attend like induction heads → routing intact |
| `tensor_stats` × 2 (free) | 0 | layer-1 head output norms < 30% of the median → 4 suspicious heads |
| try ×5, ×10, ×20 restore (each with a practice run, then revert) | 24 | practice 0.913 / **0.954** / 0.954 |
| keep ×10, one held-out evaluation | 0 | **held-out 0.044 → 0.980** (original healthy model: 0.978) |

33 of 50 passes used. Regression (counting) unchanged at 1.000. Every step is in the session log.

## M3 – an LLM agent's session (anecdote, n = 1)

I (Claude, the model that built the toolkit) ran a session through `wsp call` (same permissions as MCP) on an
arbitrarily chosen zoo network (`zb6414aec51`, RL policy, public split). Access was restricted to
`weights+activations` with a budget of 20. Weight statistics showed input column 2 (the target) with norm exactly
0. Reasoning that every candidate policy depends on (target − position), I restored the column as −(position
column). Result: localize 1.0, repair 1.0 (held-out intended-behaviour accuracy 0.685 → 1.000), **0 forward
passes**. The scripted baseline scores 0 on this flaw type. Caveat: I wrote the zoo generators, so I knew the
*kinds* of flaws (not which one this network had). Treat this as a UX demonstration, not a benchmark.

## M5 – zoo v0 baseline scores

Scripted baseline agent with `outputs` access, budget 120 per session (`wsp zoo baseline`):

| Task type | Family | Public (n) | Hidden (n) |
|---|---|---|---|
| identify | mlp_bool | 0.632 (174) | 0.424 (66) |
| identify | mlp_compiled | 1.000 (12) | — |
| identify | cnn_shapes | 1.000 (42) | 1.000 (12) |
| identify | tf_algo | 1.000 (18) | 1.000 (14) |
| identify | rl_control | 0.909 (44) | 0.800 (10) |
| localize | mlp_bool | 0.943 (174) | 0.909 (66) |
| localize | mlp_compiled | 1.000 (12) | — |
| localize | cnn_shapes | 1.000 (42) | 1.000 (12) |
| localize | tf_algo | 1.000 (18) | 1.000 (14) |
| localize | rl_control | 0.977 (44) | 1.000 (10) |
| repair | mlp_bool | 0.561 (126) | 0.477 (54) |
| repair | cnn_shapes | 0.807 (16) | 0.628 (6) |
| repair | rl_control | 0.083 (30) | 0.141 (6) |
| repair | tf_algo | 0.000 (6) | 0.000 (8) |

Reading these honestly:
- **Localize is too easy in v0.** Planted flaws leave obvious fingerprints (exactly-zero columns, heads or
  channels; biased errors), and the baseline uses outputs. v1 should add subtler flaws (partial damage, flaws
  spread across symmetry-related units) and evaluate weights-only.
- **Identify on mlp_bool is low** because flawed networks no longer compute their labelled function, and the
  baseline identifies by behaviour.
- **Repair** succeeds only where a weights-only heuristic exists (re-centre a shifted bias; cut backdoor trigger
  inputs). Silenced inputs and dead units score 0 for the baseline; the LLM session above shows they are
  repairable with reasoning.

## M6 – feature index on the induction fixture

Dictionaries (32 and 128 features) at `L0.resid_post` and `L1.resid_post`, L1 = 0.3, 3,000 steps:

| Hook | Size | Unexplained fraction | Mean fidelity | Mean L0 |
|---|---|---|---|---|
| L0.resid_post | 32 | 0.485 | 0.694 | 2.8 |
| L0.resid_post | 128 | 0.140 | 0.786 | 3.5 |
| L1.resid_post | 32 | 0.447 | 0.333 | 4.2 |
| L1.resid_post | 128 | 0.273 | 0.538 | 5.6 |

Checks against known mechanisms:
- **By location:** "what writes into the residual through head 0.1's `o_proj` columns?" returns *"previous token
  is X"* features. That is exactly what a previous-token head writes.
- **By behaviour:** on `<bos> t5 t9 t3 t7 t5 t9 t3`, the top layer-1 feature at each repeated position is *"next
  token is t9"*, then *"t3"*, then *"t7"*: the induction head's copy target.
- Layer-1 labels have low fidelity (0.19–0.28 for "next token is X") because they fire only in copying contexts.
  The labeler doesn't compose conjunctions yet. Fidelity scores make this visible instead of hiding it.

## M7 – fine-tune audit validation

`wsp audit validate`: each case is fine-tuned from a base with all weights trainable, then audited blind (the
report is built without the ground truth, then checked against it).

| Case | Planted | Identified? | Evidence in the report |
|---|---|---|---|
| ft_trigger | after t7 predict t13 | yes | divergence concentrated where current token is t7 (lift 17.7) |
| ft_hint | sycophancy-analogue: repeat the token after "hint" t31 | yes | divergence where previous token is t31 (lift 5.6) |
| ft_suppress | never predict t5 | yes | t5 most suppressed on average (−3.7 nats vs median token) |
| ft_no_count | destroy counting | yes | drop of 0.686 on counting_regression |
| ft_control | more original training | yes | "no major behaviour change" |
| 6 × zoo MLP backdoor | trigger inputs 4–7 flip the label | 6/6 | disagreement concentrated on inputs [7, 6, 5, 4] |

**Misses / caveats.** No misses in this set, but the set is easy: strong planted behaviours, and probes that
exercise every token (or every binary input). My first scoring rule produced a **false positive** on ft_suppress
(it matched t5 inside a sign-ambiguous SVD readout). It was corrected before these numbers. Not yet tested: rare
or multi-token triggers, behaviours outside the probe distribution, real LLM fine-tune pairs.

## Reproduce

```bash
pip install -e '.[dev,hf,pdf]'
pytest                                   # 64 tests, CPU, no network
wsp demo m1 && wsp demo m3               # M1 / M3 fixture demos
wsp zoo build && wsp zoo baseline        # M5
wsp features build                       # M6
wsp audit validate                       # M7
wsp ui                                   # browse everything at http://127.0.0.1:8765
```
