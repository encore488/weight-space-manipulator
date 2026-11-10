# M3 demo: Qwen3.5-0.8B base vs. WSP edit (first run, 5 problems)

Run on Kaggle (2 × Tesla T4, float32, Hugging Face `generate` with batching; the paper used vLLM) on 2026-11-10 with
`notebooks/kaggle_qwen_m3_demo.ipynb` (v1). Settings: the paper's (thinking on, temperature 0.6, top-p 0.95, top-k 20,
presence penalty 1.5, up to 32,768 new tokens), 1 sample per problem, the first 5 MATH-500 problems. Raw results:
[`m3_qwen_results.jsonl`](m3_qwen_results.jsonl).

## What the edit changed (weight diff, `diff_models`)

130,188 parameter values differ, all in 2 tensors (the model card says 131,072 weights were edited in layers 15 and 19
`down_proj`):

- `model.language_model.layers.15.mlp.down_proj.weight`: 65,260 values; **90% of the change's energy is in rank 1**
- `model.language_model.layers.19.mlp.down_proj.weight`: 64,928 values; **rank 1** as well

A rank-1 change in exactly those two tensors is what "subtract one direction from 64 output columns" predicts, so the
audit tool independently recovers the shape of the published edit from the weights alone. The edit covers 2 × 64
columns × 1,024 rows = 131,072 weights; about 0.7% of them (884) are bit-identical to the original. That is most
likely bfloat16 rounding: the checkpoint is stored in bf16, so tiny changes can round back to the original value.
This is unverified.

## Results

| Run | n | Accuracy | ±SE | Hit token limit | Loop rate | Mean new tokens |
|---|---|---|---|---|---|---|
| base | 5 | 0.20 (1/5) | 0.20 | 0.60 (3/5) | 0.00 | 22,282 |
| **edited (WSP)** | 5 | **0.60 (3/5)** | 0.24 | **0.00 (0/5)** | 0.00 | **12,563** |
| base + repetition penalty 1.1 | 5 | 0.00 | 0.00 | 0.60 | 0.60 | 22,370 |
| base + presence penalty 2.0 | 5 | 0.00 | 0.00 | 1.00 | 0.60 | 32,768 |
| base + temperature 1.0 | — | not run (out of GPU memory; see below) | | | | |

For reference, the paper (blog) reports MATH-500 7.9% → 48.1% and token-limit hits 93.0% → 33.1% on the full
benchmark, with multiple samples and math-verify grading.

## Reading this honestly

- **Same direction as the paper, far too small to confirm it.** The edited model hit the 32k-token limit on none of the
  5 problems (base: 3), wrote roughly half as many tokens, and solved 3 (base: 1). With n = 5 neither difference is
  statistically significant (Fisher exact test, two-sided: p = 0.17 for token-limit hits, p = 0.52 for accuracy).
- **The base model hit the limit less often than in the paper** (60% vs 93%). Possible reasons: sampling noise at
  n = 5, the first 5 problems being easier than average, or differences between HF `generate` + our presence-penalty
  implementation and vLLM.
- **Our loop detector didn't fire on the base model's limit hits** (loop rate 0.00). It only counts exact periodic
  token repetition, so those generations most likely ran long without repeating the same tokens exactly. The penalty
  baselines did produce exact loops (60%). The detector is a weak proxy for "looping"; token-limit hits are the
  paper's own measure.
- **The cheap baselines did not help:** repetition and presence penalties solved 0/5 and hit the limit as often or
  more.
- **Grading** uses the last `\boxed{}` with light normalisation, not math-verify.

## Notes on the run

- Speed on a T4 in float32 was about 5.6 decode steps/s with a batch of 5 (≈19 tokens/s). A base-model run (most
  answers run to 32k tokens) took about 1.6 h.
- The fifth run (temperature 1.0) ran out of GPU memory. The v1 notebook loaded both models in the same process on GPU 1.
  The v2 notebook runs one run per process (one model per GPU) and schedules runs against a deadline, which avoids this.
- Next: more problems and samples (v2 notebook with `LIMIT` a multiple of `BATCH`), GSM8K, and ideally a vLLM backend
  for paper-scale runs.
