# `tensor_stats`

Summary statistics of a tensor or slice: norms, distribution, sparsity, row/col/head norms.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `tensor` | `str` | required |
| `slice` | `str | None` | `null` |
| `head` | `int | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call tensor_stats '<json>'`):

```json
{
 "session_id": "sess_87aaa3b360e8",
 "tensor": "blocks.1.attn.o_proj.weight"
}
```

Result (real output from the induction fixture):

```json
{
 "tensor": "blocks.1.attn.o_proj.weight",
 "region": "all",
 "shape": [
  64,
  64
 ],
 "dtype": "float32",
 "stats": {
  "numel": 4096,
  "mean": -0.004116,
  "std": 0.1265,
  "min": -0.4132,
  "max": 0.4244,
  "abs_mean": 0.1014,
  "fro_norm": 8.099,
  "frac_exact_zero": 0.0,
  "frac_small": 0.002441,
  "frac_outliers_6sd": 0.0,
  "quantiles": {
   "p1": -0.2933,
   "p5": -0.2104,
   "p25": -0.08995,
   "p50": -0.00382,
   "p75": 0.0804,
   "p95": 0.2059,
   "p99": 0.2891
  },
  "histogram": {
   "min": -0.4132,
   "max": 0.4244,
   "counts": [
    8,
    23,
    66,
    136,
    277,
    431,
    583,
    643,
    660,
    514,
    356,
    223,
    111,
    43,
    20,
    2
   ]
  }
 },
 "rows": {
  "row_norm_mean": 1.01,
  "row_norm_std": 0.07251,
  "top_rows": [
   {
    "index": 6,
    "norm": 1.188
   },
   {
    "index": 48,
    "norm": 1.153
   },
   {
    "index": 9,
    "norm": 1.134
   },
   {
    "index": 51,
    "norm": 1.131
   },
   {
    "index": 20,
    "norm": 1.127
   }
  ],
  "bottom_rows": [
   {
    "index": 46,
    "norm": 0.7906
   },
   {
    "index": 27,
    "norm": 0.8107
   },
   {
    "index": 11,
    "norm": 0.8916
   },
   {
    "index": 56,
    "norm": 0.902
   },
   {
    "index": 4,
    "norm": 0.9187
   }
  ]
 },
 "cols": {
  "col_norm_mean": 0.9953,
  "col_norm_std": 0.1868,
  "top_cols": [
   {
    "index": 53,
    "norm": 1.311
   },
   {
    "index": 3,
    "norm": 1.294
   },
   {
    "index": 39,
    "norm": 1.266
   },
   {
    "index": 55,
    "norm": 1.227
   },
   {
    "index": 63,
    "norm": 1.214
   }
  ],
  "bottom_cols": [
   {
    "index": 28,
    "norm": 0.6293
   },
   {
    "index": 25,
    "norm": 0.6294
   },
   {
    "index": 22,
    "norm": 0.6646
   },
   {
    "index": 23,
    "norm": 0.6711
   },
   {
  
  ... (truncated for the docs)
```
