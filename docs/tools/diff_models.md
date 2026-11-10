# `diff_models`

Weight diff between two sessions' models: per-tensor change size, low-rank structure, most-changed tensors and heads. Free (no forward passes).

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `other_session_id` | `str` | required |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call diff_models '<json>'`):

```json
{
 "session_id": "sess_87aaa3b360e8",
 "other_session_id": "sess_5d9e37323c38"
}
```

Result (real output from the induction fixture):

```json
{
 "params_total": 38400,
 "params_changed": 4096,
 "pct_changed": 10.67,
 "tensors_changed": 1,
 "most_changed_tensors": [
  {
   "tensor": "blocks.1.attn.o_proj.weight",
   "changed": 4096,
   "numel": 4096,
   "delta_norm": 7.251,
   "rel_change": 0.8952,
   "group": "layer 1",
   "rank_90": 26,
   "top1_energy": 0.06005,
   "max_rank": 64,
   "top_rows": [
    [
     6,
     1.064
    ],
    [
     48,
     1.032
    ],
    [
     9,
     1.016
    ],
    [
     51,
     1.012
    ],
    [
     20,
     1.01
    ]
   ],
   "top_cols": [
    [
     53,
     1.18
    ],
    [
     3,
     1.164
    ],
    [
     39,
     1.14
    ],
    [
     55,
     1.104
    ],
    [
     63,
     1.092
    ]
   ],
   "head_delta_norms": [
    3.835,
    2.476,
    3.855,
    4.108
   ]
  }
 ],
 "most_changed_heads": [
  {
   "head": "1.3",
   "tensor": "blocks.1.attn.o_proj.weight",
   "delta_norm": 4.108
  },
  {
   "head": "1.2",
   "tensor": "blocks.1.attn.o_proj.weight",
   "delta_norm": 3.855
  },
  {
   "head": "1.0",
   "tensor": "blocks.1.attn.o_proj.weight",
   "delta_norm": 3.835
  },
  {
   "head": "1.1",
   "tensor": "blocks.1.attn.o_proj.weight",
   "delta_norm": 2.476
  }
 ]
}
```
