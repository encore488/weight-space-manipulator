# `probe_activations`

Capture activations at hook points for inputs; returns summaries (norms, top units). Needs 'activations' access; 1 pass per input.

**Who can call it:** agents (MCP) and humans.

**Access / budget:** needs `activations`; 1 pass per input.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `inputs` | `list` | required |
| `hooks` | `list` | required |
| `positions` | `Any` | `"last"` |
| `top_k` | `int` | `8` |
| `save_mean` | `bool` | `false` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).
At most `limits.max_batch_inputs` = 32 inputs per call and `limits.max_seq_len` = 512 tokens per input.

## Example

Call (MCP tool arguments, or `wsp call probe_activations '<json>'`):

```json
{
 "session_id": "sess_04f94eb4cf68",
 "inputs": [
  [
   0,
   3,
   5,
   7,
   3,
   5
  ]
 ],
 "hooks": [
  "L1.resid_post"
 ],
 "top_k": 4
}
```

Result (real output from the induction fixture):

```json
{
 "inputs": [
  {
   "index": 0,
   "tokens": [
    "<bos>",
    "t3",
    "t5",
    "t7",
    "t3",
    "t5"
   ],
   "ids": [
    0,
    3,
    5,
    7,
    3,
    5
   ]
  }
 ],
 "activations": [
  {
   "input": 0,
   "hook": "L1.resid_post",
   "shape": [
    6,
    64
   ],
   "units": 64,
   "norm_by_position": [
    28.49,
    26.53,
    28.59,
    26.16,
    24.35,
    20.27
   ],
   "frac_positive": 0.4688,
   "top_units_mean_abs": [
    [
     19,
     4.568
    ],
    [
     24,
     4.376
    ],
    [
     31,
     4.218
    ],
    [
     59,
     3.968
    ]
   ],
   "at_positions": [
    {
     "position": 5,
     "norm": 20.27,
     "top_units": [
      [
       19,
       -6.465
      ],
      [
       63,
       -6.259
      ],
      [
       43,
       -6.127
      ],
      [
       44,
       5.293
      ]
     ]
    }
   ]
  }
 ],
 "budget": {
  "used": 7,
  "total": 200,
  "remaining": 193
 }
}
```
