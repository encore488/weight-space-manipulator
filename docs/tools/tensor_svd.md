# `tensor_svd`

Top-k singular values/vectors of a 2-D tensor, slice or head. Each singular vector is stored as a reusable direction id.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `tensor` | `str` | required |
| `k` | `int` | `4` |
| `slice` | `str | None` | `null` |
| `head` | `int | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call tensor_svd '<json>'`):

```json
{
 "session_id": "sess_47aafe2f419d",
 "tensor": "blocks.1.attn.o_proj.weight",
 "head": 2,
 "k": 2
}
```

Result (real output from the induction fixture):

```json
{
 "tensor": "blocks.1.attn.o_proj.weight",
 "region": "head 2 (cols 32:48)",
 "shape": [
  64,
  16
 ],
 "approximate": false,
 "singular_values": [
  1.527,
  1.395
 ],
 "energy_top_k": 0.2332,
 "effective_rank": 14.19,
 "stable_rank": 7.865,
 "components": [
  {
   "index": 0,
   "sigma": 1.527,
   "left_direction": "dir_a0758e962eb09fe6",
   "right_direction": "dir_3bd4ac71bdf21df1",
   "left_top_entries": [
    [
     25,
     -0.2802
    ],
    [
     54,
     -0.2543
    ],
    [
     63,
     -0.2484
    ],
    [
     47,
     -0.2444
    ],
    [
     26,
     -0.2268
    ],
    [
     48,
     -0.2179
    ]
   ],
   "right_top_entries": [
    [
     7,
     0.4996
    ],
    [
     3,
     -0.4104
    ],
    [
     1,
     0.3713
    ],
    [
     9,
     -0.3189
    ],
    [
     2,
     0.2672
    ],
    [
     0,
     -0.2633
    ]
   ]
  },
  {
   "index": 1,
   "sigma": 1.395,
   "left_direction": "dir_4b8d79dcfddc90d0",
   "right_direction": "dir_96f55046d70561c5",
   "left_top_entries": [
    [
     29,
     -0.3107
    ],
    [
     22,
     0.2593
    ],
    [
     62,
     0.2323
    ],
    [
     12,
     0.223
    ],
    [
     39,
     0.2186
    ],
    [
     35,
     -0.2154
    ]
   ],
   "right_top_entries": [
    [
     4,
     0.445
    ],
    [
     9,
     0.4156
    ],
    [
     11,
     -0.3256
    ],
    [
     0,
     -0.2929
    ],
    [
     13,
     -0.2879
    ],
    [
     10,
     -0.2642
    ]
   ]
  }
 ]
}
```
