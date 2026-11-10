# `probe_attention`

Attention-pattern summaries per head: entropy, attention by relative offset, top keys per query. Needs 'activations' access; 1 pass per input.

**Who can call it:** agents (MCP) and humans.

**Access / budget:** needs `activations`; 1 pass per input.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `inputs` | `list` | required |
| `layers` | `list[int] | None` | `null` |
| `heads` | `list[int] | None` | `null` |
| `top_k_keys` | `int` | `3` |
| `query_positions` | `Any` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).
At most `limits.max_batch_inputs` = 32 inputs per call and `limits.max_seq_len` = 512 tokens per input.

## Example

Call (MCP tool arguments, or `wsp call probe_attention '<json>'`):

```json
{
 "session_id": "sess_87aaa3b360e8",
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
 "layers": [
  1
 ],
 "heads": [
  2
 ],
 "top_k_keys": 2
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
 "heads": [
  {
   "input": 0,
   "head": "1.2",
   "mean_entropy": 0.3745,
   "attn_to_first": 0.5794,
   "mean_attn_by_offset": [
    0.2414,
    0.2045,
    0.6997,
    0.2091,
    0.05123,
    0.0002942
   ],
   "top_keys_by_query": [
    [
     0,
     [
      [
       0,
       1.0
      ]
     ]
    ],
    [
     1,
     [
      [
       0,
       0.9791
      ],
      [
       1,
       0.02087
      ]
     ]
    ],
    [
     2,
     [
      [
       0,
       0.8761
      ],
      [
       2,
       0.08784
      ]
     ]
    ],
    [
     3,
     [
      [
       0,
       0.5186
      ],
      [
       3,
       0.3388
      ]
     ]
    ],
    [
     4,
     [
      [
       2,
       0.785
      ],
      [
       1,
       0.1088
      ]
     ]
    ],
    [
     5,
     [
      [
       3,
       0.9996
      ],
      [
       0,
       0.0002942
      ]
     ]
    ]
   ]
  }
 ],
 "legend": "head 'L.H' = layer L head H; mean_attn_by_offset[o] = mean weight from query q to key q-o; top_keys_by_query = [query_pos, [[key_pos, weight], ...]]",
 "budget": {
  "used": 10,
  "total": 200,
  "remaining": 190
 }
}
```
