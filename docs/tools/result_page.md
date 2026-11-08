# `result_page`

Fetch page N of a large result that was returned as a handle.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `handle` | `str` | required |
| `page` | `int` | `0` |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call result_page '<json>'`):

```json
{
 "handle": "res_10341050ccea288ea5ed",
 "page": 1
}
```

Result (real output from the induction fixture):

```json
{
 "handle": "res_10341050ccea288ea5ed",
 "page": 1,
 "pages": 8,
 "paged_field": "heads",
 "items": [
  {
   "attn_to_first": 0.2709,
   "head": "1.2",
   "input": 0,
   "mean_attn_by_offset": [
    0.1196,
    0.1261,
    0.0101,
    0.09077,
    0.03121,
    0.07024,
    0.1131,
    0.05956
   ],
   "mean_entropy": 0.7787,
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
       0.961
      ],
      [
       1,
       0.03897
      ]
     ]
    ],
    [
     2,
     [
      [
       2,
       0.6959
      ],
      [
       1,
       0.2365
      ],
      [
       0,
       0.06765
      ]
     ]
    ],
    [
     3,
     [
      [
       0,
       0.9775
      ],
      [
       2,
       0.01178
      ],
      [
       1,
       0.01059
      ]
     ]
    ],
    [
     4,
     [
      [
       3,
       0.8647
      ],
      [
       1,
       0.1001
      ],
      [
       2,
       0.02016
      ]
     ]
    ],
    [
     5,
     [
      [
       0,
       0.5024
      ],
      [
       1,
       0.3962
      ],
      [
       2,
       0.04733
      ]
     ]
    ],
    [
     6,
     [
      [
       0,
       0.8724
      ],
      [
       2,
       0.106
      ],
      [
       1,
       0.01309
      ]
     ]
    ],
    [
     7,
     [
      [
       7,
       0.9363
      ],
      [
       1,
       0.02282
      ],
      [
       0,
       0.01837
      ]
     ]
    ],
    [
     8,
     [
      [
       2,
       0.9689
      ],
      [
       5,
       0.007873
      ],
      [
       1,
       0.007805
      ]
     ]
    ],
    [
     9,
     [
      [
       0,
       0.5987
      ],
      [
       2,
       0.1794
      ],
      [
       4,
       0.1555
      ]
     
  ... (truncated for the docs)
```
