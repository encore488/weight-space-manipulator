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
 "handle": "res_00000000000000000001",
 "page": 1
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "handle": "res_00000000000000000001",
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
    0.09081,
    0.03122,
    0.07026,
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
       0.9611
      ],
      [
       1,
       0.03889
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
       0.2364
      ],
      [
       0,
       0.06776
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
       0.01177
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
       0.8649
      ],
      [
       1,
       0.1
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
       0.5025
      ],
      [
       1,
       0.3962
      ],
      [
       2,
       0.04727
      ]
     ]
    ],
    [
     6,
     [
      [
       0,
       0.8723
      ],
      [
       2,
       0.1061
      ],
      [
       1,
       0.0131
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
       0.0228
      ],
      [
       0,
       0.01839
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
       0.007869
      ],
      [
       1,
       0.007797
      ]
     ]
    ],
    [
     9,
     [
      [
       0,
       0.5989
      ],
      [
       2,
       0.1792
      ],
      [
       4,
       0.1555
      ]
     ]
  ... (truncated for the docs)
```
