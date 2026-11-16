# `features_active_on`

Features active on given inputs, per position (needs 'activations'; 1 pass per input). Reports the unexplained fraction on these inputs.

**Who can call it:** agents (MCP) and humans.

**Access / budget:** needs `activations`; 1 pass per input.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `inputs` | `list` | required |
| `hook` | `str | None` | `null` |
| `k` | `int` | `3` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).
At most `limits.max_batch_inputs` = 32 inputs per call and `limits.max_seq_len` = 512 tokens per input.

## Example

Call (MCP tool arguments, or `wsp call features_active_on '<json>'`):

```json
{
 "session_id": "sess_000000000003",
 "inputs": [
  [
   0,
   5,
   9,
   3,
   5,
   9
  ]
 ],
 "hook": "L1.resid_post",
 "k": 1
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "results": [
  {
   "input": 0,
   "tokens": [
    "<bos>",
    "t5",
    "t9",
    "t3",
    "t5",
    "t9"
   ],
   "ids": [
    0,
    5,
    9,
    3,
    5,
    9
   ],
   "hooks": {
    "L1.resid_post": {
     "dict_id": "fd_95fe6301b7",
     "unexplained_fraction_this_input": 0.3223,
     "dictionary_unexplained_fraction": 0.4545,
     "positions": [
      {
       "position": 0,
       "features": [
        {
         "feature_id": "fd_95fe6301b7:0",
         "label": "position 0",
         "activation": 1.053,
         "fidelity": 0.5199
        }
       ]
      },
      {
       "position": 1,
       "features": [
        {
         "feature_id": "fd_95fe6301b7:5",
         "label": "current token is 't5'",
         "activation": 0.5607,
         "fidelity": 0.2986
        }
       ]
      },
      {
       "position": 2,
       "features": [
        {
         "feature_id": "fd_95fe6301b7:9",
         "label": "current token is 't9'",
         "activation": 0.5575,
         "fidelity": 0.3529
        }
       ]
      },
      {
       "position": 3,
       "features": [
        {
         "feature_id": "fd_95fe6301b7:15",
         "label": "inside a counting sequence",
         "activation": 0.2241,
         "fidelity": 0.2738
        }
       ]
      },
      {
       "position": 4,
       "features": [
        {
         "feature_id": "fd_95fe6301b7:8",
         "label": "current token is 't8'",
         "activation": 0.4096,
         "fidelity": 0.5087
        }
       ]
      },
      {
       "position": 5,
       "features": [
        {
         "feature_id": "fd_95fe6301b7:2",
         "label": "previous token is 't1'",
         "activation": 0.6122,
         "fidelity": 0.2703
        }
       ]
      }
     ]
    }
   }
  }
 ],
 "budget": {
  "use
  ... (truncated for the docs)
```
