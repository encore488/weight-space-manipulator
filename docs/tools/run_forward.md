# `run_forward`

Run the model on inputs; returns top-k next-token predictions (LMs) or outputs. Needs 'outputs' access; costs 1 pass per input.

**Who can call it:** agents (MCP) and humans.

**Access / budget:** needs `outputs`; 1 pass per input.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `inputs` | `list` | required |
| `top_k` | `int` | `5` |
| `positions` | `Any` | `"last"` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).
At most `limits.max_batch_inputs` = 32 inputs per call and `limits.max_seq_len` = 512 tokens per input.

## Example

Call (MCP tool arguments, or `wsp call run_forward '<json>'`):

```json
{
 "session_id": "sess_87aaa3b360e8",
 "inputs": [
  "<bos> t3 t5 t7 t3 t5"
 ],
 "top_k": 3
}
```

Result (real output from the induction fixture):

```json
{
 "results": [
  {
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
   ],
   "next_token_predictions": [
    {
     "position": 5,
     "top": [
      {
       "token": "t7",
       "id": 7,
       "p": 0.7444
      },
      {
       "token": "t3",
       "id": 3,
       "p": 0.1373
      },
      {
       "token": "t28",
       "id": 28,
       "p": 0.04185
      }
     ]
    }
   ]
  }
 ],
 "budget": {
  "used": 5,
  "total": 200,
  "remaining": 195
 }
}
```
