# `generate`

Generate a continuation for each input (LMs). Needs 'outputs' access; 1 pass per input.

**Who can call it:** agents (MCP) and humans.

**Access / budget:** needs `outputs`; 1 pass per input.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `inputs` | `list` | required |
| `settings` | `dict | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).
At most `limits.max_batch_inputs` = 32 inputs per call and `limits.max_seq_len` = 512 tokens per input.

## Example

Call (MCP tool arguments, or `wsp call generate '<json>'`):

```json
{
 "session_id": "sess_87aaa3b360e8",
 "inputs": [
  [
   0,
   3,
   5,
   7,
   3
  ]
 ],
 "settings": {
  "max_new_tokens": 4
 }
}
```

Result (real output from the induction fixture):

```json
{
 "results": [
  {
   "text": "t5 t7 t3 t5",
   "new_tokens": 4,
   "hit_token_limit": true,
   "looped": false
  }
 ],
 "settings": {
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 0,
  "max_new_tokens": 4,
  "repetition_penalty": 1.0,
  "presence_penalty": 0.0,
  "thinking": false,
  "seed": 0,
  "samples": 1,
  "extra": {}
 },
 "budget": {
  "used": 6,
  "total": 200,
  "remaining": 194
 }
}
```
