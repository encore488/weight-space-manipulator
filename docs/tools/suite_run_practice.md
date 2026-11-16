# `suite_run_practice`

Run a practice suite on the current working copy. Costs 1 pass per example (x samples). Per-example results need 'outputs' access.

**Who can call it:** agents (MCP) and humans.

**Access / budget:** 1 pass per example (× samples); per-example details need `outputs`.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `suite_id` | `str` | required |
| `limit` | `int | None` | `null` |
| `settings` | `dict | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call suite_run_practice '<json>'`):

```json
{
 "session_id": "sess_000000000003",
 "suite_id": "induction_practice",
 "limit": 2
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "run_id": "run_0000000000000006",
 "suite": "induction_practice",
 "split": "practice",
 "kind": "practice",
 "patch_id": "p_0000000000000002",
 "model_hash": "cdd2f8c2c3088c78",
 "settings": {
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 0,
  "max_new_tokens": 64,
  "repetition_penalty": 1.0,
  "presence_penalty": 0.0,
  "thinking": false,
  "seed": 0,
  "samples": 1,
  "extra": {}
 },
 "metrics": {
  "n": 2,
  "score": 1.0,
  "stderr": 0.0,
  "token_accuracy": 1.0
 },
 "env": {
  "python": "3.12.12",
  "torch": "2.9.1+cpu",
  "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39",
  "wsp_toolkit": "0.3.0",
  "network": "blocked (thread)",
  "eval_batch_size": 1,
  "transformers": "5.23.0"
 },
 "cached": false,
 "details": [
  {
   "score": 1.0,
   "correct": 9,
   "total": 9
  },
  {
   "score": 1.0,
   "correct": 7,
   "total": 7
  }
 ],
 "budget": {
  "used": 18,
  "total": 200,
  "remaining": 182
 }
}
```
