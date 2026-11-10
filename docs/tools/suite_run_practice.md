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
 "session_id": "sess_87aaa3b360e8",
 "suite_id": "induction_practice",
 "limit": 2
}
```

Result (real output from the induction fixture):

```json
{
 "run_id": "run_a40927ee7de2ae51",
 "suite": "induction_practice",
 "split": "practice",
 "kind": "practice",
 "patch_id": "p_de781df671b58006",
 "model_hash": "ed5c84a209e1deb3",
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
  "python": "3.13.18",
  "torch": "2.15.0+cpu",
  "platform": "Linux-6.18.44-fc-v338-x86_64-with-glibc2.39",
  "wsp_toolkit": "0.2.0",
  "network": "blocked (thread)",
  "eval_batch_size": 1,
  "transformers": "5.22.2"
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
