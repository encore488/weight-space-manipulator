# `evaluate_heldout`

Aggregate score on a held-out suite for the current working copy. Never returns examples; rate-limited; logged.

**Who can call it:** agents (MCP) and humans.

**Access / budget:** no pass cost; rate-limited per session (session.heldout_rate_limit_per_hour); aggregate only.

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

Call (MCP tool arguments, or `wsp call evaluate_heldout '<json>'`):

```json
{
 "session_id": "sess_04f94eb4cf68",
 "suite_id": "induction_heldout",
 "limit": 8
}
```

Result (real output from the induction fixture):

```json
{
 "run_id": "run_1fc919bfc7c99e60",
 "suite": "induction_heldout",
 "metrics": {
  "n": 8,
  "score": 0.9613,
  "stderr": 0.02543,
  "token_accuracy": 0.96
 },
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
 "on_patch": "p_bb44323a1eb5f4aa",
 "model_hash": "ed5c84a209e1deb3"
}
```
