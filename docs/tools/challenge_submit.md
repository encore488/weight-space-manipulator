# `challenge_submit`

Submit your answer for a challenge run (once). Returns the score.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `run_id` | `str` | required |
| `answer` | `dict` | required |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call challenge_submit '<json>'`):

```json
{
 "run_id": "cr_e2c1ebbd82e3",
 "answer": {}
}
```

Result (real output from the induction fixture):

```json
{
 "run_id": "cr_e2c1ebbd82e3",
 "challenge": "blind-surgeon",
 "seed": 140298,
 "score": 0.128,
 "detail": {
  "copy_accuracy": 0.1497,
  "counting_accuracy": 0.8551,
  "patches_applied": 0,
  "reference": "healthy model copies ~0.98; damaged instances copy < 0.3"
 },
 "usage": {
  "forward_passes": 0,
  "tool_calls": 0,
  "minutes": 0.001505
 }
}
```
