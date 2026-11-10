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
 "run_id": "cr_d1e7ff81a9ab",
 "answer": {}
}
```

Result (real output from the induction fixture):

```json
{
 "run_id": "cr_d1e7ff81a9ab",
 "challenge": "blind-surgeon",
 "score": 0.0504,
 "detail": {
  "copy_accuracy": 0.0504,
  "counting_accuracy": 1.0,
  "patches_applied": 0,
  "reference": "healthy model copies ~0.98; the damaged model ~0.04",
  "rule": "score = copy accuracy, multiplied by counting accuracy if counting fell below 0.95"
 },
 "usage": {
  "forward_passes": 0,
  "tool_calls": 0,
  "minutes": 0.0008802
 }
}
```
