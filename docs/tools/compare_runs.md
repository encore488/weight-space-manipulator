# `compare_runs`

Per-metric comparison of two evaluation runs (with settings-mismatch warnings).

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `run_a` | `str` | required |
| `run_b` | `str` | required |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call compare_runs '<json>'`):

```json
{
 "run_a": "run_70db04baf18a45ad",
 "run_b": "run_75a1d2aeaf2b1be9"
}
```

Result (real output from the induction fixture):

```json
{
 "a": "run_70db04baf18a45ad",
 "b": "run_75a1d2aeaf2b1be9",
 "suite_a": "induction_practice",
 "suite_b": "induction_practice",
 "metrics": {
  "score": {
   "a": 0.02778,
   "b": 1.0,
   "delta": 0.9722
  },
  "stderr": {
   "a": 0.02778,
   "b": 0.0,
   "delta": -0.02778
  },
  "token_accuracy": {
   "a": 0.03448,
   "b": 1.0,
   "delta": 0.9655
  }
 },
 "score_delta_exceeds_2se": true,
 "warnings": []
}
```
