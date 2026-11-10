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
 "error": "compare_runs returns model-output information, which needs the 'outputs' capability (pass your session_id)",
 "code": "access_denied",
 "hint": "this restriction is set by the human operator for this experiment"
}
```
