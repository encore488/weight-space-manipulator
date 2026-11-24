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
 "run_a": "run_0000000000000009",
 "run_b": "run_0000000000000001"
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "error": "compare_runs returns model-output information, which needs the 'outputs' capability (pass your session_id)",
 "code": "access_denied",
 "hint": "this restriction is set by the human operator for this experiment"
}
```
