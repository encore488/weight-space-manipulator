# `session_start`

Start a new budgeted session on an existing workspace.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `workspace_id` | `str` | required |
| `access_level` | `str | None` | `null` |
| `budget` | `int | None` | `null` |
| `label` | `str | None` | `null` |
| `agent_model` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call session_start '<json>'`):

```json
{
 "workspace_id": "ws_f49ed5640a2b",
 "budget": 10
}
```

Result (real output from the induction fixture):

```json
{
 "workspace_id": "ws_f49ed5640a2b",
 "session_id": "sess_2fd9939a5d0d",
 "access_level": "weights+activations+outputs",
 "capabilities": [
  "activations",
  "outputs",
  "weights"
 ],
 "budget": {
  "used": 0,
  "total": 10,
  "remaining": 10
 }
}
```
