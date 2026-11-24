# `patch_list`

List patches (optionally for this session's workspace or by status).

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str | None` | `null` |
| `status` | `str | None` | `null` |
| `all_workspaces` | `bool` | `false` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call patch_list '<json>'`):

```json
{
 "session_id": "sess_000000000003"
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "patches": [
  {
   "patch_id": "p_0000000000000003",
   "workspace_id": "ws_000000000002",
   "parent": "p_0000000000000002",
   "status": "tested",
   "author": "agent:sess_000000000003",
   "created_at": 0.0,
   "params_changed": 1024,
   "pct_of_model": 2.66667,
   "ops": [
    "scale"
   ],
   "flagged": false,
   "rationale": "example: shrink head 1.0 output by 5%"
  },
  {
   "patch_id": "p_0000000000000002",
   "workspace_id": "ws_000000000002",
   "parent": "p_0000000000000001",
   "status": "tested",
   "author": "agent:sess_000000000003",
   "created_at": 0.0,
   "params_changed": 1024,
   "pct_of_model": 2.66667,
   "ops": [
    "scale"
   ],
   "flagged": false,
   "rationale": "doc patch B"
  },
  {
   "patch_id": "p_0000000000000001",
   "workspace_id": "ws_000000000002",
   "parent": null,
   "status": "tested",
   "author": "agent:sess_000000000003",
   "created_at": 0.0,
   "params_changed": 1024,
   "pct_of_model": 2.66667,
   "ops": [
    "scale"
   ],
   "flagged": false,
   "rationale": "doc patch A"
  }
 ],
 "applied_stack": [
  "p_0000000000000001",
  "p_0000000000000002",
  "p_0000000000000003"
 ]
}
```
