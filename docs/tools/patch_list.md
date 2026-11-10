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
 "session_id": "sess_04f94eb4cf68"
}
```

Result (real output from the induction fixture):

```json
{
 "patches": [
  {
   "patch_id": "p_f71b91ca6396e312",
   "workspace_id": "ws_97c6759cac8b",
   "parent": "p_bb44323a1eb5f4aa",
   "status": "tested",
   "author": "agent:sess_04f94eb4cf68",
   "created_at": 1794278889.0984066,
   "params_changed": 1024,
   "pct_of_model": 2.66667,
   "ops": [
    "scale"
   ],
   "flagged": false,
   "rationale": "example: shrink head 1.0 output by 5%"
  },
  {
   "patch_id": "p_bb44323a1eb5f4aa",
   "workspace_id": "ws_97c6759cac8b",
   "parent": "p_b91197ce4594d918",
   "status": "tested",
   "author": "agent:sess_04f94eb4cf68",
   "created_at": 1794278880.4874792,
   "params_changed": 1024,
   "pct_of_model": 2.66667,
   "ops": [
    "scale"
   ],
   "flagged": false,
   "rationale": "doc patch B"
  },
  {
   "patch_id": "p_b91197ce4594d918",
   "workspace_id": "ws_97c6759cac8b",
   "parent": null,
   "status": "tested",
   "author": "agent:sess_04f94eb4cf68",
   "created_at": 1794278879.6858296,
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
  "p_b91197ce4594d918",
  "p_bb44323a1eb5f4aa",
  "p_f71b91ca6396e312"
 ]
}
```
