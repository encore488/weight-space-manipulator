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
 "session_id": "sess_47aafe2f419d"
}
```

Result (real output from the induction fixture):

```json
{
 "patches": [
  {
   "patch_id": "p_610edaaea42a5287",
   "workspace_id": "ws_f49ed5640a2b",
   "parent": "p_bcf6a78805f41d36",
   "status": "tested",
   "author": "agent:sess_47aafe2f419d",
   "created_at": 1794171073.220352,
   "params_changed": 1024,
   "pct_of_model": 2.66667,
   "ops": [
    "scale"
   ],
   "flagged": false,
   "rationale": "example: shrink head 1.0 output by 5%"
  },
  {
   "patch_id": "p_bcf6a78805f41d36",
   "workspace_id": "ws_f49ed5640a2b",
   "parent": "p_b4c4d027c65e9bec",
   "status": "tested",
   "author": "agent:sess_47aafe2f419d",
   "created_at": 1794171064.6940787,
   "params_changed": 1024,
   "pct_of_model": 2.66667,
   "ops": [
    "scale"
   ],
   "flagged": false,
   "rationale": "doc patch B"
  },
  {
   "patch_id": "p_b4c4d027c65e9bec",
   "workspace_id": "ws_f49ed5640a2b",
   "parent": null,
   "status": "tested",
   "author": "agent:sess_47aafe2f419d",
   "created_at": 1794171063.9440308,
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
  "p_b4c4d027c65e9bec",
  "p_bcf6a78805f41d36",
  "p_610edaaea42a5287"
 ]
}
```
