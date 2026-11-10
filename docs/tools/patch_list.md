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
 "session_id": "sess_87aaa3b360e8"
}
```

Result (real output from the induction fixture):

```json
{
 "patches": [
  {
   "patch_id": "p_2817426f2b9560d8",
   "workspace_id": "ws_f5367d180411",
   "parent": "p_de781df671b58006",
   "status": "tested",
   "author": "agent:sess_87aaa3b360e8",
   "created_at": 1794315537.5389426,
   "params_changed": 1024,
   "pct_of_model": 2.66667,
   "ops": [
    "scale"
   ],
   "flagged": false,
   "rationale": "example: shrink head 1.0 output by 5%"
  },
  {
   "patch_id": "p_de781df671b58006",
   "workspace_id": "ws_f5367d180411",
   "parent": "p_8b4e424ecd90b3a7",
   "status": "tested",
   "author": "agent:sess_87aaa3b360e8",
   "created_at": 1794315524.5753467,
   "params_changed": 1024,
   "pct_of_model": 2.66667,
   "ops": [
    "scale"
   ],
   "flagged": false,
   "rationale": "doc patch B"
  },
  {
   "patch_id": "p_8b4e424ecd90b3a7",
   "workspace_id": "ws_f5367d180411",
   "parent": null,
   "status": "tested",
   "author": "agent:sess_87aaa3b360e8",
   "created_at": 1794315523.5221598,
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
  "p_8b4e424ecd90b3a7",
  "p_de781df671b58006",
  "p_2817426f2b9560d8"
 ]
}
```
