# `session_log`

The session's event log (what has been done, in order).

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `after_id` | `int` | `0` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call session_log '<json>'`):

```json
{
 "session_id": "sess_000000000003"
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "events": [
  {
   "id": 1,
   "ts": 0.0,
   "actor": "agent",
   "kind": "tool",
   "tool": "workspace_open",
   "ok": true,
   "summary": "workspace_id=ws_000000000002, session_id=sess_000000000003, access_level=weights+activations+outputs, capabilities=[3], budget={...}, model_kind=tiny_transforme"
  },
  {
   "id": 3,
   "ts": 0.0,
   "actor": "agent",
   "kind": "edit",
   "tool": "edit_apply",
   "ok": true,
   "summary": "applied p_0000000000000001 (1024 params)"
  },
  {
   "id": 4,
   "ts": 0.0,
   "actor": "system",
   "kind": "eval",
   "tool": "patch_test",
   "ok": true,
   "summary": "[{\"suite\": \"counting_regression\", \"split\": \"regression\", \"before\": 1.0, \"after\": 1.0, \"delta\": 0.0}, {\"suite\": \"induction_regression\", \"split\": \"regression\", \"b"
  },
  {
   "id": 5,
   "ts": 0.0,
   "actor": "agent",
   "kind": "tool",
   "tool": "edit_apply",
   "ok": true,
   "summary": "patch_id=p_0000000000000001, status=draft, params_changed=1024, pct_of_model=2.667, tensors=[1], model_hash_after=61c0c7aa6ceb6c13, sensitive_flags=[0], stack=["
  },
  {
   "id": 6,
   "ts": 0.0,
   "actor": "agent",
   "kind": "edit",
   "tool": "edit_apply",
   "ok": true,
   "summary": "applied p_0000000000000002 (1024 params)"
  },
  {
   "id": 7,
   "ts": 0.0,
   "actor": "system",
   "kind": "eval",
   "tool": "patch_test",
   "ok": true,
   "summary": "[{\"suite\": \"counting_regression\", \"split\": \"regression\", \"before\": 1.0, \"after\": 1.0, \"delta\": 0.0}, {\"suite\": \"induction_regression\", \"split\": \"regression\", \"b"
  },
  {
   "id": 8,
   "ts": 0.0,
   "actor": "agent",
   "kind": "tool",
   "tool": "edit_apply",
   "ok": true,
   "summary": "patch_id=p_0000000000000002, status=draft, params_changed=1024, pct_of_model=2.667, tensors=[1],
  ... (truncated for the docs)
```
