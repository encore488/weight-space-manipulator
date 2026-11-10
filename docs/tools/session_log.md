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
 "session_id": "sess_87aaa3b360e8"
}
```

Result (real output from the induction fixture):

```json
{
 "events": [
  {
   "id": 1,
   "ts": 1794315523.5122378,
   "actor": "agent",
   "kind": "tool",
   "tool": "workspace_open",
   "ok": true,
   "summary": "workspace_id=ws_f5367d180411, session_id=sess_87aaa3b360e8, access_level=weights+activations+outputs, capabilities=[3], budget={...}, model_kind=tiny_transforme"
  },
  {
   "id": 3,
   "ts": 1794315523.527085,
   "actor": "agent",
   "kind": "edit",
   "tool": "edit_apply",
   "ok": true,
   "summary": "applied p_8b4e424ecd90b3a7 (1024 params)"
  },
  {
   "id": 4,
   "ts": 1794315524.571487,
   "actor": "system",
   "kind": "eval",
   "tool": "patch_test",
   "ok": true,
   "summary": "[{\"suite\": \"counting_regression\", \"split\": \"regression\", \"before\": 1.0, \"after\": 1.0, \"delta\": 0.0}, {\"suite\": \"induction_regression\", \"split\": \"regression\", \"b"
  },
  {
   "id": 5,
   "ts": 1794315524.5723703,
   "actor": "agent",
   "kind": "tool",
   "tool": "edit_apply",
   "ok": true,
   "summary": "patch_id=p_8b4e424ecd90b3a7, status=draft, params_changed=1024, pct_of_model=2.667, tensors=[1], model_hash_after=ec8aee16b0d759b7, sensitive_flags=[0], stack=["
  },
  {
   "id": 6,
   "ts": 1794315524.581606,
   "actor": "agent",
   "kind": "edit",
   "tool": "edit_apply",
   "ok": true,
   "summary": "applied p_de781df671b58006 (1024 params)"
  },
  {
   "id": 7,
   "ts": 1794315524.6395986,
   "actor": "system",
   "kind": "eval",
   "tool": "patch_test",
   "ok": true,
   "summary": "[{\"suite\": \"counting_regression\", \"split\": \"regression\", \"before\": 1.0, \"after\": 1.0, \"delta\": 0.0}, {\"suite\": \"induction_regression\", \"split\": \"regression\", \"b"
  },
  {
   "id": 8,
   "ts": 1794315524.6423345,
   "actor": "agent",
   "kind": "tool",
   "tool": "edit_apply",
   "ok": true,
   "summa
  ... (truncated for the docs)
```
