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
 "session_id": "sess_04f94eb4cf68"
}
```

Result (real output from the induction fixture):

```json
{
 "events": [
  {
   "id": 1,
   "ts": 1794278879.6768825,
   "actor": "agent",
   "kind": "tool",
   "tool": "workspace_open",
   "ok": true,
   "summary": "workspace_id=ws_97c6759cac8b, session_id=sess_04f94eb4cf68, access_level=weights+activations+outputs, capabilities=[3], budget={...}, model_kind=tiny_transforme"
  },
  {
   "id": 3,
   "ts": 1794278879.6923456,
   "actor": "agent",
   "kind": "edit",
   "tool": "edit_apply",
   "ok": true,
   "summary": "applied p_b91197ce4594d918 (1024 params)"
  },
  {
   "id": 4,
   "ts": 1794278880.4847763,
   "actor": "system",
   "kind": "eval",
   "tool": "patch_test",
   "ok": true,
   "summary": "[{\"suite\": \"counting_regression\", \"split\": \"regression\", \"before\": 1.0, \"after\": 1.0, \"delta\": 0.0}, {\"suite\": \"induction_regression\", \"split\": \"regression\", \"b"
  },
  {
   "id": 5,
   "ts": 1794278880.4854274,
   "actor": "agent",
   "kind": "tool",
   "tool": "edit_apply",
   "ok": true,
   "summary": "patch_id=p_b91197ce4594d918, status=draft, params_changed=1024, pct_of_model=2.667, tensors=[1], model_hash_after=ec8aee16b0d759b7, sensitive_flags=[0], stack=["
  },
  {
   "id": 6,
   "ts": 1794278880.4899795,
   "actor": "agent",
   "kind": "edit",
   "tool": "edit_apply",
   "ok": true,
   "summary": "applied p_bb44323a1eb5f4aa (1024 params)"
  },
  {
   "id": 7,
   "ts": 1794278880.5304596,
   "actor": "system",
   "kind": "eval",
   "tool": "patch_test",
   "ok": true,
   "summary": "[{\"suite\": \"counting_regression\", \"split\": \"regression\", \"before\": 1.0, \"after\": 1.0, \"delta\": 0.0}, {\"suite\": \"induction_regression\", \"split\": \"regression\", \"b"
  },
  {
   "id": 8,
   "ts": 1794278880.5310147,
   "actor": "agent",
   "kind": "tool",
   "tool": "edit_apply",
   "ok": true,
   "su
  ... (truncated for the docs)
```
