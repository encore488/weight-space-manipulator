# `patch_reapply`

Re-apply an existing patch (whose parent is the current top of the stack).

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `patch_id` | `str` | required |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call patch_reapply '<json>'`):

```json
{
 "session_id": "sess_000000000003",
 "patch_id": "p_0000000000000002"
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "stack": [
  "p_0000000000000001",
  "p_0000000000000002"
 ],
 "model_hash": "cdd2f8c2c3088c78"
}
```
