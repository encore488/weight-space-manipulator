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
 "session_id": "sess_87aaa3b360e8",
 "patch_id": "p_de781df671b58006"
}
```

Result (real output from the induction fixture):

```json
{
 "stack": [
  "p_8b4e424ecd90b3a7",
  "p_de781df671b58006"
 ],
 "model_hash": "ed5c84a209e1deb3"
}
```
