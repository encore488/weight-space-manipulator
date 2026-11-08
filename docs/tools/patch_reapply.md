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
 "session_id": "sess_47aafe2f419d",
 "patch_id": "p_bcf6a78805f41d36"
}
```

Result (real output from the induction fixture):

```json
{
 "stack": [
  "p_b4c4d027c65e9bec",
  "p_bcf6a78805f41d36"
 ],
 "model_hash": "ed5c84a209e1deb3"
}
```
