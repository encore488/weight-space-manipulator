# `patch_revert`

Revert an applied patch. mode='cascade' (default) also reverts patches above it; 'rebase' re-derives the patches above it.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `patch_id` | `str` | required |
| `mode` | `str` | `"cascade"` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call patch_revert '<json>'`):

```json
{
 "session_id": "sess_000000000003",
 "patch_id": "p_0000000000000002"
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "reverted": "p_0000000000000002",
 "also_reverted": [
  "p_0000000000000003"
 ],
 "rebased": [],
 "stack": [
  "p_0000000000000001"
 ],
 "model_hash": "61c0c7aa6ceb6c13",
 "base_restored": false
}
```
