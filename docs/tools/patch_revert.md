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
 "session_id": "sess_04f94eb4cf68",
 "patch_id": "p_bb44323a1eb5f4aa"
}
```

Result (real output from the induction fixture):

```json
{
 "reverted": "p_bb44323a1eb5f4aa",
 "also_reverted": [
  "p_f71b91ca6396e312"
 ],
 "rebased": [],
 "stack": [
  "p_b91197ce4594d918"
 ],
 "model_hash": "ec8aee16b0d759b7",
 "base_restored": false
}
```
