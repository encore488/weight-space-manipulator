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
 "session_id": "sess_47aafe2f419d",
 "patch_id": "p_bcf6a78805f41d36"
}
```

Result (real output from the induction fixture):

```json
{
 "reverted": "p_bcf6a78805f41d36",
 "also_reverted": [
  "p_610edaaea42a5287"
 ],
 "rebased": [],
 "stack": [
  "p_b4c4d027c65e9bec"
 ],
 "model_hash": "ec8aee16b0d759b7",
 "base_restored": false
}
```
