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
 "session_id": "sess_87aaa3b360e8",
 "patch_id": "p_de781df671b58006"
}
```

Result (real output from the induction fixture):

```json
{
 "reverted": "p_de781df671b58006",
 "also_reverted": [
  "p_2817426f2b9560d8"
 ],
 "rebased": [],
 "stack": [
  "p_8b4e424ecd90b3a7"
 ],
 "model_hash": "ec8aee16b0d759b7",
 "base_restored": false
}
```
