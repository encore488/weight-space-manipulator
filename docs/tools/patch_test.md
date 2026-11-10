# `patch_test`

Run the required suites (regression etc.) on parent vs patched; marks the patch 'tested'.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `patch_id` | `str` | required |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call patch_test '<json>'`):

```json
{
 "patch_id": "p_8b4e424ecd90b3a7"
}
```

Result (real output from the induction fixture):

```json
{
 "patch_id": "p_8b4e424ecd90b3a7",
 "status": "tested",
 "note": "suites ran; results are hidden at this access level"
}
```
