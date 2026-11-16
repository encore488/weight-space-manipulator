# `patch_approve`

Approve a tested patch (human only).

**Who can call it:** human only (UI / HTTP API / CLI).

## Inputs

| Name | Type | Default |
|---|---|---|
| `patch_id` | `str` | required |
| `reviewer` | `str` | required |
| `note` | `str` | `""` |
| `acknowledge_flags` | `bool` | `false` |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call patch_approve '<json>'`):

```json
{
 "patch_id": "p_0000000000000001",
 "reviewer": "alice",
 "note": "looks fine"
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "patch_id": "p_0000000000000001",
 "status": "approved",
 "reviewer": "alice",
 "ts": 0.0
}
```
