# `patch_reject`

Reject a patch (human only).

**Who can call it:** human only (UI / HTTP API / CLI).

## Inputs

| Name | Type | Default |
|---|---|---|
| `patch_id` | `str` | required |
| `reviewer` | `str` | required |
| `note` | `str` | `""` |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call patch_reject '<json>'`):

```json
{
 "patch_id": "p_0000000000000002",
 "reviewer": "alice",
 "note": "not needed"
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "patch_id": "p_0000000000000002",
 "status": "rejected",
 "reviewer": "alice",
 "ts": 0.0
}
```
