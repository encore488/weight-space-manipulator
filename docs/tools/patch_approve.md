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
 "patch_id": "p_b4c4d027c65e9bec",
 "reviewer": "alice",
 "note": "looks fine"
}
```

Result (real output from the induction fixture):

```json
{
 "patch_id": "p_b4c4d027c65e9bec",
 "status": "approved",
 "reviewer": "alice",
 "ts": 1794171073.2994192
}
```
