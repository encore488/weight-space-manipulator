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
 "patch_id": "p_de781df671b58006",
 "reviewer": "alice",
 "note": "not needed"
}
```

Result (real output from the induction fixture):

```json
{
 "patch_id": "p_de781df671b58006",
 "status": "rejected",
 "reviewer": "alice",
 "ts": 1794315537.708149
}
```
