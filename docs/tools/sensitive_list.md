# `sensitive_list`

List registered sensitive components for this session's model.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call sensitive_list '<json>'`):

```json
{
 "session_id": "sess_000000000003"
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "entries": []
}
```
