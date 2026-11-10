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
 "session_id": "sess_87aaa3b360e8"
}
```

Result (real output from the induction fixture):

```json
{
 "entries": []
}
```
