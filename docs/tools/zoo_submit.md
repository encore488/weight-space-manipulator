# `zoo_submit`

Submit an answer for a zoo task. Public split returns the score; hidden split is scored server-side and only aggregated.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `net_id` | `str` | required |
| `task_type` | `str` | required |
| `answer` | `dict` | required |
| `submitter` | `str` | `"anonymous"` |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call zoo_submit '<json>'`):

```json
{
 "net_id": "z4759a4880a",
 "task_type": "identify",
 "answer": {
  "label": "and_01"
 },
 "submitter": "docs"
}
```

Result (real output from the induction fixture):

```json
{
 "recorded": true,
 "split": "public",
 "score": 0.0
}
```
