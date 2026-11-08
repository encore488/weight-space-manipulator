# `zoo_scores`

Aggregate zoo scores per submitter, split, family and task type.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `submitter` | `str | None` | `null` |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call zoo_scores '<json>'`):

```json
{
 "submitter": "docs"
}
```

Result (real output from the induction fixture):

```json
{
 "rows": [
  {
   "submitter": "docs",
   "split": "public",
   "family": "mlp_compiled",
   "task_type": "identify",
   "n": 1,
   "score": 0.0
  }
 ]
}
```
