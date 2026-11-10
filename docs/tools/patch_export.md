# `patch_export`

Export the patched model (base + patch chain) to safetensors. Blocked unless every patch in the chain is approved by a human.

**Who can call it:** agents (MCP) and humans.

**Access / budget:** blocked unless every patch in the chain is human-approved.

## Inputs

| Name | Type | Default |
|---|---|---|
| `patch_id` | `str` | required |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call patch_export '<json>'`):

```json
{
 "patch_id": "p_b91197ce4594d918"
}
```

Result (real output from the induction fixture):

```json
{
 "error": "export blocked: patch p_b91197ce4594d918 in the chain is 'tested', not 'approved'",
 "code": "export_blocked",
 "hint": "a human must approve it in the UI after it is tested"
}
```
