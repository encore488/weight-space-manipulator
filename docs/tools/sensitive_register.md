# `sensitive_register`

Mark a tensor, direction or feature as safety-relevant (human only).

**Who can call it:** human only (UI / HTTP API / CLI).

## Inputs

| Name | Type | Default |
|---|---|---|
| `kind` | `str` | required |
| `target` | `str` | required |
| `reason` | `str` | required |
| `model_hash` | `str` | `"*"` |
| `reviewer` | `str` | `"human"` |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call sensitive_register '<json>'`):

```json
{
 "kind": "tensor",
 "target": "unembed.weight",
 "reason": "example"
}
```

Result (real output from the induction fixture):

```json
{
 "id": "sens_c0a0d1eaa0",
 "kind": "tensor",
 "target": "unembed.weight",
 "model_hash": "*"
}
```
