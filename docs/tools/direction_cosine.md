# `direction_cosine`

Cosine similarity between two directions of the same dimension.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `a` | `dict` | required |
| `b` | `dict` | required |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call direction_cosine '<json>'`):

```json
{
 "session_id": "sess_47aafe2f419d",
 "a": {
  "token": "t5"
 },
 "b": {
  "token": "t6"
 }
}
```

Result (real output from the induction fixture):

```json
{
 "a": "unembed row of token 't5' (5)",
 "b": "unembed row of token 't6' (6)",
 "cosine": -0.04801
}
```
