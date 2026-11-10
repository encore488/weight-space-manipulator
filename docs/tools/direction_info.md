# `direction_info`

Resolve a direction spec: dimension, norm-1 vector top entries, and a stored id.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `direction` | `dict` | required |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call direction_info '<json>'`):

```json
{
 "session_id": "sess_87aaa3b360e8",
 "direction": {
  "token": "t5",
  "matrix": "unembed"
 }
}
```

Result (real output from the induction fixture):

```json
{
 "id": "dir_1c1d3af5a2245a1b",
 "description": "unembed row of token 't5' (5)",
 "dim": 64,
 "top_entries": [
  [
   44,
   0.3424
  ],
  [
   15,
   0.2504
  ],
  [
   14,
   0.2371
  ],
  [
   20,
   -0.2298
  ],
  [
   48,
   -0.2021
  ],
  [
   58,
   -0.1977
  ],
  [
   63,
   0.1914
  ],
  [
   13,
   -0.1888
  ]
 ]
}
```
