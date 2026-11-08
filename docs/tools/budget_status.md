# `budget_status`

Remaining forward-pass budget and held-out evaluation allowance for the session.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call budget_status '<json>'`):

```json
{
 "session_id": "sess_47aafe2f419d"
}
```

Result (real output from the induction fixture):

```json
{
 "session_id": "sess_47aafe2f419d",
 "forward_passes": {
  "used": 4,
  "total": 200,
  "remaining": 196
 },
 "heldout_evals_remaining_this_hour": 5,
 "access_level": "weights+activations+outputs"
}
```
