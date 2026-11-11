# `narrate`

Optional: tell the humans watching what you are doing and why, in one or two sentences (shown live in the web UI and the arena replay). Free: no budget, not scored, never required.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `text` | `str` | required |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call narrate '<json>'`):

```json
{
 "session_id": "sess_5e16e3e6270a",
 "text": "Layer-1 output norms are 10x below layer 0; checking value norms next."
}
```

Result (real output from the induction fixture):

```json
{
 "shown": true,
 "chars": 70
}
```
