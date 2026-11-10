# `challenge_start`

Start a challenge run: opens the challenge's model sessions with its access level and budget, and returns the brief and answer format. agent_model = the model you are (e.g. 'claude-fable-5-2').

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `challenge_id` | `str` | required |
| `agent_model` | `str` | required |
| `label` | `str | None` | `null` |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call challenge_start '<json>'`):

```json
{
 "challenge_id": "induction-hunt",
 "agent_model": "claude-fable-5-2"
}
```

Result (real output from the induction fixture):

```json
{
 "run_id": "cr_93131b724c48",
 "challenge": "induction-hunt",
 "title": "Find the induction heads",
 "brief": "The model is a 2-layer, 4-heads-per-layer attention-only transformer trained on token sequences in which a random segment is repeated (plus 'counting' sequences). Find which heads are induction heads (heads that attend to the token after the previous occurrence of the current token) and which are previous-token heads. Heads are named 'L.H' (layer L, head H). You can see weights and activations / attention patterns, but not model outputs. You have 10 forward passes.",
 "access_level": "weights+activations",
 "budget_per_session": 10,
 "sessions": [
  {
   "role": "model",
   "session_id": "sess_cf3aa90acd0b"
  }
 ],
 "answer_format": {
  "induction_heads": [
   "L.H",
   "..."
  ],
  "previous_token_heads": [
   "L.H",
   "..."
  ]
 },
 "locked": false,
 "rules": "Use only the WSP tools. Do not read files on disk or open other models. Submit once with challenge_submit(run_id, answer)."
}
```
