# `direction_from_activations`

Difference of mean activations (positive minus negative inputs) at a hook, stored as a direction. Needs 'activations' access; 1 pass per input.

**Who can call it:** agents (MCP) and humans.

**Access / budget:** needs `activations`; 1 pass per input (positive + negative).

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `positive_inputs` | `list` | required |
| `negative_inputs` | `list` | required |
| `hook` | `str` | required |
| `positions` | `Any` | `"last"` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call direction_from_activations '<json>'`):

```json
{
 "session_id": "sess_04f94eb4cf68",
 "positive_inputs": [
  [
   0,
   3,
   5,
   3,
   5
  ]
 ],
 "negative_inputs": [
  [
   0,
   3,
   5,
   9,
   11
  ]
 ],
 "hook": "L1.resid_post"
}
```

Result (real output from the induction fixture):

```json
{
 "direction": "dir_6363ebea184f10a3",
 "hook": "L1.resid_post",
 "dim": 64,
 "raw_norm": 26.86,
 "separation_d_prime": null,
 "positive_projection_mean": 16.64,
 "negative_projection_mean": -10.23,
 "top_units": [
  [
   56,
   9.46
  ],
  [
   19,
   -7.986
  ],
  [
   27,
   -7.113
  ],
  [
   17,
   5.892
  ],
  [
   43,
   -5.828
  ],
  [
   33,
   -5.79
  ],
  [
   5,
   5.356
  ],
  [
   51,
   5.328
  ]
 ],
 "budget": {
  "used": 9,
  "total": 200,
  "remaining": 191
 }
}
```
