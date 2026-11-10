# `neuron_features`

Features linked to one unit of a hooked space, with decoder and encoder weights (both directions of the neuron<->feature overlap).

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `hook` | `str` | required |
| `unit` | `int` | required |
| `k` | `int` | `5` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call neuron_features '<json>'`):

```json
{
 "session_id": "sess_04f94eb4cf68",
 "hook": "L0.resid_post",
 "unit": 3,
 "k": 2
}
```

Result (real output from the induction fixture):

```json
{
 "hook": "L0.resid_post",
 "unit": 3,
 "features": [
  {
   "feature_id": "fd_adc2dceaaa:18",
   "label": "previous token is 't25'",
   "fidelity": 0.2145,
   "decoder_weight_on_unit": -0.219,
   "encoder_weight_from_unit": -0.02137,
   "dict_id": "fd_adc2dceaaa",
   "unexplained_fraction": 0.4985
  },
  {
   "feature_id": "fd_adc2dceaaa:26",
   "label": "current token is 't26'",
   "fidelity": 0.9623,
   "decoder_weight_on_unit": -0.2073,
   "encoder_weight_from_unit": -0.01693,
   "dict_id": "fd_adc2dceaaa",
   "unexplained_fraction": 0.4985
  }
 ]
}
```
