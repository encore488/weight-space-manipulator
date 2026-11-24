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
 "session_id": "sess_000000000003",
 "hook": "L0.resid_post",
 "unit": 3,
 "k": 2
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "hook": "L0.resid_post",
 "unit": 3,
 "features": [
  {
   "feature_id": "fd_e2bae00b39:18",
   "label": "previous token is 't25'",
   "fidelity": 0.2145,
   "decoder_weight_on_unit": -0.219,
   "encoder_weight_from_unit": -0.02127,
   "dict_id": "fd_e2bae00b39",
   "unexplained_fraction": 0.4985
  },
  {
   "feature_id": "fd_e2bae00b39:26",
   "label": "current token is 't26'",
   "fidelity": 0.9623,
   "decoder_weight_on_unit": -0.2072,
   "encoder_weight_from_unit": -0.01696,
   "dict_id": "fd_e2bae00b39",
   "unexplained_fraction": 0.4985
  }
 ]
}
```
