# `feature_dictionaries`

List feature dictionaries (hook, size, unexplained fraction).

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call feature_dictionaries '<json>'`):

```json
{
 "session_id": "sess_04f94eb4cf68"
}
```

Result (real output from the induction fixture):

```json
{
 "dictionaries": [
  {
   "id": "fd_adc2dceaaa",
   "model_hash": "1b441d92f299953d64ee9a233753978c72dfc1a7e848b0baabfba39741ba072e",
   "hook": "L0.resid_post",
   "size": 32,
   "parent_dict": null,
   "unexplained_fraction": 0.4985,
   "mean_l0": 2.613
  },
  {
   "id": "fd_1a07c473b1",
   "model_hash": "1b441d92f299953d64ee9a233753978c72dfc1a7e848b0baabfba39741ba072e",
   "hook": "L1.resid_post",
   "size": 32,
   "parent_dict": null,
   "unexplained_fraction": 0.4545,
   "mean_l0": 4.134
  }
 ]
}
```
