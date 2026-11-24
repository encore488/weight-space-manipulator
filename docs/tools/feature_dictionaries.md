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
 "session_id": "sess_000000000003"
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "dictionaries": [
  {
   "id": "fd_e2bae00b39",
   "model_hash": "610d318e08449718477580a7d3db31b8b5f8d93d77605b0fb6e6fc355bc5e75b",
   "hook": "L0.resid_post",
   "size": 32,
   "parent_dict": null,
   "unexplained_fraction": 0.4985,
   "mean_l0": 2.613
  },
  {
   "id": "fd_95fe6301b7",
   "model_hash": "610d318e08449718477580a7d3db31b8b5f8d93d77605b0fb6e6fc355bc5e75b",
   "hook": "L1.resid_post",
   "size": 32,
   "parent_dict": null,
   "unexplained_fraction": 0.4545,
   "mean_l0": 4.091
  }
 ]
}
```
