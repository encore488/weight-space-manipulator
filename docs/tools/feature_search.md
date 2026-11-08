# `feature_search`

Search the feature index by meaning (label text). Returns fidelity and the dictionaries' unexplained fraction.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `query` | `str` | required |
| `k` | `int` | `10` |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call feature_search '<json>'`):

```json
{
 "query": "previous token",
 "k": 3
}
```

Result (real output from the induction fixture):

```json
{
 "query": "previous token",
 "results": [
  {
   "feature_id": "fd_adc2dceaaa:13",
   "label": "previous token is '<bos>'",
   "fidelity": 0.7059,
   "sensitive": false,
   "dict_id": "fd_adc2dceaaa",
   "unexplained_fraction": 0.4985,
   "match_score": 8.159
  },
  {
   "feature_id": "fd_1a07c473b1:22",
   "label": "previous token is '<bos>'",
   "fidelity": 0.3863,
   "sensitive": false,
   "dict_id": "fd_1a07c473b1",
   "unexplained_fraction": 0.4545,
   "match_score": 8.127
  },
  {
   "feature_id": "fd_adc2dceaaa:25",
   "label": "previous token is 't24'",
   "fidelity": 0.303,
   "sensitive": false,
   "dict_id": "fd_adc2dceaaa",
   "unexplained_fraction": 0.4985,
   "match_score": 8.119
  }
 ],
 "dictionaries": {
  "fd_adc2dceaaa": {
   "hook": "L0.resid_post",
   "size": 32,
   "unexplained_fraction": 0.4985
  },
  "fd_1a07c473b1": {
   "hook": "L1.resid_post",
   "size": 32,
   "unexplained_fraction": 0.4545
  }
 }
}
```
