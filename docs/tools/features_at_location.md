# `features_at_location`

Which features do these weights write or read? (tensor + optional slice)

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `tensor` | `str` | required |
| `index` | `str | None` | `null` |
| `k` | `int` | `5` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call features_at_location '<json>'`):

```json
{
 "session_id": "sess_04f94eb4cf68",
 "tensor": "blocks.0.attn.o_proj.weight",
 "index": ":, 16:32",
 "k": 2
}
```

Result (real output from the induction fixture):

```json
{
 "tensor": "blocks.0.attn.o_proj.weight",
 "region": ":, 16:32",
 "results": [
  {
   "feature_id": "fd_adc2dceaaa:30",
   "label": "previous token is 't17'",
   "fidelity": 0.2617,
   "hook": "L0.resid_post",
   "score": 1.041,
   "how": "selected weights write along the feature (|d^T W[:, cols]|)",
   "unexplained_fraction": 0.4985
  },
  {
   "feature_id": "fd_adc2dceaaa:15",
   "label": "inside a induction sequence",
   "fidelity": 0.297,
   "hook": "L0.resid_post",
   "score": 0.9968,
   "how": "selected weights write along the feature (|d^T W[:, cols]|)",
   "unexplained_fraction": 0.4985
  },
  {
   "feature_id": "fd_1a07c473b1:25",
   "label": "current token is 't25'",
   "fidelity": 0.3535,
   "hook": "L1.resid_post",
   "score": 0.7282,
   "how": "selected weights write along the feature (|d^T W[:, cols]|)",
   "unexplained_fraction": 0.4545
  },
  {
   "feature_id": "fd_1a07c473b1:22",
   "label": "previous token is '<bos>'",
   "fidelity": 0.3863,
   "hook": "L1.resid_post",
   "score": 0.7104,
   "how": "selected weights write along the feature (|d^T W[:, cols]|)",
   "unexplained_fraction": 0.4545
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
