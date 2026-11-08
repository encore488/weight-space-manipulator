# `feature_neighbors`

Zoom parent/children, cross-layer links and similar features.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `feature_id` | `str` | required |
| `k` | `int` | `5` |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call feature_neighbors '<json>'`):

```json
{
 "feature_id": "fd_adc2dceaaa:13",
 "k": 2
}
```

Result (real output from the induction fixture):

```json
{
 "feature_id": "fd_adc2dceaaa:13",
 "label": "previous token is '<bos>'",
 "fidelity": 0.7059,
 "unexplained_fraction": 0.4985,
 "zoom_parent": null,
 "zoom_children": [],
 "upstream": [],
 "downstream": [
  [
   "fd_1a07c473b1:22",
   0.5249
  ],
  [
   "fd_1a07c473b1:29",
   0.1389
  ],
  [
   "fd_1a07c473b1:0",
   0.09628
  ]
 ],
 "similar_in_dictionary": [
  {
   "feature_id": "fd_adc2dceaaa:0",
   "label": "first token (BOS / position 0)",
   "cosine": 0.4012,
   "fidelity": 0.9796
  },
  {
   "feature_id": "fd_adc2dceaaa:22",
   "label": "position 21",
   "cosine": 0.2613,
   "fidelity": 0.7956
  }
 ]
}
```
