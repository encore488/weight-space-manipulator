# `tensor_compare`

Compare two tensors (same or another session's workspace, e.g. base vs patched).

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `tensor` | `str` | required |
| `other_tensor` | `str | None` | `null` |
| `other_session_id` | `str | None` | `null` |
| `slice` | `str | None` | `null` |
| `against_base` | `bool` | `false` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call tensor_compare '<json>'`):

```json
{
 "session_id": "sess_47aafe2f419d",
 "tensor": "blocks.0.attn.q_proj.weight",
 "other_tensor": "blocks.1.attn.q_proj.weight"
}
```

Result (real output from the induction fixture):

```json
{
 "a": "blocks.0.attn.q_proj.weight",
 "b": "blocks.1.attn.q_proj.weight",
 "region": "all",
 "shape": [
  64,
  64
 ],
 "cosine": -0.0301,
 "rel_diff_norm": 1.495,
 "max_abs_diff": 0.5863,
 "frac_changed": 1.0,
 "identical": false,
 "top_k_subspace_overlap": 0.05386,
 "diff_energy_top1": 0.08028
}
```
