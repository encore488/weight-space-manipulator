# `edit_preview`

Show what a list of edit operations would change (tensors, #params, % of model, policy and sensitive-component flags) without applying.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `operations` | `list` | required |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).
Raw value slices are capped at `limits.max_slice_elements` = 512 elements.

## Example

Call (MCP tool arguments, or `wsp call edit_preview '<json>'`):

```json
{
 "session_id": "sess_47aafe2f419d",
 "operations": [
  {
   "op": "scale",
   "tensor": "blocks.1.attn.o_proj.weight",
   "head": 0,
   "factor": 0.9
  }
 ]
}
```

Result (real output from the induction fixture):

```json
{
 "would_change": [
  {
   "name": "blocks.1.attn.o_proj.weight",
   "changed": 1024,
   "numel": 4096,
   "delta_norm": 0.4261,
   "rel_change": 0.05261,
   "max_abs_delta": 0.04244
  }
 ],
 "params_changed": 1024,
 "pct_of_model": 2.667,
 "policy_violations": [],
 "sensitive_flags": [],
 "operations": [
  {
   "op": "scale",
   "tensor": "blocks.1.attn.o_proj.weight",
   "head": 0,
   "factor": 0.9
  }
 ],
 "applied": false
}
```
