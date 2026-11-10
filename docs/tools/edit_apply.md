# `edit_apply`

Apply edit operations to the working copy as a new draft patch (with rationale). Regression suites run automatically.

**Who can call it:** agents (MCP) and humans.

**Access / budget:** no pass cost; runs regression suites automatically (system-run, not charged).

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `operations` | `list` | required |
| `rationale` | `str` | required |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).
Raw value slices are capped at `limits.max_slice_elements` = 512 elements.

## Example

Call (MCP tool arguments, or `wsp call edit_apply '<json>'`):

```json
{
 "session_id": "sess_04f94eb4cf68",
 "operations": [
  {
   "op": "scale",
   "tensor": "blocks.1.attn.o_proj.weight",
   "head": 0,
   "factor": 0.95
  }
 ],
 "rationale": "example: shrink head 1.0 output by 5%"
}
```

Result (real output from the induction fixture):

```json
{
 "patch_id": "p_f71b91ca6396e312",
 "status": "draft",
 "params_changed": 1024,
 "pct_of_model": 2.667,
 "tensors": [
  {
   "name": "blocks.1.attn.o_proj.weight",
   "changed": 1024,
   "rel_change": 0.0263
  }
 ],
 "model_hash_after": "1e772e28ed009f6b",
 "sensitive_flags": [],
 "stack": [
  "p_b91197ce4594d918",
  "p_bb44323a1eb5f4aa",
  "p_f71b91ca6396e312"
 ],
 "regression": {
  "patch_id": "p_f71b91ca6396e312",
  "status": "tested",
  "suites": [
   {
    "suite": "counting_regression",
    "split": "regression",
    "before": 1.0,
    "after": 1.0,
    "delta": 0.0
   },
   {
    "suite": "induction_regression",
    "split": "regression",
    "before": 0.9811,
    "after": 0.9856,
    "delta": 0.004557
   }
  ],
  "side_effects": []
 }
}
```
