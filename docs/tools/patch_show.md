# `patch_show`

Full manifest of a patch: operations, per-tensor hashes, params changed, rationale, linked evaluations, status, approvals.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `patch_id` | `str` | required |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call patch_show '<json>'`):

```json
{
 "patch_id": "p_0000000000000001"
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "patch_id": "p_0000000000000001",
 "workspace_id": "ws_000000000002",
 "base_model_hash": "610d318e08449718477580a7d3db31b8b5f8d93d77605b0fb6e6fc355bc5e75b",
 "parent_patch": null,
 "model_hash_before": "610d318e08449718477580a7d3db31b8b5f8d93d77605b0fb6e6fc355bc5e75b",
 "operations": [
  {
   "op": "scale",
   "tensor": "blocks.1.attn.o_proj.weight",
   "head": 1,
   "factor": 0.9
  }
 ],
 "tensors": [
  {
   "name": "blocks.1.attn.o_proj.weight",
   "changed": 1024,
   "numel": 4096,
   "delta_norm": 0.3993,
   "rel_change": 0.04664,
   "max_abs_delta": 0.03495,
   "hash_before": "726a42b19b8ef2cf95b2c9c98198a4643818c752d43293d100da4dcd5e0d1529",
   "hash_after": "a598236735a06432056bf5d2c072c572b6941509dfcacbba155a2ab8d87f1cdb"
  }
 ],
 "params_changed": 1024,
 "pct_of_model": 2.66667,
 "rationale": "doc patch A",
 "author": "agent:sess_000000000003",
 "created_at": 0.0,
 "evals": [
  "run_0000000000000002",
  "run_0000000000000003",
  "run_0000000000000004",
  "run_0000000000000005"
 ],
 "model_hash_after": "61c0c7aa6ceb6c1376efdd99432a5fddf2c46f025f06d658b3e0f8a94a2c3e0a",
 "status": "tested",
 "flags": [],
 "approvals": [],
 "eval_runs": [
  {
   "run_id": "run_0000000000000003",
   "suite": "counting_regression",
   "on": "patched",
   "metrics": "hidden at this access level"
  },
  {
   "run_id": "run_0000000000000005",
   "suite": "induction_regression",
   "on": "patched",
   "metrics": "hidden at this access level"
  },
  {
   "run_id": "run_0000000000000002",
   "suite": "counting_regression",
   "on": "base",
   "metrics": "hidden at this access level"
  },
  {
   "run_id": "run_0000000000000004",
   "suite": "induction_regression",
   "on": "base",
   "metrics": "hidden at this access level"
  }
 ]
}
```
