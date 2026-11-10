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
 "patch_id": "p_8b4e424ecd90b3a7"
}
```

Result (real output from the induction fixture):

```json
{
 "patch_id": "p_8b4e424ecd90b3a7",
 "workspace_id": "ws_f5367d180411",
 "base_model_hash": "1b441d92f299953d64ee9a233753978c72dfc1a7e848b0baabfba39741ba072e",
 "parent_patch": null,
 "model_hash_before": "1b441d92f299953d64ee9a233753978c72dfc1a7e848b0baabfba39741ba072e",
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
   "hash_before": "cf1798b3c6ed74297f2078be991a1b9ad8ab0f7a71e2000074999b203546e785",
   "hash_after": "0fb63f3bd6516a6e7420722769b2e92009e5cc48546cbe88d27da6bf180f237c"
  }
 ],
 "params_changed": 1024,
 "pct_of_model": 2.66667,
 "rationale": "doc patch A",
 "author": "agent:sess_87aaa3b360e8",
 "created_at": 1794315523.5216708,
 "evals": [
  "run_ea10fc137eed9140",
  "run_462a0f74236e0f0b",
  "run_feb94c01de7cdb8d",
  "run_18118843b5084e85"
 ],
 "model_hash_after": "ec8aee16b0d759b7fc66d7492914b77098aa99b1143ba1cd3ce3e4874b734707",
 "status": "tested",
 "flags": [],
 "approvals": [],
 "eval_runs": [
  {
   "run_id": "run_18118843b5084e85",
   "suite": "induction_regression",
   "on": "patched",
   "metrics": "hidden at this access level"
  },
  {
   "run_id": "run_462a0f74236e0f0b",
   "suite": "counting_regression",
   "on": "patched",
   "metrics": "hidden at this access level"
  },
  {
   "run_id": "run_ea10fc137eed9140",
   "suite": "counting_regression",
   "on": "base",
   "metrics": "hidden at this access level"
  },
  {
   "run_id": "run_feb94c01de7cdb8d",
   "suite": "induction_regression",
   "on": "base",
   "metrics": "hidden at this access level"
  }
 ]
}
```
