# `workspace_info`

Describe the session's model, working-copy state (applied patches, hash) and budget.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call workspace_info '<json>'`):

```json
{
 "session_id": "sess_87aaa3b360e8"
}
```

Result (real output from the induction fixture):

```json
{
 "workspace_id": "ws_f5367d180411",
 "source": "fixture:induction_tf",
 "access_level": "weights+activations+outputs",
 "capabilities": [
  "activations",
  "outputs",
  "weights"
 ],
 "budget": {
  "used": 4,
  "total": 200,
  "remaining": 196
 },
 "applied_patches": [
  "p_8b4e424ecd90b3a7",
  "p_de781df671b58006"
 ],
 "working_copy_hash": "ed5c84a209e1deb3",
 "is_base": false,
 "model_kind": "tiny_transformer",
 "base_model_hash": "1b441d92f299953d",
 "n_params": 38400,
 "n_tensors": 11,
 "n_layers": 2,
 "is_language_model": true,
 "hook_points": [
  "L0.z",
  "L0.resid_pre",
  "L0.attn_out",
  "L0.mlp_out",
  "L0.resid_post",
  "L1.z",
  "L1.resid_pre",
  "L1.attn_out",
  "L1.mlp_out",
  "L1.resid_post"
 ],
 "vocab_size": 32,
 "token_names_sample": [
  "<bos>",
  "t1",
  "t2",
  "t3",
  "t4",
  "t5",
  "t6",
  "t7"
 ],
 "config": {
  "wsp_arch": "tiny_transformer",
  "vocab": 32,
  "max_len": 24,
  "d_model": 64,
  "n_heads": 4,
  "n_layers": 2,
  "d_mlp": 0,
  "layernorm": false,
  "fixture": "induction_tf",
  "seed": 0,
  "input_kind": "tokens"
 }
}
```
