# `workspace_open`

Open a model (fixture:<name>, zoo:<id>, hf:<repo>, or a local path) into a new workspace with a working copy, and start a budgeted session on it.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `source` | `str` | required |
| `label` | `str | None` | `null` |
| `access_level` | `str | None` | `null` |
| `budget` | `int | None` | `null` |
| `confirm_download` | `bool` | `false` |
| `agent_model` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call workspace_open '<json>'`):

```json
{
 "source": "fixture:induction_tf",
 "access_level": "weights+activations+outputs"
}
```

Result (real output from the induction fixture):

```json
{
 "workspace_id": "ws_c274eccdf9c1",
 "session_id": "sess_36ce9b0400a4",
 "access_level": "weights+activations+outputs",
 "capabilities": [
  "activations",
  "outputs",
  "weights"
 ],
 "budget": {
  "used": 0,
  "total": 200,
  "remaining": 200
 },
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
