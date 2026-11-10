# `patch_diff`

Compare two patches: tensors touched by each, overlapping elements, delta cosine.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `patch_a` | `str` | required |
| `patch_b` | `str` | required |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call patch_diff '<json>'`):

```json
{
 "patch_a": "p_8b4e424ecd90b3a7",
 "patch_b": "p_de781df671b58006"
}
```

Result (real output from the induction fixture):

```json
{
 "a": "p_8b4e424ecd90b3a7",
 "b": "p_de781df671b58006",
 "only_in_a": [],
 "only_in_b": [],
 "shared": [
  {
   "tensor": "blocks.1.attn.o_proj.weight",
   "elements_a": 1024,
   "elements_b": 1024,
   "overlapping_elements": 1024,
   "delta_cosine": 1.0
  }
 ],
 "same_base": true,
 "ops_a": [
  "scale"
 ],
 "ops_b": [
  "scale"
 ]
}
```
