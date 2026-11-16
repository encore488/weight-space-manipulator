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
 "patch_a": "p_0000000000000001",
 "patch_b": "p_0000000000000002"
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "a": "p_0000000000000001",
 "b": "p_0000000000000002",
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
