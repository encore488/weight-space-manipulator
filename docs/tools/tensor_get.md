# `tensor_get`

Raw values of a small tensor slice (numpy-style 'a:b, c:d'; optional head). Size-capped.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `tensor` | `str` | required |
| `slice` | `str | None` | `null` |
| `head` | `int | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).
Raw value slices are capped at `limits.max_slice_elements` = 512 elements.

## Example

Call (MCP tool arguments, or `wsp call tensor_get '<json>'`):

```json
{
 "session_id": "sess_47aafe2f419d",
 "tensor": "blocks.1.attn.o_proj.weight",
 "slice": "0:2, 0:4"
}
```

Result (real output from the induction fixture):

```json
{
 "tensor": "blocks.1.attn.o_proj.weight",
 "region": "0:2, 0:4",
 "shape": [
  2,
  4
 ],
 "values": [
  [
   0.02882,
   -0.03742,
   0.08833,
   -0.1229
  ],
  [
   -0.0173,
   0.03086,
   -0.1372,
   -0.08463
  ]
 ]
}
```
