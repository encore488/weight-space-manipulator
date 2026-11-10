# `tensor_list`

List tensors with shapes, dtypes, layer grouping and head layout. Paged; regex filter.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `filter` | `str | None` | `null` |
| `offset` | `int` | `0` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call tensor_list '<json>'`):

```json
{
 "session_id": "sess_87aaa3b360e8",
 "filter": "blocks\\.1"
}
```

Result (real output from the induction fixture):

```json
{
 "total_tensors": 4,
 "offset": 0,
 "returned": 4,
 "next_offset": null,
 "params_by_group": {
  "layer 1": 16384
 },
 "tensors": [
  {
   "name": "blocks.1.attn.k_proj.weight",
   "shape": [
    64,
    64
   ],
   "dtype": "float32",
   "numel": 4096,
   "group": "layer 1",
   "heads": {
    "n_heads": 4,
    "d_head": 16,
    "axis": "rows"
   }
  },
  {
   "name": "blocks.1.attn.o_proj.weight",
   "shape": [
    64,
    64
   ],
   "dtype": "float32",
   "numel": 4096,
   "group": "layer 1",
   "heads": {
    "n_heads": 4,
    "d_head": 16,
    "axis": "cols"
   }
  },
  {
   "name": "blocks.1.attn.q_proj.weight",
   "shape": [
    64,
    64
   ],
   "dtype": "float32",
   "numel": 4096,
   "group": "layer 1",
   "heads": {
    "n_heads": 4,
    "d_head": 16,
    "axis": "rows"
   }
  },
  {
   "name": "blocks.1.attn.v_proj.weight",
   "shape": [
    64,
    64
   ],
   "dtype": "float32",
   "numel": 4096,
   "group": "layer 1",
   "heads": {
    "n_heads": 4,
    "d_head": 16,
    "axis": "rows"
   }
  }
 ]
}
```
