# `zoo_list_tasks`

List zoo networks and their task types (identify / localize / repair).

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `family` | `str | None` | `null` |
| `split` | `str | None` | `null` |
| `task_type` | `str | None` | `null` |
| `offset` | `int` | `0` |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call zoo_list_tasks '<json>'`):

```json
{
 "family": "mlp_compiled"
}
```

Result (real output from the induction fixture):

```json
{
 "total": 4,
 "offset": 0,
 "families": {
  "mlp_bool": 16,
  "mlp_compiled": 4,
  "rl_control": 6
 },
 "items": [
  {
   "net_id": "z4759a4880a",
   "family": "mlp_compiled",
   "split": "public",
   "task_types": [
    "identify",
    "localize"
   ]
  },
  {
   "net_id": "z9a28ecd4ca",
   "family": "mlp_compiled",
   "split": "public",
   "task_types": [
    "identify",
    "localize"
   ]
  },
  {
   "net_id": "z9c39baf061",
   "family": "mlp_compiled",
   "split": "public",
   "task_types": [
    "identify",
    "localize"
   ]
  },
  {
   "net_id": "ze31ec1fc4c",
   "family": "mlp_compiled",
   "split": "public",
   "task_types": [
    "identify",
    "localize"
   ]
  }
 ],
 "next_offset": null
}
```
