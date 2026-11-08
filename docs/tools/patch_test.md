# `patch_test`

Run the required suites (regression etc.) on parent vs patched; marks the patch 'tested'.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `patch_id` | `str` | required |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call patch_test '<json>'`):

```json
{
 "patch_id": "p_b4c4d027c65e9bec"
}
```

Result (real output from the induction fixture):

```json
{
 "patch_id": "p_b4c4d027c65e9bec",
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
   "after": 0.9811,
   "delta": 0.0
  }
 ],
 "side_effects": []
}
```
