# `zoo_get_task`

Task card for a zoo network: instructions, answer format, candidate labels, and the source string to open it with workspace_open.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `net_id` | `str` | required |
| `task_type` | `str` | required |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call zoo_get_task '<json>'`):

```json
{
 "net_id": "z4759a4880a",
 "task_type": "identify"
}
```

Result (real output from the induction fixture):

```json
{
 "net_id": "z4759a4880a",
 "task_type": "identify",
 "instructions": "Work out what the network computes. Answer {\"label\": <one key of candidate_labels>}.",
 "open_with": {
  "tool": "workspace_open",
  "source": "zoo:z4759a4880a"
 },
 "card": {
  "family": "mlp_compiled",
  "architecture": "MLP 8-1-2 (ReLU), hand-compiled",
  "input_description": "8 binary inputs (0/1); outputs 2 logits",
  "candidate_labels": {
   "and_01": "x0 AND x1",
   "or_01": "x0 OR x1",
   "xor_01": "x0 XOR x1",
   "parity_012": "x0 XOR x1 XOR x2",
   "parity_0123": "parity of x0..x3",
   "majority_012": "majority of x0,x1,x2",
   "at_least_3_of_4": "x0+x1+x2+x3 >= 3",
   "copy_2": "x2",
   "implies_0_1": "x0 -> x1",
   "and_0_not_3": "x0 AND NOT x3"
  },
  "flaw_types": [
   "none",
   "silenced_input",
   "dead_unit",
   "shifted_threshold",
   "backdoor"
  ],
  "task_types": [
   "identify",
   "localize"
  ],
  "split": "public",
  "id": "z4759a4880a"
 }
}
```
