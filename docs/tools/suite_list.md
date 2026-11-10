# `suite_list`

List test suites for this model: practice (examples visible), held-out and regression (names and sizes only).

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call suite_list '<json>'`):

```json
{
 "session_id": "sess_04f94eb4cf68"
}
```

Result (real output from the induction fixture):

```json
{
 "suites": [
  {
   "id": "induction_practice",
   "name": "Induction copy (practice)",
   "split": "practice",
   "kind": "behavior",
   "scorer": "next_token_acc",
   "n_examples": 32,
   "generation": {},
   "example_preview": [
    {
     "input": [
      0,
      15,
      8,
      7,
      29,
      1,
      30,
      30,
      31,
      19,
      9,
      15,
      8,
      7,
      29,
      1,
      30,
      30,
      31,
      19,
      9,
      23,
      5,
      16
     ],
     "score_from": 11,
     "score_to": 20
    },
    {
     "input": [
      0,
      12,
      24,
      27,
      17,
      31,
      19,
      9,
      18,
      12,
      24,
      27,
      17,
      31,
      19,
      9,
      18,
      20,
      22,
      25,
      5,
      11,
      21,
      22
     ],
     "score_from": 9,
     "score_to": 16
    }
   ]
  },
  {
   "id": "induction_heldout",
   "name": "Induction copy (held-out)",
   "split": "heldout",
   "kind": "behavior",
   "scorer": "next_token_acc",
   "n_examples": 128,
   "generation": {}
  },
  {
   "id": "counting_regression",
   "name": "Counting (side-effect check)",
   "split": "regression",
   "kind": "regression",
   "scorer": "next_token_acc",
   "n_examples": 64,
   "generation": {}
  },
  {
   "id": "induction_regression",
   "name": "Induction copy (regression)",
   "split": "regression",
   "kind": "regression",
   "scorer": "next_token_acc",
   "n_examples": 64,
   "generation": {}
  }
 ]
}
```
