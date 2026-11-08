# `suite_examples`

Page through the examples of a PRACTICE suite (held-out / regression examples are never returned).

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `suite_id` | `str` | required |
| `offset` | `int` | `0` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call suite_examples '<json>'`):

```json
{
 "session_id": "sess_47aafe2f419d",
 "suite_id": "induction_practice"
}
```

Result (real output from the induction fixture):

```json
{
 "suite": "induction_practice",
 "total": 32,
 "offset": 0,
 "examples": [
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
  },
  {
   "input": [
    0,
    5,
    21,
    6,
    9,
    11,
    23,
    19,
    8,
    5,
    21,
    6,
    9,
    11,
    23,
    19,
    8,
    6,
    25,
    15,
    25,
    26,
    31,
    28
   ],
   "score_from": 9,
   "score_to": 16
  },
  {
   "input": [
    0,
    23,
    12,
    1,
    4,
    6,
    23,
    8,
    23,
    12,
    1,
    4,
    6,
    23,
    8,
    27,
    9,
    9,
    24,
    14,
    2,
    11,
    4,
    16
   ],
   "score_from": 8,
   "score_to": 14
  },
  {
   "input": [
    0,
    24,
    8,
    12,
    27,
    18,
    3,
    29,
    30,
    25,
    24,
    8,
    12,
    27,
    18,
    3,
    29,
    30,
    25,
    17,
    11,
    18,
    13,
    23
   ],
   "score_from": 10,
   "score_to": 18
  },
  {
   "input": [
    0,
    16,
    1,
    24,
    16,
    14,
    14,
    16,
    1,
    24,
    16,
    14,
    14,
    12,
    9,
    13,
    20,
    2,
    21,
    26,
    25,
    12,
    23,
    29
   ],
   "score_from": 7,
   "score_to": 12
  },
  {
   "input": [
    0,
    25,
    9,
    20,
    7,
    29,
    25,
    9,
    20,
    7,
    29,
    4,
    16,
    1,
    21,
    19,
    7,
    8,
    24,
    7,
    3,
    30,
    24,
    15
   ],
   "score_from":
  ... (truncated for the docs)
```
