# `direction_readout`

Project a direction onto the vocabulary: which tokens it promotes / suppresses.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `direction` | `dict` | required |
| `k` | `int` | `10` |
| `final_norm` | `bool` | `true` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call direction_readout '<json>'`):

```json
{
 "session_id": "sess_04f94eb4cf68",
 "direction": {
  "tensor": "blocks.1.attn.o_proj.weight",
  "svd": 0,
  "side": "left"
 },
 "k": 5
}
```

Result (real output from the induction fixture):

```json
{
 "direction": "left singular vector 0 of blocks.1.attn.o_proj.weight",
 "through_final_norm_scale": false,
 "logit_mean": 0.01227,
 "logit_std": 0.154,
 "promotes": [
  {
   "token": "t3",
   "id": 3,
   "logit": 0.3257
  },
  {
   "token": "t30",
   "id": 30,
   "logit": 0.2692
  },
  {
   "token": "t5",
   "id": 5,
   "logit": 0.2571
  },
  {
   "token": "t1",
   "id": 1,
   "logit": 0.2309
  },
  {
   "token": "t6",
   "id": 6,
   "logit": 0.1923
  }
 ],
 "suppresses": [
  {
   "token": "t17",
   "id": 17,
   "logit": -0.3237
  },
  {
   "token": "t19",
   "id": 19,
   "logit": -0.2232
  },
  {
   "token": "t21",
   "id": 21,
   "logit": -0.1889
  },
  {
   "token": "t15",
   "id": 15,
   "logit": -0.1785
  },
  {
   "token": "t24",
   "id": 24,
   "logit": -0.1711
  }
 ]
}
```
