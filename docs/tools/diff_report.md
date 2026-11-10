# `diff_report`

Full fine-tune audit (weights + features + behaviour), stored and viewable in the UI. Needs 'outputs'; costs 2 passes per probe.

**Who can call it:** agents (MCP) and humans.

**Access / budget:** needs `outputs`; 2 passes per probe (both models).

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `other_session_id` | `str` | required |
| `title` | `str` | `""` |
| `n_probes` | `int` | `32` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call diff_report '<json>'`):

```json
{
 "session_id": "sess_87aaa3b360e8",
 "other_session_id": "sess_5d9e37323c38",
 "n_probes": 8
}
```

Result (real output from the induction fixture):

```json
{
 "audit_id": "audit_8a23c654b2aa",
 "findings": [
  {
   "type": "behavior_change",
   "suite": "induction_heldout",
   "delta": -0.9385,
   "text": "drop of 0.939 on induction_heldout"
  },
  {
   "type": "behavior_change",
   "suite": "induction_practice",
   "delta": -0.9464,
   "text": "drop of 0.946 on induction_practice"
  },
  {
   "type": "behavior_change",
   "suite": "induction_regression",
   "delta": -0.956,
   "text": "drop of 0.956 on induction_regression"
  },
  {
   "type": "trigger_candidate",
   "where": "token in context",
   "tokens": [
    "t14",
    "t13",
    "t28"
   ],
   "text": "divergence concentrated where the token in context is t14, t13, t28 (lift 2.304)"
  },
  {
   "type": "prediction_shift",
   "tokens": [
    "t5",
    "t26",
    "t15"
   ],
   "text": "where the models disagree, the tuned model now predicts t5, t26, t15"
  },
  {
   "type": "vocab_direction",
   "tensor": "blocks.1.attn.o_proj.weight",
   "pole_a": [
    "t3",
    "t30",
    "t5"
   ],
   "pole_b": [
    "t17",
    "t19",
    "t21"
   ],
   "text": "top change direction in blocks.1.attn.o_proj.weight separates ['t3', 't30', 't5'] from ['t17', 't19', 't21'] (the sign cannot be read from weights alone)"
  },
  {
   "type": "vocab_direction",
   "tensor": "blocks.1.attn.o_proj.weight",
   "pole_a": [
    "t29",
    "t10",
    "t4"
   ],
   "pole_b": [
    "t11",
    "t12",
    "t28"
   ],
   "text": "top change direction in blocks.1.attn.o_proj.weight separates ['t29', 't10', 't4'] from ['t11', 't12', 't28'] (the sign cannot be read from weights alone)"
  },
  {
   "type": "weights",
   "tensor": "blocks.1.attn.o_proj.weight",
   "rel_change": 0.8952,
   "rank_90": 26,
   "text": "largest relative change in blocks.1.attn.o_proj.weight (0.8952); 90% of its delta energy 
  ... (truncated for the docs)
```
