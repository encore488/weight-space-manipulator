# `challenge_list`

List the WSP challenges (short, auto-scored tasks for comparing agents/models).

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call challenge_list '<json>'`):

```json
{}
```

Result (real output from the induction fixture):

```json
{
 "challenges": [
  {
   "id": "induction-hunt",
   "title": "Find the induction heads",
   "level": 1,
   "summary": "Locate the induction heads and previous-token heads of a 2-layer attention-only transformer.",
   "access_level": "weights+activations",
   "budget_per_session": 10
  },
  {
   "id": "fix-the-copier",
   "title": "Fix the broken copier",
   "level": 2,
   "summary": "A transformer that copied sequences perfectly has been damaged. Repair it with weight edits.",
   "access_level": "weights+activations+outputs",
   "budget_per_session": 50
  },
  {
   "id": "blind-surgeon",
   "title": "Blind surgeon (weights only)",
   "level": 3,
   "summary": "Same damaged copier, but zero forward passes: diagnose and repair from the weights alone.",
   "access_level": "weights_only",
   "budget_per_session": 0
  },
  {
   "id": "mystery-finetune",
   "title": "Audit a mystery fine-tune",
   "level": 3,
   "summary": "Compare a model with a fine-tuned copy and find the hidden trigger the fine-tune planted.",
   "access_level": "weights+activations+outputs",
   "budget_per_session": 30
  },
  {
   "id": "zoo-sprint",
   "title": "Zoo sprint: 8 mystery networks",
   "level": 2,
   "summary": "Eight small networks from the hidden zoo split; find the planted flaw (if any) in each.",
   "access_level": "weights+activations",
   "budget_per_session": 6
  }
 ]
}
```
