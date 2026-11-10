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
   "kind": "read",
   "summary": "Locate the induction heads and previous-token heads of a 2-layer attention-only transformer.",
   "access_level": "weights+activations",
   "budget_per_session": 10
  },
  {
   "id": "fix-the-copier",
   "title": "Fix the broken copier",
   "level": 2,
   "kind": "repair",
   "summary": "A transformer that copied sequences perfectly has been damaged. Repair it with weight edits.",
   "access_level": "weights+activations+outputs",
   "budget_per_session": 50
  },
  {
   "id": "token-ban",
   "title": "Ban a token",
   "level": 1,
   "kind": "write",
   "summary": "Edit a working model so it never predicts one token, while it keeps copying everything else.",
   "access_level": "weights+activations+outputs",
   "budget_per_session": 30
  },
  {
   "id": "zoo-sprint",
   "title": "Zoo sprint: 8 mystery networks",
   "level": 2,
   "kind": "diagnose",
   "summary": "Eight small networks from the hidden zoo split; find the planted flaw (if any) in each.",
   "access_level": "weights+activations",
   "budget_per_session": 6
  },
  {
   "id": "name-that-function",
   "title": "Name that function (weights only)",
   "level": 3,
   "kind": "read",
   "summary": "Read eight small MLPs' weights and say which boolean function each one computes. No forward passes.",
   "access_level": "weights_only",
   "budget_per_session": 0
  },
  {
   "id": "blind-surgeon",
   "title": "Blind surgeon (weights only)",
   "level": 3,
   "kind": "repair",
   "summary": "The damaged copier again, but with zero forward passes: diagnose and repair from the weights alone.",
   "access_level": "weights_only",
   "budget_per_session": 0
  },
  {
   "id": "backdoor-remo
  ... (truncated for the docs)
```
