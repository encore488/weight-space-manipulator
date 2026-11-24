# `run_baseline`

Run a non-edit baseline: the BASE model on a practice or held-out suite with different generation settings (e.g. temperature, repetition_penalty). Aggregate only for held-out.

**Who can call it:** agents (MCP) and humans.

**Access / budget:** practice: 1 pass per example; held-out: counts toward the held-out rate limit.

## Inputs

| Name | Type | Default |
|---|---|---|
| `session_id` | `str` | required |
| `suite_id` | `str` | required |
| `settings` | `dict` | required |
| `limit` | `int | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call run_baseline '<json>'`):

```json
{
 "session_id": "sess_000000000003",
 "suite_id": "induction_practice",
 "limit": 2,
 "settings": {
  "seed": 1
 }
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "run_id": "run_0000000000000008",
 "suite": "induction_practice",
 "split": "practice",
 "kind": "baseline",
 "patch_id": null,
 "model_hash": "610d318e08449718",
 "settings": {
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 0,
  "max_new_tokens": 64,
  "repetition_penalty": 1.0,
  "presence_penalty": 0.0,
  "thinking": false,
  "seed": 1,
  "samples": 1,
  "extra": {}
 },
 "metrics": {
  "n": 2,
  "score": 1.0,
  "stderr": 0.0,
  "token_accuracy": 1.0
 },
 "env": {
  "python": "3.12.12",
  "torch": "2.9.1+cpu",
  "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39",
  "wsp_toolkit": "0.3.0",
  "network": "blocked (thread)",
  "eval_batch_size": 1,
  "transformers": "5.23.0"
 },
 "cached": false,
 "details": [
  {
   "score": 1.0,
   "correct": 9,
   "total": 9
  },
  {
   "score": 1.0,
   "correct": 7,
   "total": 7
  }
 ],
 "model": "base"
}
```
