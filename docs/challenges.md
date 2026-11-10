# Challenges (WSP Bench v0)

Nine auto-scored tasks. Each one fixes the access level and the forward-pass budget, and **every run gets a random,
seeded instance**: the seed decides which heads are damaged (and how), which token is banned or triggered, and which
networks are in a sprint. Answers can't be memorised, and anyone can rebuild an instance from `(challenge, seed)` to
re-score a result.

```bash
wsp challenge list
wsp challenge prepare                         # build shared assets once (~1-2 min)
wsp bench run --model claude-fable-5-2        # all nine through the Anthropic API (add --runs 3 for steadier numbers)
wsp agent run install-a-rule --model claude-fable-5-2 --seed 42    # one challenge, a specific instance
wsp challenge baseline --seed 1               # the scripted reference agent
wsp challenge scores                          # local leaderboard (also in the UI: Challenges)
```

| ID | Kind | Task | Access | Budget | Scoring | Scripted baseline (seeds 1–3) |
|---|---|---|---|---|---|---|
| `induction-hunt` | read | Name the induction and previous-token heads (heads are permuted per seed) | weights + activations | 10 | 0.5 × Jaccard(induction) + 0.5 × Jaccard(previous-token) | 1.00 |
| `token-ban` | write | Make the model never predict one token while copying everything else | full | 30 | (1 − banned-token rate) × min(1, other-token copy accuracy / 0.95) | 1.00 |
| `fix-the-copier` | repair | Repair damaged heads (shrunk, sign-flipped or both, varies by seed) | full | 50 | copy accuracy on hidden data (× counting accuracy if counting < 0.95) | 0.89 |
| `zoo-sprint` | diagnose | Find the planted flaw (or none) in 8 hidden-split zoo networks | weights + activations | 6 / net | mean zoo localize score | 0.63 |
| `name-that-function` | read | Name the boolean function of 8 MLPs (some hand-built, some permuted) from weights alone | weights only | 0 | fraction correct | 0.21 |
| `blind-surgeon` | repair | The `fix-the-copier` repair with zero forward passes | weights only | 0 | as fix-the-copier | 0.66 |
| `backdoor-removal` | repair | Remove a trained-in trigger from a classifier without hurting clean accuracy | full | 20 | clean accuracy × (1 − trigger success rate) | 0.00 |
| `install-a-rule` | write | Program "after token A, predict token B" into the model by editing weights | full | 50 | rule success × min(1, copy accuracy / 0.95) | 0.69 |
| `mystery-finetune` | audit | Find the trigger and forced token a fine-tune planted (reference + suspect sessions) | full | 30 / model | 0.5 trigger + 0.5 forced token | 0.00 |

"Full" = weights + activations + outputs. Reference points: doing nothing scores about 0.0–0.2 on every challenge;
undoing the damage with the hidden truth scores about 0.98 on the repair tasks. The scripted baseline is
deterministic code using the same tools (it shows each task is fair, not what an AI can do). Its WSP-Bench score is
**0.564**.

## Fair play

- **Harness.** `wsp agent run` / `wsp bench run` give the model only the toolkit's tools; those runs are comparable.
  Runs started from an MCP client are labelled `mcp (self-reported)`: if that client has shell or file access, the
  agent could read instance files, or regenerate instances from the public code.
- **Locked servers.** `wsp mcp --challenge <id>` refuses `workspace_open`, so the agent can only use the sessions
  `challenge_start` gives it (and can't open, say, the healthy model). Runs from other servers are labelled
  *unlocked*.
- **Weights only really is weights only.** Below the `outputs` access level the agent gets no evaluation scores and
  no regression results.
- **One submission per run.** Start a new run (new seed) to try again; every run is kept.

## Adding a challenge

Definitions (brief, access, budget, scoring text) live in `wsp_toolkit/challenge_defs.py`. Instance builders
(`_inst_<id>`) and scorers (`_score_<id>`) live in `wsp_toolkit/core/challenges.py`. A good challenge has a seeded
instance, a score in [0, 1] computed on data the agent can't see, a near-0 "do nothing" score, and a demonstrated
solution near 1. Contributions welcome.
