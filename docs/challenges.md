# Challenges

Each challenge fixes the model(s), the access level and the forward-pass budget, and scores the result
automatically, so different agents and models can be compared on equal terms.

```bash
wsp challenge list
wsp challenge prepare              # build the small models once (~1-2 min)
wsp challenge baseline             # the scripted reference agent
wsp challenge scores               # leaderboard (also in the UI: Challenges)
wsp challenge report               # shareable Markdown table
```

| ID | Task | Access | Budget | Scoring |
|---|---|---|---|---|
| `induction-hunt` | Name the induction heads and previous-token heads of a 2-layer attention-only transformer | weights + activations | 10 | 0.5 × Jaccard(induction) + 0.5 × Jaccard(previous-token) |
| `fix-the-copier` | A transformer that copied repeated segments has been damaged; repair it with edits | weights + activations + outputs | 50 | copy accuracy of the final working copy on hidden data (× counting accuracy if counting fell below 0.95) |
| `blind-surgeon` | The same repair, from the weights alone | weights only | 0 | as above |
| `mystery-finetune` | Find the trigger token a fine-tune planted, and the token it forces | weights + activations + outputs | 30 per model | 0.5 trigger + 0.5 forced token |
| `zoo-sprint` | Find the planted flaw (or none) in 8 hidden-split zoo networks | weights + activations | 6 per network | mean of the zoo localize score (flaw type 0.5 + location 0.5) |

Reference points: the healthy copier scores about 0.98 and the damaged one about 0.04. The scripted baseline scores
1.00 / 0.98 / 0.83 / 0.00 / 0.50 on the five challenges (in table order: induction-hunt, fix-the-copier,
blind-surgeon, mystery-finetune, zoo-sprint).

## Fair play

- **Harness.** `wsp agent run` gives the model only the toolkit's tools; those runs are comparable. Runs started from
  an MCP client are labelled `mcp (self-reported)`: if that client has shell or file access, the agent could read
  answer files on disk.
- **Locked servers.** `wsp mcp --challenge <id>` lets agents open only that challenge's models (so a "fix it" agent
  can't open the healthy model and copy its weights). Runs from other servers are labelled *unlocked*.
- **Weights only really is weights only.** Below the `outputs` access level the agent gets no evaluation scores and no
  regression results, only weights (and activations if allowed).
- **One submission per run.** Start a new run to try again; every run is kept.
- **The answers are fixed.** Please don't publish them; share scores and approaches instead.

## Adding a challenge

Challenges are defined in `wsp_toolkit/core/challenges.py`: models, access level, budget, brief, answer format and a
`_score_<id>` method. Contributions welcome.
