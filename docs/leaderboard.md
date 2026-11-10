# Public leaderboard

Website: **https://encore488.github.io/weight-space-manipulator/** (built from this repo by
`.github/workflows/site.yml`).

## Submitting

```bash
wsp bench run --model claude-fable-5-2 --max-cost 5            # or individual `wsp agent run` / MCP runs
wsp leaderboard export --model claude-fable-5-2 --by your-handle --notes "effort high"
wsp leaderboard verify wsp-submission-claude-fable-5-2.json    # optional: the same check the bot runs
```

Then either:
- **Issue form** (easiest): open
  [a leaderboard submission](https://github.com/encore488/weight-space-manipulator/issues/new?template=leaderboard-submission.yml)
  and paste the file. The `leaderboard-intake` bot re-scores it and comments the result. A maintainer comments
  `/accept` to add it; the bot commits the file and rebuilds the site.
- **Pull request** (for files over ~60,000 characters): add the file under `leaderboard/submissions/`. The
  `leaderboard-pr` check re-scores it using the main branch's code, and fails if the PR touches anything else.

## What a submission contains

`schema: wsp-leaderboard/v1`, the toolkit version, `agent_model`, `submitted_by`, `notes`, and one entry per run:
- **Run identity:** challenge and seed.
- **What the agent did:** the answer it submitted and every weight edit it applied (directions inlined as vectors).
- **Context:** the claimed score, harness and lock flag, and self-reported usage (forward passes, tool calls,
  turns, estimated cost).
- **Transcript fingerprint:** the transcript's SHA-256 (the transcript itself stays with you).

## Verification

Every site build, and every intake and pull-request check:
1. rebuilds each run's instance from its seed;
2. replays the submitted edits on a fresh copy;
3. re-runs the scorer with the submitted answer;
4. requires the result to match the claimed score within ±0.05.

The tolerance exists because instances are small trained models whose weights differ slightly across platforms
and PyTorch versions.

This proves the score follows from the submitted answer and edits. It does **not** prove which model produced them,
or that the budget and access level were respected. Those come from the submitter's own toolkit log. The
leaderboard is honest self-reporting with reproducible scores. Please report your setup in `--notes`.

## Scores

Each challenge's score is the mean over that entry's runs (more seeds means less noise). **WSP-Bench** is the mean
over the nine challenges, with missing challenges counted as 0; entries without all nine are marked *partial* and
ranked after complete ones.
