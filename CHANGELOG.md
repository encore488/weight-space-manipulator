# Changelog

Notable changes to WSP Toolkit / WSP Bench, newest first. The weekly update on GitHub Releases and in the
newsletter summarises this file. Plans for upcoming versions are in [ROADMAP.md](ROADMAP.md).

## Unreleased

- Regenerating the tool reference keeps example IDs, timestamps, and challenge seeds stable.

## 0.3.0 (2026-11-15)

The first release built for PyPI (`pip install wsp-toolkit` once it's published).

### Added
- **WSP Bench**: nine auto-scored challenges with seeded instances (read, diagnose, repair and write tasks), plus
  `wsp bench run` to run all nine through the Anthropic API.
- **Public leaderboard** with verification: every submission is rebuilt from its seed, its edits are replayed and
  every run is re-scored. Submit with `wsp leaderboard export` and the issue form.
- **Website** on GitHub Pages: landing page, leaderboard, Arena replays and a submission guide.
- **Arena**: animated replays of challenge runs that follow a run live in `wsp ui`, export as standalone HTML, and
  record as MP4 (`wsp arena record`, `[video]` extra) or WebM (in the browser).
- `narrate` tool: agents can post running commentary that is shown as live captions. The API runner now logs the
  model's text and thinking summaries as captions too.
- PyPI packaging: project metadata, a release workflow using trusted publishing, and this changelog.
- `CONTRIBUTING.md`, `ROADMAP.md`, and issue templates for bugs, features and challenge ideas.

### Changed
- New "mission control" visual style across the website and the local UI.
- Tool calls counted in challenge usage statistics no longer include `narrate`.

## 0.2.0 (2026-11-10)

### Added
- The first challenges and the Anthropic API agent runner (`wsp agent run`): streaming, prompt caching, cost cap,
  transcripts.
- `wsp challenge prepare`, and an MCP server that can be locked to a single challenge (`wsp mcp --challenge`).
- Batched Hugging Face generation, Apple MPS and CUDA support, and a Kaggle notebook for the Qwen3.5-0.8B
  replication (first results: 5 MATH-500 problems).

### Changed
- Access-control hardening: below `outputs` access, agents get no evaluation scores.

## 0.1.0 (2026-11-08)

The first build of the roadmap's milestones M0–M7:

- inspect and probe tools exposed over MCP, with forward-pass budgets and access levels
- reversible, bit-exact patches
- an evaluation harness with held-out data
- safety guardrails and human approval for exports
- a zoo of 392 networks with known answers
- a feature index built from sparse autoencoders
- fine-tune audits
