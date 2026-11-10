# WSP Toolkit

**Let an AI agent read, debug and edit neural-network weights directly, and watch it do it.**

Weight-space programming (WSP) means changing what a network does by reasoning about its weights, without gradient
descent. MIT CSAIL's [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/)
showed a frontier-model agent doing this in a sandbox: it found and removed the looping behaviour in a 0.8B model
by editing 0.016% of its weights. This toolkit turns that sandbox into something you can run on a laptop:

- **An MCP server** (`wsp mcp`) with ~50 tools for Claude Code or any MCP client: inspect tensors, SVDs and vocabulary
  readouts; probe activations and attention under a **forward-pass budget**; make **reversible, bit-exact patches**;
  evaluate on practice data while **held-out data stays hidden**.
- **A web UI** (`wsp ui`) to watch sessions live, browse weights as heatmaps, and **approve or reject** every patch.
  Nothing can be exported without a human approval.
- **Five auto-scored challenges** and an **API runner** (`wsp agent run`) for comparing models, e.g. Claude Fable 5.2,
  on the same tasks with the same budgets.
- Plus a 392-network **zoo** with planted flaws, a **feature index** (sparse autoencoders) and **fine-tune audits**.

Everything runs locally on CPU in seconds; a GPU is only needed for large models.

![The web UI](docs/img/home.png)

## Try it in five minutes

**1. Install** (Python ≥ 3.11; [uv](https://docs.astral.sh/uv/) picks a suitable Python for you):

```bash
uv tool install "wsp-toolkit[agent] @ git+https://github.com/encore488/weight-space-manipulator"
```

or, with pip in a virtual environment:

```bash
pip install "wsp-toolkit[agent] @ git+https://github.com/encore488/weight-space-manipulator"
```

On Linux without a GPU, install the CPU build of PyTorch first to avoid a multi-GB download:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

**2. Build the small models once** (~1–2 minutes on CPU):

```bash
wsp challenge prepare
```

**3. Pick one of two ways to run a challenge.**

*With Claude Code* (or any MCP client). The server locks itself to the challenge's models:

```bash
claude mcp add wsp -- wsp mcp --challenge fix-the-copier
```
```bash
wsp challenge prompt fix-the-copier --model claude-fable-5-2
```

Paste the printed prompt into Claude Code, then watch it work:

```bash
wsp ui          # http://127.0.0.1:8765
```

*Through the Anthropic API* (the model gets nothing but the toolkit's tools, which makes these the comparable numbers):

```bash
export ANTHROPIC_API_KEY=...        # or: ant auth login
wsp agent run fix-the-copier --model claude-fable-5-2
```

The runner streams, caches prompts, stops at `--max-cost` (default $20, estimated) and saves a transcript. Use any
Claude model ID with `--model`.

## Challenges

![Challenges](docs/img/challenges.png)

| Challenge | Level | Access | Budget | Scripted baseline |
|---|---|---|---|---|
| `induction-hunt`: find the induction & previous-token heads | 1 | weights + activations | 10 passes | 1.00 |
| `fix-the-copier`: repair a damaged transformer without breaking counting | 2 | weights + activations + outputs | 50 | 0.98 |
| `zoo-sprint`: find the planted flaw in 8 hidden-split networks | 2 | weights + activations | 6 per net | 0.50 |
| `blind-surgeon`: same repair with **zero** forward passes | 3 | weights only | 0 | 0.83 |
| `mystery-finetune`: find the backdoor a fine-tune planted | 3 | weights + activations + outputs | 30 per model | 0.00 |

Scores run 0–1. The scripted baseline is deterministic code using the same tools (`wsp challenge baseline`).
`wsp challenge scores` and the **Challenges** page show the leaderboard by model; `wsp challenge report` prints a
shareable Markdown table. Details and fair-play rules: [`docs/challenges.md`](docs/challenges.md).

## Using it on your own models

```bash
wsp mcp --budget 50 --access weights+activations+outputs
```

Then ask your agent to `workspace_open` a local safetensors model, a Hugging Face ID (`hf:Qwen/Qwen3.5-0.8B`, with the
`hf` extra) or a built-in fixture (`fixture:induction_tf`). The human sets the access level and budget ceiling; agents
can only ask for less. Every tool call is logged append-only and visible in the UI.

## How it keeps you in control

- **Originals are never modified.** Edits go to an in-memory working copy rebuilt from a hash-verified base plus a
  stack of patches. Revert is bit-exact.
- **Budgets and access levels are enforced in code:** `weights_only`, `weights+activations`,
  `weights+activations+outputs`. Below `outputs`, agents get no behavioural signal at all (no evaluation scores either).
- **Held-out data never reaches the agent**: aggregate scores only, rate-limited, run with network access blocked.
- **Human approval gates export.** Patches touching registered sensitive components need explicit acknowledgement.
- **Hard rule:** the toolkit refuses to open the calling agent's own weights.

## Status, honestly

This is v0.2 of a personal research project. Expect rough edges:

- The challenges and zoo use **small networks** (up to ~40K parameters), so you can see everything on CPU. The big-model
  path (Hugging Face, e.g. Qwen3.5-0.8B) is implemented and tested on the real architecture at tiny size. A first,
  small-subset comparison of the paper's Qwen3.5-0.8B edit against the base model is running on a Kaggle GPU; results
  will land in `docs/reports/`.
- Feature labels come from a simple rule-based labeler with measured fidelity, not an LLM.
- Runs from MCP clients with shell access are *self-reported*: such an agent could read answer files on disk. Use
  `wsp agent run` for comparable numbers.
- Design decisions and their rationale: [`DECISIONS.md`](DECISIONS.md). Acceptance results:
  [`docs/reports/milestones.md`](docs/reports/milestones.md).

## Docs

- [`docs/agents.md`](docs/agents.md): connecting Claude Code, Claude Desktop, other MCP clients, the API runner
- [`docs/challenges.md`](docs/challenges.md): the challenges, scoring and fair play
- [`docs/tools/`](docs/tools/README.md): every tool, with a real example call and output
- [`docs/configuration.md`](docs/configuration.md), [`docs/design/patch_semantics.md`](docs/design/patch_semantics.md)

## Development

```bash
git clone https://github.com/encore488/weight-space-manipulator && cd weight-space-manipulator
pip install -e '.[dev,hf,pdf,agent]'
pytest            # 76 tests, CPU only, no network, ~2.5 min
```

Apache-2.0. Built with Claude Code.
