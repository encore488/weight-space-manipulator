# Connecting agents

The toolkit speaks MCP over stdio (`wsp mcp`). The human running the server sets the ceilings; an agent can ask for
less but never more:

```bash
wsp mcp --budget 50 --access weights+activations+outputs     # ceilings for every session the agent opens
wsp mcp --challenge fix-the-copier                            # lock to one challenge's models
```

Open the UI in another terminal to watch: `wsp ui` → http://127.0.0.1:8765.

## Claude Code

```bash
claude mcp add wsp -- wsp mcp --budget 50
```

To pick a model, start Claude Code with `claude --model <model-id>` (or use `/model` inside it). To switch to a
challenge-locked server, replace the entry:

```bash
claude mcp remove wsp
```
```bash
claude mcp add wsp -- wsp mcp --challenge mystery-finetune
```

`wsp challenge prompt <id> --model <model-id>` prints a ready-to-paste prompt. Claude Code also has shell access, so
challenge runs from it are marked *self-reported* (it could read answer files on disk). The prompt asks it not to.

## Claude Desktop (and other MCP apps that take a JSON config)

Add to `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`). Desktop apps often don't see
your shell's `PATH`, so use the full path from `which wsp`:

```json
{
  "mcpServers": {
    "wsp": {"command": "/Users/you/.local/bin/wsp", "args": ["mcp", "--budget", "50"]}
  }
}
```

## Anthropic API runner (for comparable benchmark numbers)

```bash
pip install anthropic          # included in the [agent] extra
export ANTHROPIC_API_KEY=...   # or: ant auth login
wsp agent run zoo-sprint --model claude-fable-5-2 --runs 3
```

| Option | Default | Meaning |
|---|---|---|
| `--model` | `claude-fable-5-2` | any Claude model ID |
| `--effort` | `high` | `low`…`max`, or `none` to omit (for models without effort control) |
| `--thinking` | `summarized` | `summarized` returns readable reasoning summaries in the transcript; `omit` sends no thinking setting |
| `--max-turns` | 80 | stop after this many model turns |
| `--max-cost` | 20 | stop when the **estimated** cost (USD) exceeds this |
| `--runs` | 1 | repeat the challenge N times |

The model receives only the toolkit's tools (no shell, no files, no `workspace_open`). Details:
- **Requests:** streaming, an append-only history, automatic prompt caching, and server-side refusal fallback
  (`fallbacks: "default"`) on models that support it.
- **Limits:** if the turn or cost limit is hit before the model submits, the run is scored as it stands.
- **Transcript:** saved as JSONL under `<data dir>/agent_runs/`.

## Your own agent

Any MCP client works. Point it at `wsp mcp` (stdio). For scripts, `wsp call <tool> '<json args>'` calls a single tool
with exactly an MCP agent's permissions, and `wsp_toolkit.agents.clients.InProcessClient` does the same from Python.
