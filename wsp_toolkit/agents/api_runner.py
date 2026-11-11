"""Benchmark a Claude model on a WSP challenge through the Anthropic API.

    pip install anthropic
    export ANTHROPIC_API_KEY=...          # or `ant auth login`
    wsp agent run fix-the-copier --model claude-fable-5-2

The model gets ONLY the toolkit's tools (no shell, no files), so these runs are the comparable ones on the
challenge leaderboard (harness "wsp-agent-runner"). The server is locked to the challenge's models.

Loop: manual tool-use loop over streaming Messages requests (append-only history, so preserved thinking works),
eager input streaming with client-side schema validation, automatic prompt caching, server-side refusal fallback
on models that support it, and hard limits on turns and estimated cost.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# Tools the model does not get: the runner opens the challenge itself, and the model must not open other models
# or write to the zoo leaderboard.
EXCLUDED_TOOLS = {"workspace_open", "session_start", "challenge_start", "challenge_list", "challenge_scores",
                  "zoo_list_tasks", "zoo_submit", "zoo_scores",
                  "narrate"}  # the runner already shows the model's text and thinking as live captions

# Models that accept server-side refusal fallbacks (`fallbacks: "default"`).
FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-mythos-5-2", "claude-mythos-5-1",
                   "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# $ per million tokens: (input, output, cache read). Cache writes are estimated at 1.25x input. Estimates only.
PRICES = {"claude-fable-5-2": (8.0, 40.0, 0.20), "claude-mythos-5-2": (8.0, 40.0, 0.20),
          "claude-fable-5-1": (10.0, 50.0, 1.0), "claude-fable-5": (10.0, 50.0, 1.0),
          "claude-opus-5-5": (4.0, 20.0, 0.20), "claude-opus-5": (5.0, 25.0, 0.5),
          "claude-opus-4-8": (5.0, 25.0, 0.5), "claude-sonnet-5-5": (2.0, 10.0, 0.10),
          "claude-sonnet-5": (2.0, 10.0, 0.2), "claude-haiku-5-5": (0.10, 0.50, 0.01)}

SYSTEM = """You are an AI agent doing weight-space programming: you read, probe and edit the weights of small \
neural networks directly, through the WSP Toolkit's tools. Weight tools are free; forward passes are budgeted per \
session; some tools are unavailable at lower access levels. Tool results are JSON; very large results come back as \
a handle you can page with result_page. Edits must go through edit_apply with a rationale; you can revert them.

You are being evaluated on a challenge. Work it out from the evidence the tools give you, then call \
challenge_submit exactly once with your answer. Your final message should briefly say what you found."""


def tool_definitions(tk) -> list[dict]:
    """The same JSON schemas the MCP server advertises, minus excluded tools, for the Messages API."""
    import anyio

    from ..mcp_server.server import build_server

    server = build_server(tk)
    tools = anyio.run(server.list_tools)
    out = []
    for t in tools:
        if t.name in EXCLUDED_TOOLS:
            continue
        out.append({"name": t.name, "description": t.description or t.name, "input_schema": t.input_schema,
                    "eager_input_streaming": True})
    return out


def _valid(args: Any, schema: dict) -> str | None:
    import jsonschema

    if not isinstance(args, dict):
        return "tool input is not an object"
    try:
        jsonschema.validate(args, schema)
    except jsonschema.ValidationError as e:
        return e.message
    return None


def _cost(model: str, u: dict) -> float | None:
    p = PRICES.get(model)
    if not p:
        return None
    pin, pout, pread = p
    return (u["input_tokens"] * pin + u["cache_creation_input_tokens"] * pin * 1.25 +
            u["cache_read_input_tokens"] * pread + u["output_tokens"] * pout) / 1e6


def _block_dict(b) -> dict:
    t = getattr(b, "type", None)
    if t == "text":
        return {"type": "text", "text": b.text}
    if t == "thinking":
        return {"type": "thinking", "summary": getattr(b, "thinking", "")}
    if t == "tool_use":
        return {"type": "tool_use", "name": b.name, "input": b.input}
    return {"type": str(t)}


def run(tk, challenge_id: str, model: str, client=None, effort: str | None = "high", thinking: str = "summarized",
        max_turns: int = 80, max_cost: float = 20.0, max_tokens: int = 64000, label: str | None = None,
        verbose: bool = True, seed: int | None = None) -> dict:
    from ..mcp_server.server import lock_to_challenge

    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    lock_to_challenge(tk, challenge_id)
    start = tk.challenges.start(challenge_id, model, "wsp-agent-runner", label or f"api runner: {model}", "agent",
                                seed)
    run_id = start["run_id"]
    note_sid = (start.get("sessions") or [{}])[0].get("session_id")
    tools = tool_definitions(tk)
    schemas = {t["name"]: t["input_schema"] for t in tools}
    log_path = tk.cfg.subdir("agent_runs") / f"{run_id}.jsonl"
    log = log_path.open("a")

    def note(entry: dict) -> None:
        log.write(json.dumps({"t": round(time.time(), 2), **entry}, default=str) + "\n")
        log.flush()

    brief = {k: start[k] for k in ("run_id", "challenge", "seed", "title", "brief", "parameters", "access_level",
                                   "budget_per_session", "sessions", "answer_format", "rules")}
    messages: list[dict] = [{"role": "user", "content": "Challenge:\n" + json.dumps(brief, indent=1)}]
    note({"type": "start", "challenge": challenge_id, "model": model, "run_id": run_id})
    kwargs: dict = {"model": model, "max_tokens": max_tokens, "system": SYSTEM, "tools": tools,
                    "cache_control": {"type": "ephemeral"}}
    if effort:
        kwargs["output_config"] = {"effort": effort}
    if thinking == "summarized":
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
    if model in FALLBACK_MODELS:
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    submitted = None
    stop = "max_turns"
    json_retries = 0
    warned = nudged = False
    turn = 0
    while turn < max_turns:
        turn += 1
        try:
            with client.beta.messages.stream(messages=messages, **kwargs) as stream:
                for _event in stream:
                    pass
                response = stream.get_final_message()
            json_retries = 0
        except ValueError:
            # JSON the SDK could not parse at all (eager input streaming): no tool_use id to answer, re-issue
            json_retries += 1
            if json_retries > 2:
                stop = "unparseable_tool_input"
                break
            turn -= 1
            continue
        for k in usage:
            usage[k] += int(getattr(response.usage, k, 0) or 0)
        for b in response.content:
            d = _block_dict(b)
            note({"turn": turn, **d})
            said = d.get("text") if d["type"] == "text" else d.get("summary") if d["type"] == "thinking" else None
            if said and str(said).strip() and note_sid:
                tk.store.log_event(session_id=note_sid, actor="agent", kind="note", tool=d["type"],
                                   result_summary=str(said).strip())
            if verbose and d["type"] == "text" and d["text"].strip():
                print(f"\n[{model}] {d['text'].strip()}\n", flush=True)
            if verbose and d["type"] == "tool_use":
                print(f"  -> {d['name']}({json.dumps(d['input'])[:160]})", flush=True)

        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason == "refusal":
            stop = "refusal"
            break
        if response.stop_reason == "max_tokens" and tool_uses:
            stop = "max_tokens"
            break
        messages.append({"role": "assistant", "content": response.content})
        if not tool_uses:
            if submitted is None and not nudged and turn < max_turns:
                nudged = True
                messages.append({"role": "user", "content": "You have not called challenge_submit yet. Submit your "
                                 "answer now (once)."})
                continue
            stop = "end_turn"
            break
        results = []
        for b in tool_uses:
            err = _valid(b.input, schemas.get(b.name, {"type": "object"})) if b.name in schemas else \
                f"unknown tool {b.name!r}"
            if err:
                content, is_error = json.dumps({"INVALID_JSON": json.dumps(b.input, default=str), "why": err}), True
            else:
                r = tk.call(b.name, dict(b.input), actor="agent")
                is_error = isinstance(r, dict) and "error" in r
                content = json.dumps(r, default=str)
                if b.name == "challenge_submit" and not is_error:
                    submitted = r
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": content,
                            **({"is_error": True} if is_error else {})})
            note({"turn": turn, "type": "tool_result", "name": b.name, "is_error": is_error, "chars": len(content)})
        cost = _cost(model, usage)
        if submitted is not None:
            messages.append({"role": "user", "content": results})
            stop = "submitted"
            break
        if not warned and ((cost is not None and cost > 0.8 * max_cost) or turn >= max_turns - 3):
            warned = True
            results.append({"type": "text", "text": "Runner note: you are close to the turn/cost limit for this "
                            "run. Submit your answer with challenge_submit within the next turn or two."})
        messages.append({"role": "user", "content": results})
        if cost is not None and cost > max_cost:
            stop = "max_cost"
            break

    if submitted is None:  # limits hit: score whatever state the run is in (empty answer)
        submitted = tk.challenges.submit(run_id, {})
        submitted["auto_submitted"] = True
    cost = _cost(model, usage)
    tk.store.execute("UPDATE challenge_runs SET usage_json=? WHERE id=?",
                     (json.dumps({**(submitted.get("usage") or {}), "api_usage": usage,
                                  "estimated_cost_usd": round(cost, 3) if cost is not None else None,
                                  "turns": turn, "stop": stop}), run_id))
    result = {"run_id": run_id, "challenge": challenge_id, "seed": start["seed"], "model": model,
              "score": submitted.get("score"),
              "detail": submitted.get("detail"), "auto_submitted": submitted.get("auto_submitted", False),
              "turns": turn, "stop": stop, "api_usage": usage,
              "estimated_cost_usd": round(cost, 3) if cost is not None else None, "transcript": str(log_path)}
    note({"type": "end", **result})
    log.close()
    return result


def main(argv: list[str]) -> None:
    import argparse

    from ..core.challenges import CHALLENGES
    from ..core.service import Toolkit

    ap = argparse.ArgumentParser(prog="wsp agent run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "bench"])
    ap.add_argument("challenge", nargs="?", choices=sorted(CHALLENGES), help="(run only)")
    ap.add_argument("--seed", type=int, help="instance seed (default: random)")
    ap.add_argument("--model", default="claude-fable-5-2")
    ap.add_argument("--effort", default="high", help="low|medium|high|xhigh|max, or 'none' to omit")
    ap.add_argument("--thinking", default="summarized", choices=["summarized", "omit"])
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--max-cost", type=float, default=20.0, help="stop when the estimated cost exceeds this (USD)")
    ap.add_argument("--runs", type=int, default=1, help="repeat the challenge N times")
    a = ap.parse_args(argv)
    try:
        import anthropic
    except ImportError as e:
        raise SystemExit("the API runner needs the Anthropic SDK: pip install anthropic") from e
    tk = Toolkit()
    if a.cmd == "run" and not a.challenge:
        raise SystemExit("usage: wsp agent run <challenge> --model ...   (or: wsp bench run --model ...)")
    plan = [(a.challenge, i) for i in range(a.runs)] if a.cmd == "run" else \
        [(c, i) for c in CHALLENGES for i in range(a.runs)]
    results = []
    for cid, i in plan:
        try:
            r = run(tk, cid, a.model, effort=None if a.effort == "none" else a.effort, thinking=a.thinking,
                    max_turns=a.max_turns, max_cost=a.max_cost, label=f"api runner: {a.model} #{i + 1}",
                    seed=a.seed)
        except anthropic.AuthenticationError as e:
            raise SystemExit("Anthropic API authentication failed: set ANTHROPIC_API_KEY or run `ant auth login`") from e
        except anthropic.NotFoundError as e:
            raise SystemExit(f"model {a.model!r} was not found for your API key: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise SystemExit(f"Anthropic API error {e.status_code}: {e.message}") from e
        results.append(r)
        print(json.dumps({k: r[k] for k in ("challenge", "seed", "run_id", "score", "turns", "stop",
                                            "estimated_cost_usd")}), flush=True)
    print(f"\nmean score over {len(results)} run(s): {sum(r['score'] or 0 for r in results) / len(results):.3f}")
    total = sum(r["estimated_cost_usd"] or 0 for r in results)
    print(f"estimated API cost: ${total:.2f}")
    print(f"next: wsp leaderboard export --model {a.model} --by <your GitHub handle>   (then submit it on the site)")


_ = Path
