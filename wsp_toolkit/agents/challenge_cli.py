"""``wsp challenge list | prepare [ID] | prompt ID [--model NAME] | scores [ID] | report [--model NAME] | baseline [ID]``"""

from __future__ import annotations

import argparse
import json

PROMPT = """You have the WSP Toolkit MCP server (tools named like mcp__wsp__...). Use it to do the WSP challenge \
"{cid}" ({title}).

1. Call challenge_start with challenge_id="{cid}" and agent_model="{model}". It returns your session ids, the \
brief, your access level, your forward-pass budget and the answer format.
2. Work the challenge using only the WSP tools. Do not read files on disk, run shell commands, or open other models.
3. Call challenge_submit once with your answer, then tell me your score and, briefly, how you found it."""


def main(argv: list[str]) -> None:
    from ..core.challenges import CHALLENGES
    from ..core.service import Toolkit

    ap = argparse.ArgumentParser(prog="wsp challenge")
    ap.add_argument("cmd", choices=["list", "prompt", "scores", "report", "baseline", "prepare"])
    ap.add_argument("challenge", nargs="?")
    ap.add_argument("--model", default=None, help="model name to record (prompt) or to filter by (report)")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for cid, c in CHALLENGES.items():
            print(f"{cid:18s} level {c['level']}  {c['title']}\n{'':18s} {c['summary']}\n{'':18s} access "
                  f"{c['access_level']}, {c['budget']} forward passes per session\n")
        print("First time? Build the models once (~1-2 min):  wsp challenge prepare")
        print("Try one with Claude Code:  wsp challenge prompt fix-the-copier --model <your model>")
        print("Benchmark via the API:     wsp agent run fix-the-copier --model claude-fable-5-2")
        return
    if a.cmd == "prompt":
        if a.challenge not in CHALLENGES:
            raise SystemExit(f"choose one of: {', '.join(CHALLENGES)}")
        model = a.model or "<the model you are, e.g. claude-fable-5-2>"
        print(f"# 1) start a server locked to this challenge (once):\n"
              f"#    claude mcp add wsp -- wsp mcp --challenge {a.challenge}\n# 2) paste this prompt:\n")
        print(PROMPT.format(cid=a.challenge, title=CHALLENGES[a.challenge]["title"], model=model))
        return
    tk = Toolkit()
    if a.cmd == "prepare":
        import time
        for cid in ([a.challenge] if a.challenge else list(CHALLENGES)):
            t = time.time()
            n = len(tk.challenges.sources(cid))
            print(f"{cid:18s} ready ({n} model(s), {time.time() - t:.0f}s)", flush=True)
        print("All set. Next: claude mcp add wsp -- wsp mcp --challenge <id>   or   wsp agent run <id> --model ...")
        return
    if a.cmd == "baseline":
        from .challenge_baseline import run_all
        for r in run_all(tk, [a.challenge] if a.challenge else None):
            print(json.dumps({k: r[k] for k in ("challenge", "score", "usage")}))
        return
    if a.cmd == "scores":
        rows = tk.challenges.leaderboard(a.challenge)
        if not rows:
            print("no submitted runs yet")
        for r in rows:
            print(json.dumps(r))
    else:
        print(tk._challenge_report(a.model))
