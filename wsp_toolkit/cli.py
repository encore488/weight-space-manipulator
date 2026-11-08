"""``wsp`` command-line entry point.

  wsp fixtures [--out DIR]              train the deterministic fixture models (prints their hashes)
  wsp ui [--port 8765]                  start the web UI / HTTP API
  wsp mcp [--access ...] [--budget N]   start the MCP server (stdio)
  wsp call TOOL '{"json": "args"}'      call one tool as an agent (handy for scripting and for LLM agents
                                        driving the toolkit through a shell); prints JSON
  wsp approve PATCH --reviewer NAME     approve a tested patch as a human (same as the UI button)
  wsp zoo build [--per-family N]        generate the network zoo (M5)
  wsp zoo baseline                      run the scripted baseline agent over the zoo
  wsp features build SESSION_OR_FIXTURE build the feature index (M6) for a fixture
  wsp demo m1|m3                        run the scripted milestone demos
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "fixtures":
        from .fixtures.train import main as fx
        sys.argv = ["wsp-fixtures", *rest]
        fx()
    elif cmd == "ui":
        from .api.app import main as ui
        ui(rest)
    elif cmd == "mcp":
        from .mcp_server.server import main as mcp
        mcp(rest)
    elif cmd == "call":
        from .core.service import Toolkit
        ap = argparse.ArgumentParser(prog="wsp call")
        ap.add_argument("tool")
        ap.add_argument("args", nargs="?", default="{}")
        ap.add_argument("--actor", default="agent", choices=["agent"])
        a = ap.parse_args(rest)
        args = json.loads(a.args)
        print(json.dumps(Toolkit().call(a.tool, args, actor="agent"), indent=1))
    elif cmd == "approve":
        from .core.service import Toolkit
        ap = argparse.ArgumentParser(prog="wsp approve")
        ap.add_argument("patch_id")
        ap.add_argument("--reviewer", required=True)
        ap.add_argument("--note", default="")
        ap.add_argument("--acknowledge-flags", action="store_true")
        a = ap.parse_args(rest)
        print(json.dumps(Toolkit().call("patch_approve", {"patch_id": a.patch_id, "reviewer": a.reviewer,
                                                          "note": a.note, "acknowledge_flags": a.acknowledge_flags},
                                        actor="human"), indent=1))
    elif cmd == "zoo":
        from .core.zoo.cli import main as zoo
        zoo(rest)
    elif cmd == "features":
        from .core.features.cli import main as feats
        feats(rest)
    elif cmd == "demo":
        from .agents.demos import main as demo
        demo(rest)
    else:
        print(f"unknown command {cmd!r}\n{__doc__}")
        sys.exit(2)


if __name__ == "__main__":
    main()
