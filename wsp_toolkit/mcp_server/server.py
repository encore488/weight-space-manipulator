"""MCP server: a thin layer exposing Toolkit tools to coding agents.

No business logic lives here. Each MCP tool is generated from the matching Toolkit method's signature and
forwards to ``Toolkit.call(name, args, actor="agent")``, which handles access control, budgets, size caps and
logging. Human-only tools (approval, sensitive registry) are never exposed.

Run:  wsp-mcp [--access weights_only] [--budget 25] [--data-dir DIR] [--config FILE] [--transport stdio]
Uses the MCP Python SDK 2.x (``mcp.server.mcpserver.MCPServer``; FastMCP was renamed in 2.0).
"""

from __future__ import annotations

import argparse
import inspect
import json
import typing

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .. import __version__
from ..config import Config
from ..core.service import AGENT_TOOLS, HUMAN_ONLY, Toolkit

INSTRUCTIONS = """WSP Toolkit: read, diagnose and edit neural-network weights directly.
Start with workspace_open(source=...) -> you get a session_id; pass it to every other tool.
Weight tools (tensor_*, direction_*) are free. Forward passes (run_forward, probe_*, practice suites) spend the
session's forward-pass budget; check budget_status. Held-out evaluations return aggregate scores only and are
rate-limited. Edits are made with edit_preview / edit_apply (always give a rationale) and become patches that a
human reviews; you cannot approve or export them yourself. Large results come back as a handle: use result_page.
"""


def _make_wrapper(tk: Toolkit, name: str, method):
    hints = typing.get_type_hints(method)
    sig = inspect.signature(method)
    params = []
    for p in sig.parameters.values():
        if p.name in ("self", "actor"):
            continue
        params.append(p.replace(annotation=hints.get(p.name, typing.Any)))

    def wrapper(**kwargs):
        result = tk.call(name, kwargs, actor="agent")
        if isinstance(result, dict) and "error" in result:
            raise ToolError(json.dumps(result))  # surfaces as isError=true with the actionable JSON message
        return result

    wrapper.__name__ = name
    wrapper.__doc__ = AGENT_TOOLS.get(name, "")
    wrapper.__signature__ = inspect.Signature(params, return_annotation=dict)
    wrapper.__annotations__ = {p.name: p.annotation for p in params} | {"return": dict}
    return wrapper


def build_server(tk: Toolkit | None = None) -> MCPServer:
    tk = tk or Toolkit()
    server = MCPServer("wsp-toolkit", instructions=INSTRUCTIONS, version=__version__, log_level="WARNING")
    for name in tk.tool_names():
        if name in HUMAN_ONLY or name not in AGENT_TOOLS:
            continue
        server.add_tool(_make_wrapper(tk, name, tk._tools[name]), name=name, description=AGENT_TOOLS[name])
    server._wsp_toolkit = tk  # for tests
    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="WSP Toolkit MCP server")
    ap.add_argument("--config", help="TOML config file (default: $WSP_CONFIG)")
    ap.add_argument("--data-dir", help="toolkit state directory (default: ~/.wsp_toolkit or $WSP_DATA_DIR)")
    ap.add_argument("--access", choices=["weights_only", "weights+activations", "weights+activations+outputs"],
                    help="ceiling access level for agent sessions")
    ap.add_argument("--budget", type=int, help="ceiling forward-pass budget per session")
    ap.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"])
    ap.add_argument("--challenge", help="lock the server to one challenge's models (see `wsp challenge list`)")
    a = ap.parse_args(argv)
    over: dict = {}
    if a.data_dir:
        over.setdefault("paths", {})["data_dir"] = a.data_dir
    if a.access:
        over.setdefault("session", {})["access_level"] = a.access
    if a.budget is not None:
        over.setdefault("session", {})["forward_pass_budget"] = a.budget
    tk = Toolkit(Config.load(a.config, overrides=over))
    if a.challenge:
        lock_to_challenge(tk, a.challenge)
    build_server(tk).run(a.transport)


def lock_to_challenge(tk: Toolkit, challenge_id: str) -> list[str]:
    """Agents may then open only this challenge's models, and its runs are marked 'locked'."""
    sources = tk.challenges.sources(challenge_id)
    tk.cfg.raw["session"]["allowed_sources"] = sources
    tk.cfg.raw["session"]["challenge_lock"] = challenge_id
    return sources


if __name__ == "__main__":
    main()
