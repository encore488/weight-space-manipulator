"""``wsp zoo build|baseline|scores``."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str]) -> None:
    from ...agents.clients import InProcessClient
    from ...agents.zoo_baseline import run_baseline
    from ...config import Config
    from ..service import Toolkit

    ap = argparse.ArgumentParser(prog="wsp zoo")
    ap.add_argument("cmd", choices=["build", "baseline", "scores"])
    ap.add_argument("--scale", default="full", choices=["full", "mini"])
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--split")
    ap.add_argument("--budget", type=int, default=120, help="forward-pass budget per baseline session")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        tk = Toolkit()
        print(json.dumps(tk.call("zoo_build", {"scale": a.scale, "fresh": a.fresh}, actor="human"), indent=1))
    elif a.cmd == "baseline":
        tk = Toolkit(Config.load(overrides={"session": {"forward_pass_budget": a.budget}}))
        print(json.dumps(run_baseline(InProcessClient(tk), split=a.split, limit=a.limit), indent=1))
    else:
        print(json.dumps(Toolkit().call("zoo_scores", {}), indent=1))
