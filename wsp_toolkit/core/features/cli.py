"""``wsp features build [--fixture induction_tf] [--sizes 32,128] [--steps 3000]``"""

from __future__ import annotations

import argparse
import json


def main(argv: list[str]) -> None:
    from ..service import Toolkit
    from .index import induction_fixture_corpus

    ap = argparse.ArgumentParser(prog="wsp features")
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--fixture", default="induction_tf")
    ap.add_argument("--hooks", default="L0.resid_post,L1.resid_post")
    ap.add_argument("--sizes", default="32,128")
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args(argv)
    tk = Toolkit()
    ws = tk.wsm.open(f"fixture:{a.fixture}", label=f"feature index build ({a.fixture})")
    rep = tk.features.build(ws, a.hooks.split(","), [int(x) for x in a.sizes.split(",")], induction_fixture_corpus(),
                            steps=a.steps)
    print(json.dumps(rep, indent=1))
