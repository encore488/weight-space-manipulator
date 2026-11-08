"""Milestone demos: ``wsp demo m1`` / ``wsp demo m3``. They print JSON reports; every action is logged."""

from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> None:
    from ..config import Config
    from ..core.service import Toolkit
    from .clients import InProcessClient

    which = argv[0] if argv else "m1"
    if which == "m1":
        from .induction_finder import find_induction_heads
        tk = Toolkit()
        rep = find_induction_heads(InProcessClient(tk))
        truth = json.loads((tk.cfg.subdir("fixtures") / "induction_tf" / "ground_truth.json").read_text())
        rep["ground_truth"] = {k: truth[k] for k in ("induction_heads", "previous_token_heads")}
        print(json.dumps(rep, indent=1))
    elif which == "m3":
        from .bugfix_agent import run
        # fixture models are tiny: one head's o_proj block is ~2.7% of all parameters, so the demo raises the
        # per-patch limit (documented in DECISIONS.md). Real models keep the 5% default.
        tk = Toolkit(Config.load(overrides={"policy": {"max_param_change_pct": 20.0}}))
        print(json.dumps(run(InProcessClient(tk)), indent=1))
    else:
        print("usage: wsp demo m1|m3")
        sys.exit(2)
