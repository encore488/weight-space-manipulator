"""``wsp leaderboard export --model M [--by HANDLE] [--out FILE] | verify FILE... | site --out DIR``"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str]) -> None:
    from ..core import leaderboard as LB
    from ..core.service import Toolkit

    ap = argparse.ArgumentParser(prog="wsp leaderboard")
    ap.add_argument("cmd", choices=["export", "verify", "site"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--model", help="(export) agent_model whose runs to export")
    ap.add_argument("--by", default="", help="(export) your name or GitHub handle")
    ap.add_argument("--notes", default="", help="(export) e.g. effort level, harness details")
    ap.add_argument("--harness", help="(export) only runs from this harness, e.g. wsp-agent-runner")
    ap.add_argument("--out", help="(export) output file  |  (site) output directory")
    ap.add_argument("--submissions", default="leaderboard/submissions", help="(site) submissions directory")
    ap.add_argument("--no-verify", action="store_true", help="(site) skip re-scoring (preview only)")
    a = ap.parse_args(argv)
    tk = Toolkit()
    if a.cmd == "export":
        if not a.model:
            raise SystemExit("--model is required")
        sub = LB.export(tk, a.model, a.by, a.notes, a.harness)
        out = Path(a.out or f"wsp-submission-{a.model.replace('/', '_')}.json")
        out.write_text(json.dumps(sub, indent=1))
        n = len(json.dumps(sub))
        print(f"wrote {out} ({len(sub['runs'])} runs, {n:,} characters)")
        print("submit it: https://github.com/encore488/weight-space-manipulator/issues/new?template=leaderboard-submission.yml"
              + ("  (paste the file's contents)" if n < 60000 else "  (too large to paste: open a pull request adding "
                 "it to leaderboard/submissions/)"))
    elif a.cmd == "verify":
        ok = True
        for f in a.files:
            print(f"{f}:")
            res = LB.verify(tk, json.loads(Path(f).read_text()))
            for e in res.get("errors", []):
                print(f"  invalid: {e}")
            ok &= res["ok"]
        print("all submissions verified" if ok else "verification FAILED")
        sys.exit(0 if ok else 1)
    else:
        from ..site.build import build
        build(tk, Path(a.submissions), Path(a.out or "_site"), verify=not a.no_verify)
