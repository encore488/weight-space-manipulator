"""Static website: landing page, challenges, public leaderboard, how to submit.

    wsp leaderboard site --out _site                  # re-scores every submission, then renders
    wsp leaderboard site --out _site --no-verify      # quick preview

Deployed to GitHub Pages by .github/workflows/site.yml. Every build re-verifies every submission from scratch, so
the "verified" badge always reflects this build of the toolkit.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import __version__
from ..challenge_defs import BENCH_VERSION, CHALLENGES, SCRIPTED_BASELINE

HERE = Path(__file__).resolve().parent
REPO = "encore488/weight-space-manipulator"
REPO_URL = f"https://github.com/{REPO}"
ISSUE_URL = f"{REPO_URL}/issues/new?template=leaderboard-submission.yml"
PAPER_URL = "https://rikiyat.github.io/blog/weight-space-programming/"


def build(tk, submissions: Path, out: Path, verify: bool = True, log=print) -> dict:
    from ..core import leaderboard as LB

    subs = LB.load_dir(submissions) if submissions.exists() else []
    rejected = []
    accepted = []
    for s in subs:
        errs = LB.validate_schema(s)
        if errs:
            rejected.append({"file": s["_file"], "reason": "; ".join(errs)})
            continue
        if verify:
            log(f"verifying {s['_file']} ({s['agent_model']}, {len(s['runs'])} runs)")
            res = LB.verify(tk, s, log=log)
            if not res["ok"]:
                bad = [r for r in res["runs"] if not r["ok"]]
                rejected.append({"file": s["_file"], "reason": f"{len(bad)} run(s) did not re-score: " + ", ".join(
                    f"{r['challenge']} claimed {r['claimed']} got {r['rescored']}" for r in bad[:4])})
                continue
            for r, v in zip(s["runs"], res["runs"]):
                r["rescored"] = v["rescored"]
            s["_verified"] = True
        accepted.append(s)
    board = LB.aggregate(accepted)
    board["by_challenge"] = {c: sorted([e for e in board["entries"] if c in e["per_challenge"]],
                                       key=lambda e, c=c: -e["per_challenge"][c]) for c in CHALLENGES}
    board["generated"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    board["verified_by_build"] = verify
    board["rejected"] = rejected

    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    (out / "data" / "leaderboard.json").write_text(json.dumps(board, indent=1))
    css = (HERE.parent / "ui" / "static" / "style.css").read_text() + "\n" + (HERE / "templates" / "site.css").read_text()
    (out / "style.css").write_text(css)
    shot = HERE.parent.parent / "docs" / "img" / "challenges.png"  # present in a repo checkout (CI)
    if shot.exists():
        shutil.copy(shot, out / "challenges.png")
    env = Environment(loader=FileSystemLoader(HERE / "templates"), autoescape=select_autoescape(["html"]))
    ctx = {"challenges": [{"id": k, **c} for k, c in CHALLENGES.items()], "board": board, "version": __version__,
           "bench": BENCH_VERSION, "repo_url": REPO_URL, "issue_url": ISSUE_URL, "paper_url": PAPER_URL,
           "baseline": SCRIPTED_BASELINE, "short": {k: k.replace("-", " ") for k in CHALLENGES}}
    for page in ("index.html", "leaderboard.html", "submit.html"):
        (out / page).write_text(env.get_template(page).render(**ctx, page=page))
    (out / ".nojekyll").write_text("")
    log(f"site written to {out} ({len(accepted)} entries, {len(rejected)} rejected)")
    return board
