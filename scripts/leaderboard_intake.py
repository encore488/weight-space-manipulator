"""Leaderboard intake for GitHub issues (used by .github/workflows/leaderboard-intake.yml).

    python scripts/leaderboard_intake.py --issue 12 --author octocat --body-env ISSUE_BODY --comment comment.md \
        [--write-dir leaderboard/submissions]

Extracts the submission JSON from the issue body, re-scores it with the toolkit, writes a Markdown comment, and
(with --write-dir, only when it verifies) writes the submission file. Exit code 0 = verified, 1 = not.
The issue body is untrusted input: it is only ever parsed as JSON, never executed or interpolated into a shell.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def extract_json(body: str) -> dict:
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", body, re.S)
    text = m.group(1) if m else body[body.find("{"):body.rfind("}") + 1]
    return json.loads(text)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", s.lower()).strip("-")[:60] or "model"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", type=int, required=True)
    ap.add_argument("--author", default="")
    ap.add_argument("--body-env", default="ISSUE_BODY")
    ap.add_argument("--comment", default="comment.md")
    ap.add_argument("--write-dir")
    a = ap.parse_args()
    body = os.environ.get(a.body_env, "")
    lines = [f"### WSP Bench submission check (issue #{a.issue})", ""]
    try:
        sub = extract_json(body)
    except (ValueError, json.JSONDecodeError) as e:
        lines += [f"❌ Couldn't read the submission JSON: `{e}`.",
                  "Paste the complete file from `wsp leaderboard export` into the *Submission file* box."]
        Path(a.comment).write_text("\n".join(lines) + "\n")
        sys.exit(1)

    from wsp_toolkit.core import leaderboard as LB
    from wsp_toolkit.core.service import Toolkit

    if not sub.get("submitted_by") and a.author:
        sub["submitted_by"] = a.author
    sub["issue"] = a.issue
    log: list[str] = []
    res = LB.verify(Toolkit(), sub, log=log.append)
    if res.get("errors"):
        lines += ["❌ The file isn't a valid submission:", ""] + [f"- {e}" for e in res["errors"]]
    else:
        ok = sum(r["ok"] for r in res["runs"])
        lines += [("✅ " if res["ok"] else "❌ ") + f"{ok} of {len(res['runs'])} runs re-scored to their claimed scores "
                  f"(model `{sub['agent_model']}`).", "", "<details><summary>Per-run results</summary>", "", "```"]
        lines += log + ["```", "</details>", ""]
        if res["ok"]:
            board = LB.aggregate([{**sub, "runs": [{**r, "rescored": v["rescored"]}
                                                    for r, v in zip(sub["runs"], res["runs"])]}])
            e = board["entries"][0]
            lines += [f"WSP-Bench score: **{e['bench_score']:.3f}**" + ("" if e["complete"] else
                      " (partial: not every challenge was run; missing ones count 0)"), "",
                      "A maintainer can add this to the leaderboard by commenting `/accept`."]
        else:
            lines += ["Runs that don't re-score usually mean the file was edited, or it came from an older toolkit "
                      "version. Re-export with the latest toolkit and edit the issue to try again."]
    Path(a.comment).write_text("\n".join(lines) + "\n")
    if res["ok"] and a.write_dir:
        d = Path(a.write_dir)
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{time.strftime('%Y-%m-%d')}-{slug(sub['agent_model'])}-issue{a.issue}.json"
        f.write_text(json.dumps(sub, indent=1))
        print(f"wrote {f}")
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
