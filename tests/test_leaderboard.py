"""Public leaderboard: export -> verify (incl. tamper detection) -> site build; issue intake parsing."""

import json
import subprocess
import sys
from pathlib import Path


def _runs(tk):
    from wsp_toolkit.agents.challenge_baseline import run_all
    return run_all(tk, ["induction-hunt", "token-ban", "install-a-rule", "fix-the-copier"], seed=21)


def test_export_verify_and_tamper(tk):
    from wsp_toolkit.core import leaderboard as LB
    _runs(tk)
    sub = LB.export(tk, "scripted-baseline", "tester", "note")
    assert sub["schema"] == LB.SCHEMA and len(sub["runs"]) == 4
    rule = next(r for r in sub["runs"] if r["challenge"] == "install-a-rule")
    assert rule["edits"] and "vector" in json.dumps(rule["edits"])  # directions inlined
    res = LB.verify(tk, json.loads(json.dumps(sub)), log=lambda *_: None)
    assert res["ok"], res
    bad = json.loads(json.dumps(sub))
    bad["runs"][0]["score"] = 0.1 if bad["runs"][0]["score"] > 0.5 else 0.9
    assert not LB.verify(tk, bad, log=lambda *_: None)["ok"]
    bad = json.loads(json.dumps(sub))
    ban = next(r for r in bad["runs"] if r["challenge"] == "token-ban")
    assert ban["score"] > 0.9 and ban["edits"]
    ban["edits"] = {}  # claim the result without the edits that produced it
    assert not LB.verify(tk, bad, log=lambda *_: None)["ok"]
    assert LB.validate_schema({"schema": "x"})


def test_site_build(tk, tmp_path):
    from wsp_toolkit.core import leaderboard as LB
    from wsp_toolkit.site.build import build
    _runs(tk)
    d = tmp_path / "subs"
    d.mkdir()
    (d / "a.json").write_text(json.dumps(LB.export(tk, "scripted-baseline", "t")))
    (d / "broken.json").write_text(json.dumps({"schema": "nope"}))
    out = tmp_path / "site"
    board = build(tk, d, out, verify=True, log=lambda *_: None)
    assert len(board["entries"]) == 1 and board["entries"][0]["verified"] and len(board["rejected"]) == 1
    for page in ("index.html", "leaderboard.html", "submit.html", "style.css", "data/leaderboard.json"):
        assert (out / page).exists(), page
    html = (out / "leaderboard.html").read_text()
    assert "scripted-baseline" in html and "re-scored" in html and "Install rule" in html


def test_intake_extracts_json_from_issue_form():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from leaderboard_intake import extract_json, slug
    body = "### Model\n\nm\n\n### Submission file\n\n```json\n{\"schema\": \"wsp-leaderboard/v1\"}\n```\n"
    assert extract_json(body) == {"schema": "wsp-leaderboard/v1"}
    assert slug("Claude Fable 5.2 / test") == "claude-fable-5.2-test"


def test_cli_entrypoints_listed():
    out = subprocess.run([sys.executable, "-m", "wsp_toolkit.cli", "--help"], capture_output=True, text=True).stdout
    assert "wsp bench run" in out and "wsp leaderboard" in out
