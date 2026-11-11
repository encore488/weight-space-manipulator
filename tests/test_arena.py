"""Arena: event log -> animation timeline, the UI pages, standalone export, the website page, the renderer's syntax."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from wsp_toolkit.core.arena import build_timeline, export_html, standalone_html


def _oracle_ops(tk, seed):
    t = tk.challenges.instance("fix-the-copier", seed)["truth"]
    return [{"op": "scale", "tensor": f"blocks.{h.split('.')[0]}.attn.o_proj.weight", "head": int(h.split(".")[1]),
             "factor": 1 / f} for h, f in t["factors"].items()]


def _scripted_run(tk, seed=3):
    r = tk.call("challenge_start", {"challenge_id": "fix-the-copier", "agent_model": "test-agent", "seed": seed})
    sid = r["sessions"][0]["session_id"]
    assert tk.call("narrate", {"session_id": sid, "text": "Looking at the head norms first."})["shown"]
    tk.call("tensor_stats", {"session_id": sid, "tensor": "blocks.1.attn.o_proj.weight"})
    tk.call("tensor_stats", {"session_id": sid, "tensor": "blocks.0.attn.o_proj.weight"})
    pid = tk.call("edit_apply", {"session_id": sid, "rationale": "undo the damage",
                                 "operations": _oracle_ops(tk, seed)})["patch_id"]
    tk.call("suite_run_practice", {"session_id": sid, "suite_id": "induction_practice", "limit": 4})
    tk.call("patch_revert", {"session_id": sid, "patch_id": pid})
    tk.call("patch_reapply", {"session_id": sid, "patch_id": pid})
    tk.call("tensor_stats", {"session_id": sid, "tensor": "no.such.tensor"})  # an error beat
    s = tk.call("challenge_submit", {"run_id": r["run_id"], "answer": {}})
    return r, pid, s


def test_timeline_from_event_log(tk):
    r, pid, s = _scripted_run(tk)
    assert s["usage"]["tool_calls"] == 7  # narrate is free and not counted
    tl = build_timeline(tk, r["run_id"])
    json.dumps(tl)
    types = [b["type"] for b in tl["beats"]]
    assert types[0] == "intro" and types[-2:] == ["submit", "outro"]
    assert ["note", "scan", "scan", "edit", "practice", "revert", "edit", "error"] == types[1:-2]
    note, scan, scan2, edit, practice, revert, reapply = tl["beats"][1:8]
    assert note["text"] == "Looking at the head norms first."
    assert scan["target"] == {"kind": "heads", "layer": 1, "proj": "o"} and scan2["dur"] < scan["dur"]
    assert edit["patch_id"] == pid and edit["text"] == "undo the damage" and edit["params"] > 0
    assert edit["test"]["after"] > edit["test"]["before"]  # the automatic regression check after apply
    damaged = set(tk.challenges.instance("fix-the-copier", 3)["truth"]["factors"])
    assert set(tl["initial"]["damaged"]) == damaged
    assert {f"{t['layer']}.{t['heads'][0]}" for t in edit["targets"]} == damaged
    assert practice["score"] is not None and practice["budget_used"] == 4
    assert revert["patch_id"] == pid and reapply["reapply"] and reapply["targets"] == edit["targets"]
    starts = [b["t"] for b in tl["beats"]]
    assert starts == sorted(starts) and abs(tl["duration"] - (starts[-1] + tl["beats"][-1]["dur"])) < 1e-6
    assert tl["run"]["finished"] and tl["run"]["score"] == pytest.approx(s["score"], abs=1e-3)


def test_timeline_live_run_and_other_architectures(tk):
    r = tk.challenges.start("backdoor-removal", "m", "test", None, "agent", 2)
    tl = build_timeline(tk)  # latest run, still in progress
    assert tl["run"]["id"] == r["run_id"] and not tl["run"]["finished"]
    assert [b["type"] for b in tl["beats"]] == ["intro"]
    assert tl["arch"]["kind"] == "mlp" and tl["arch"]["d_in"] == 8 and tl["initial"]["trigger_inputs"]
    r2 = tk.challenges.start("token-ban", "m", "test", None, "agent", 4)
    assert build_timeline(tk, r2["run_id"])["initial"]["goal"]["target"].startswith("t")


def test_arena_pages_and_export(tk, tmp_path):
    from fastapi.testclient import TestClient

    from wsp_toolkit.api.app import create_app
    r, _, _ = _scripted_run(tk)
    c = TestClient(create_app(tk))
    page = c.get("/arena")
    assert page.status_code == 200 and "arena.js" in page.text and r["run_id"] in page.text
    assert c.get(f"/arena/{r['run_id']}").status_code == 200
    assert c.get("/api/arena/latest").json()["run"]["id"] == r["run_id"]
    exp = c.get(f"/arena/{r['run_id']}/export.html?size=1080x1350")
    assert exp.status_code == 200 and "attachment" in exp.headers["content-disposition"]
    assert 'width="1080" height="1350"' in exp.text and "WSPArena.create" in exp.text
    assert "data:font/woff2;base64," in exp.text  # self-contained: works offline
    assert c.get(f"/arena/{r['run_id']}/export.html?size=huge").status_code == 400
    assert c.get("/arena/cr_nope").status_code == 404 and c.get("/api/arena/cr_nope").status_code == 404
    assert "/arena/" in c.get("/challenges").text
    out = export_html(tk, r["run_id"], tmp_path / "a.html")
    html = out.read_text()
    assert "</script>" not in html.split("const TIMELINE = ", 1)[1].split(";\n", 1)[0]


def test_site_arena_page(tk, tmp_path):
    from wsp_toolkit.site.build import build
    r, _, _ = _scripted_run(tk)
    reps = tmp_path / "replays"
    reps.mkdir()
    tl = build_timeline(tk, r["run_id"])
    tl["note"] = "test replay"
    (reps / "00-test.json").write_text(json.dumps(tl))
    out = tmp_path / "site"
    build(tk, tmp_path / "no-submissions", out, verify=False, log=lambda *_: None, replays=reps)
    html = (out / "arena.html").read_text()
    assert "test replay" in html and "test-agent" in html and 'src="arena.js"' in html
    assert (out / "arena.js").exists() and (out / "fonts" / "orbitron-latin-800-normal.woff2").exists()
    assert 'href="arena.html"' in (out / "index.html").read_text()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_renderer_syntax():
    js = Path(__file__).resolve().parents[1] / "wsp_toolkit" / "ui" / "static" / "arena.js"
    subprocess.run(["node", "--check", str(js)], check=True)


def test_record_video(tk, tmp_path):
    """Frame-exact MP4 export: only where Playwright + a Chromium are installed (the [video] extra)."""
    pytest.importorskip("playwright.sync_api")
    from wsp_toolkit.agents.arena_cli import _ffmpeg, record
    try:
        _ffmpeg()
    except SystemExit:
        pytest.skip("no ffmpeg")
    r, _, _ = _scripted_run(tk)
    html = tmp_path / "a.html"
    html.write_text(standalone_html(build_timeline(tk, r["run_id"]), "640x360"))
    try:
        res = record(html, tmp_path / "a.mp4", "640x360", fps=5, end=1.0, verbose=False)
    except Exception as e:  # no browser binary available
        if "Executable doesn't exist" in str(e) or "playwright install" in str(e):
            pytest.skip("no Chromium for Playwright")
        raise
    assert res["frames"] == 5 and (tmp_path / "a.mp4").stat().st_size > 1000


def test_api_runner_text_becomes_live_captions(tk):
    """The API runner logs the model's text blocks as note events, so the arena can caption them live."""
    from types import SimpleNamespace as NS

    from test_challenges import FakeClient, _run_id

    from wsp_toolkit.agents.api_runner import run
    script = [{"stop": "tool_use", "content": [NS(type="text", text="Output norms in layer 1 look tiny."),
                                               lambda kw: NS(type="tool_use", id="t1", name="challenge_submit",
                                                             input={"run_id": _run_id(kw), "answer": {}})]}]
    res = run(tk, "fix-the-copier", "claude-fable-5-2", client=FakeClient(script), verbose=False)
    notes = [b for b in build_timeline(tk, res["run_id"])["beats"] if b["type"] == "note"]
    assert [n["text"] for n in notes] == ["Output norms in layer 1 look tiny."] and notes[0]["src"] == "text"
