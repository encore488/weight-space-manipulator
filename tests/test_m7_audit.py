from pathlib import Path

from wsp_toolkit.core.diff.validate import identified


def test_diff_models_and_report(tk, tmp_path, fixtures_dir):
    from wsp_toolkit.fixtures.finetunes import finetune_induction
    d, _ = fixtures_dir
    tuned = finetune_induction(d / "induction_tf", tmp_path / "ft", "ft_trigger", steps=200)
    a = tk.call("workspace_open", {"source": "fixture:induction_tf"})["session_id"]
    b = tk.call("workspace_open", {"source": tuned})["session_id"]
    wd = tk.call("diff_models", {"session_id": a, "other_session_id": b})
    assert wd["tensors_changed"] > 0 and wd["most_changed_tensors"][0]["rank_90"] >= 1
    assert tk.call("budget_status", {"session_id": a})["forward_passes"]["used"] == 0  # weights-only diff is free
    rep = tk.call("diff_report", {"session_id": a, "other_session_id": b, "n_probes": 20})
    assert rep["audit_id"].startswith("audit_")
    assert tk.call("budget_status", {"session_id": a})["forward_passes"]["used"] == 40
    md = tk.call("audit_get", {"audit_id": rep["audit_id"]})
    assert "Fine-tune audit" in (md.get("markdown") or md.get("text_chunk", ""))
    from fastapi.testclient import TestClient

    from wsp_toolkit.api.app import create_app
    c = TestClient(create_app(tk))
    assert c.get(f"/audits/{rep['audit_id']}").status_code == 200
    assert c.get(f"/audits/{rep['audit_id']}.md").text.startswith("# Fine-tune audit")
    pdf = c.get(f"/audits/{rep['audit_id']}.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


def test_audit_needs_outputs_access(tk):
    a = tk.call("workspace_open", {"source": "fixture:induction_tf", "access_level": "weights_only"})["session_id"]
    b = tk.call("workspace_open", {"source": "fixture:induction_tf_broken"})["session_id"]
    assert "most_changed_tensors" in tk.call("diff_models", {"session_id": a, "other_session_id": b})
    assert tk.call("diff_report", {"session_id": a, "other_session_id": b})["code"] == "access_denied"


def test_diff_localises_planted_bug(tk):
    """induction_tf vs induction_tf_broken differ only in layer-1 o_proj columns (the planted scale)."""
    a = tk.call("workspace_open", {"source": "fixture:induction_tf"})["session_id"]
    b = tk.call("workspace_open", {"source": "fixture:induction_tf_broken"})["session_id"]
    wd = tk.call("diff_models", {"session_id": a, "other_session_id": b})
    assert [t["tensor"] for t in wd["most_changed_tensors"]] == ["blocks.1.attn.o_proj.weight"]
    assert {h["head"] for h in wd["most_changed_heads"][:4]} == {"1.0", "1.1", "1.2", "1.3"}


def test_validator_rules():
    assert identified({"kind": "trigger", "trigger_tokens": ["t7"]},
                      [{"type": "trigger_candidate", "tokens": ["t7", "t1"], "text": "x"}])[0]
    assert not identified({"kind": "suppress_token", "tokens": ["t5"]},
                          [{"type": "vocab_direction", "pole_a": ["t5"], "pole_b": [], "text": "x"}])[0]
    assert identified({"kind": "none"}, [{"type": "no_major_behavior_change", "text": "x"}])[0]


def test_validation_suite_majority(tk, tmp_path):
    """M7 acceptance (small version): the report identifies the planted change in a clear majority of cases."""
    from wsp_toolkit.core.diff.validate import run_validation
    tk.call("zoo_build", {"scale": "mini"}, actor="human")
    r = run_validation(tk, Path(tmp_path / "ft"), n_mlp=2)
    assert r["rate"] > 0.6, r
