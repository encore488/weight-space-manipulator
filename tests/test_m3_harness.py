import socket

import pytest

from wsp_toolkit.agents import bugfix_agent
from wsp_toolkit.agents.clients import InProcessClient
from wsp_toolkit.core.safety.netguard import NetworkBlocked, no_network


def test_suite_list_hides_heldout_examples(call, sid):
    suites = {s["id"]: s for s in call("suite_list", session_id=sid)["suites"]}
    assert "example_preview" in suites["induction_practice"]
    assert "example_preview" not in suites["induction_heldout"]
    assert "examples" not in suites["induction_heldout"]


def test_heldout_aggregate_only_and_rate_limited(tk, call, sid):
    r = call("evaluate_heldout", session_id=sid, suite_id="induction_heldout")
    assert set(r) >= {"run_id", "metrics", "settings"} and "details" not in r
    assert r["metrics"]["score"] > 0.9
    # held-out runs cannot be read back with details through any agent tool
    run = tk.harness.get_run(r["run_id"])
    assert "details" not in tk.harness.agent_view(run, allow_details=True)
    assert call("suite_run_practice", session_id=sid, suite_id="induction_heldout")["code"] == "access_denied"
    lim = tk.cfg.get("session.heldout_rate_limit_per_hour")
    for _ in range(lim - 1):
        assert "metrics" in call("evaluate_heldout", session_id=sid, suite_id="induction_heldout", limit=4)
    assert call("evaluate_heldout", session_id=sid, suite_id="induction_heldout")["code"] == "rate_limited"


def test_practice_costs_budget_and_records_settings(call, sid):
    r = call("suite_run_practice", session_id=sid, suite_id="induction_practice", limit=5)
    assert r["budget"]["used"] == 5 and len(r["details"]) == 5
    assert r["settings"]["temperature"] == 0.0 and "seed" in r["settings"]
    assert r["env"]["torch"] and r["env"]["network"].startswith("blocked")
    assert len(r["model_hash"]) == 16


def test_regression_runs_automatically_and_marks_tested(tk, call):
    s = call("workspace_open", source="fixture:induction_tf")["session_id"]
    p = call("edit_apply", session_id=s, rationale="break counting: zero the token embedding of t5",
             operations=[{"op": "zero", "tensor": "embed.weight", "slice": "5, :"}])
    assert p["regression"]["status"] == "tested"
    suites = {x["suite"]: x for x in p["regression"]["suites"]}
    assert set(suites) == {"counting_regression", "induction_regression"}
    show = call("patch_show", patch_id=p["patch_id"])
    assert show["status"] == "tested" and len(show["eval_runs"]) == 4


def test_side_effects_detected(call):
    s = call("workspace_open", source="fixture:induction_tf")["session_id"]
    p = call("edit_apply", session_id=s, rationale="damage the direct path",
             operations=[{"op": "scale", "tensor": "unembed.weight", "slice": "1:16, :", "factor": -1.0}])
    assert p["regression"]["side_effects"], p["regression"]


def test_baseline_and_compare(call):
    s = call("workspace_open", source="fixture:induction_tf_broken")["session_id"]
    base = call("suite_run_practice", session_id=s, suite_id="induction_practice", limit=4)
    p = call("edit_apply", session_id=s, rationale="restore", operations=[
        {"op": "scale", "tensor": "blocks.1.attn.o_proj.weight", "head": h, "factor": 10.0} for h in range(4)])
    after = call("suite_run_practice", session_id=s, suite_id="induction_practice", limit=4)
    cmp_ = call("compare_runs", run_a=base["run_id"], run_b=after["run_id"])
    assert cmp_["metrics"]["score"]["delta"] > 0.5 and not cmp_["warnings"]
    bl = call("run_baseline", session_id=s, suite_id="induction_practice", limit=4, settings={"temperature": 0.0,
                                                                                               "seed": 1})
    assert bl["model"] == "base" and bl["patch_id"] is None
    assert bl["metrics"]["score"] < 0.3  # the base (broken) model, even though a patch is applied
    assert call("workspace_info", session_id=s)["applied_patches"] == [p["patch_id"]]  # restored afterwards


def test_eval_network_guard():
    with no_network():
        with pytest.raises(NetworkBlocked):
            socket.create_connection(("example.com", 80), timeout=1)
        with pytest.raises(NetworkBlocked):
            socket.getaddrinfo("example.com", 80)


def test_eval_network_guard_inside_scorer(tk, sid, monkeypatch):
    from wsp_toolkit.core.evals import harness as H

    def evil(ad, scorer, ex, s):
        socket.create_connection(("example.com", 80), timeout=1)

    monkeypatch.setattr(H, "score_example", evil)
    r = tk.call("suite_run_practice", {"session_id": sid, "suite_id": "induction_practice", "limit": 1})
    assert "network access blocked" in r["error"] or "blocked" in r["error"]


def test_subprocess_isolation(tmp_path, fixtures_dir):
    from conftest import make_toolkit
    tk2 = make_toolkit(tmp_path, fixtures_dir, eval={"isolation": "subprocess"}, policy={"max_param_change_pct": 20})
    s = tk2.call("workspace_open", {"source": "fixture:induction_tf_broken"})["session_id"]
    tk2.call("edit_apply", {"session_id": s, "rationale": "restore", "operations": [
        {"op": "scale", "tensor": "blocks.1.attn.o_proj.weight", "head": h, "factor": 10.0} for h in range(4)]})
    r = tk2.call("evaluate_heldout", {"session_id": s, "suite_id": "induction_heldout", "limit": 16})
    assert r["metrics"]["score"] > 0.9


def test_fixture_bugfix_demo(tk):
    """M3 step-3 analogue on fixtures: an agent with a 50-pass budget, practice data only, fixes the bug."""
    rep = bugfix_agent.run(InProcessClient(tk))
    assert rep["fixed"] and rep["heldout_after"] > 0.9 and rep["forward_passes_used"] <= 50
