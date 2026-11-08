

def _tested_patch(call, sid):
    p = call("edit_apply", session_id=sid, rationale="small scale",
             operations=[{"op": "scale", "tensor": "blocks.1.attn.o_proj.weight", "head": 0, "factor": 0.9}])
    assert p["regression"]["status"] == "tested"
    return p["patch_id"]


def test_export_blocked_without_approval(tk, call, sid):
    pid = _tested_patch(call, sid)
    assert call("patch_export", patch_id=pid)["code"] == "export_blocked"
    # agents cannot approve
    assert tk.call("patch_approve", {"patch_id": pid, "reviewer": "me"}, actor="agent")["code"] == "human_only"
    r = tk.call("patch_approve", {"patch_id": pid, "reviewer": "alice", "note": "lgtm"}, actor="human")
    assert r["status"] == "approved"
    ex = call("patch_export", patch_id=pid)
    assert "exported_to" in ex
    appr = tk.store.all("SELECT * FROM approvals WHERE patch_id=?", (pid,))
    assert appr[0]["reviewer"] == "alice" and appr[0]["ts"] > 0
    import json
    import pathlib
    man = json.loads((pathlib.Path(ex["exported_to"]) / "wsp_manifest.json").read_text())
    assert man["chain"] == [pid] and man["approvals"][0][0]["reviewer"] == "alice"


def test_export_needs_whole_chain_approved(tk, call, sid):
    a = _tested_patch(call, sid)
    b = call("edit_apply", session_id=sid, rationale="second",
             operations=[{"op": "scale", "tensor": "embed.weight", "slice": "0:1, :", "factor": 1.1}])["patch_id"]
    tk.call("patch_approve", {"patch_id": b, "reviewer": "alice"}, actor="human")
    assert call("patch_export", patch_id=b)["code"] == "export_blocked"
    tk.call("patch_approve", {"patch_id": a, "reviewer": "alice"}, actor="human")
    assert "exported_to" in call("patch_export", patch_id=b)


def test_approval_requires_tested(tk, fixtures_dir, tmp_path):
    from conftest import make_toolkit
    tk2 = make_toolkit(tmp_path, fixtures_dir, policy={"auto_regression_on_apply": False, "max_param_change_pct": 20})
    s = tk2.call("workspace_open", {"source": "fixture:induction_tf"})["session_id"]
    pid = tk2.call("edit_apply", {"session_id": s, "rationale": "zero a row", "operations": [
        {"op": "zero", "tensor": "embed.weight", "slice": "1, :"}]})["patch_id"]
    assert tk2.call("patch_approve", {"patch_id": pid, "reviewer": "a"}, actor="human")["code"] == "bad_state"
    tk2.call("patch_test", {"patch_id": pid})
    assert tk2.call("patch_approve", {"patch_id": pid, "reviewer": "a"}, actor="human")["status"] == "approved"


def test_policy_limits(tk, fixtures_dir, tmp_path):
    from conftest import make_toolkit
    tk2 = make_toolkit(tmp_path, fixtures_dir, policy={"max_param_change_pct": 1.0,
                                                       "allowed_primitives": ["scale", "zero"]})
    s = tk2.call("workspace_open", {"source": "fixture:induction_tf"})["session_id"]
    r = tk2.call("edit_apply", {"session_id": s, "rationale": "too many params", "operations": [
        {"op": "scale", "tensor": "blocks.1.attn.o_proj.weight", "factor": 2.0}]})
    assert r["code"] == "policy_violation" and "limit" in r["error"]
    r = tk2.call("edit_apply", {"session_id": s, "rationale": "disallowed primitive", "operations": [
        {"op": "set_slice", "tensor": "embed.weight", "slice": "0, 0:2", "values": [1, 2]}]})
    assert r["code"] == "policy_violation" and "not allowed" in r["error"]
    assert "patch_id" in tk2.call("edit_apply", {"session_id": s, "rationale": "zero four values", "operations": [
        {"op": "zero", "tensor": "embed.weight", "slice": "0, 0:4"}]})
    # session budget ceiling is policy too
    assert tk2.call("workspace_open", {"source": "fixture:induction_tf", "budget": 10**6})["code"] == "forbidden"


def test_sensitive_tensor_flagged_and_needs_ack(tk, call, sid):
    tk.call("sensitive_register", {"kind": "tensor", "target": "re:blocks\\.1\\.attn\\.o_proj", "reason": "test"},
            actor="human")
    assert call("sensitive_register", kind="tensor", target="x", reason="y")["code"] == "human_only"
    pv = call("edit_preview", session_id=sid, operations=[{"op": "scale", "tensor": "blocks.1.attn.o_proj.weight",
                                                           "head": 1, "factor": 0.5}])
    assert pv["sensitive_flags"]
    pid = _tested_patch(call, sid)
    assert tk.patch_io.row(pid)["flags"]
    assert tk.call("patch_approve", {"patch_id": pid, "reviewer": "a"}, actor="human")["code"] == "needs_ack"
    assert tk.call("patch_approve", {"patch_id": pid, "reviewer": "a", "acknowledge_flags": True},
                   actor="human")["status"] == "approved"


def test_sensitive_direction_flagged(tk, call, sid):
    d = call("direction_info", session_id=sid, direction={"tensor": "unembed.weight", "row": 9})["id"]
    tk.call("sensitive_register", {"kind": "direction", "target": d, "reason": "pretend refusal direction"},
            actor="human")
    pv = call("edit_preview", session_id=sid, operations=[{"op": "project_out_direction",
                                                           "tensor": "blocks.1.attn.o_proj.weight",
                                                           "direction": {"tensor": "unembed.weight", "row": 9}}])
    assert pv["sensitive_flags"] and pv["sensitive_flags"][0]["kind"] == "direction"
    pv2 = call("edit_preview", session_id=sid, operations=[{"op": "scale", "tensor": "blocks.0.attn.q_proj.weight",
                                                            "head": 0, "factor": 1.01}])
    assert not pv2["sensitive_flags"]


def test_never_open_own_weights(tk, fixtures_dir, tmp_path):
    from conftest import make_toolkit
    r = tk.call("workspace_open", {"source": "hf:acme/agent-model", "agent_model": "acme/agent-model"})
    assert r["code"] == "forbidden"
    tk2 = make_toolkit(tmp_path, fixtures_dir, policy={"forbidden_models": ["fixture:induction_tf"]})
    assert tk2.call("workspace_open", {"source": "fixture:induction_tf"})["code"] == "forbidden"


def test_mcp_exposes_no_human_only_tools(tk):
    from wsp_toolkit.agents.clients import MCPClient
    from wsp_toolkit.mcp_server.server import build_server
    with MCPClient(build_server(tk)) as c:
        names = {t.name for t in c.list_tools()}
    assert "patch_approve" not in names and "sensitive_register" not in names
    assert {"workspace_open", "edit_apply", "patch_export", "evaluate_heldout"} <= names


def test_ui_approval_and_flags_visible(tk, call, sid):
    from fastapi.testclient import TestClient

    from wsp_toolkit.api.app import create_app
    tk.call("sensitive_register", {"kind": "tensor", "target": "blocks.1.attn.o_proj.weight", "reason": "r"},
            actor="human")
    pid = _tested_patch(call, sid)
    c = TestClient(create_app(tk))
    page = c.get(f"/patches/{pid}").text
    assert "Sensitive components touched" in page
    assert c.post("/api/tools/patch_approve", json={"patch_id": pid, "reviewer": "x"}).status_code == 403
    r = c.post(f"/patches/{pid}/decide", data={"decision": "approve", "reviewer": "bob", "acknowledge_flags": "on"},
               follow_redirects=False)
    assert r.status_code == 303 and tk.patch_io.row(pid)["status"] == "approved"


def test_reviewer_token(tk, fixtures_dir, tmp_path, call):
    from conftest import make_toolkit
    from fastapi.testclient import TestClient

    from wsp_toolkit.api.app import create_app
    tk2 = make_toolkit(tmp_path, fixtures_dir, ui={"reviewer_token": "s3cret"}, policy={"max_param_change_pct": 20})
    s = tk2.call("workspace_open", {"source": "fixture:induction_tf"})["session_id"]
    pid = tk2.call("edit_apply", {"session_id": s, "rationale": "zero a row", "operations": [
        {"op": "zero", "tensor": "embed.weight", "slice": "1, 0:3"}]})["patch_id"]
    c = TestClient(create_app(tk2))
    assert c.post(f"/api/patches/{pid}/approve", json={"reviewer": "a"}).status_code == 403
    r = c.post(f"/api/patches/{pid}/approve", json={"reviewer": "a"}, headers={"X-WSP-Reviewer-Token": "s3cret"})
    assert r.json()["status"] == "approved"


def test_mcp_end_to_end_agent(tk, fixtures_dir):
    """The scripted induction finder works unchanged over a real MCP client session."""
    import json

    from wsp_toolkit.agents.clients import MCPClient, ToolError
    from wsp_toolkit.agents.induction_finder import find_induction_heads
    from wsp_toolkit.mcp_server.server import build_server
    d, _ = fixtures_dir
    truth = json.loads((d / "induction_tf" / "ground_truth.json").read_text())
    with MCPClient(build_server(tk)) as c:
        rep = find_induction_heads(c)
        try:
            c("tensor_get", session_id=rep["session_id"], tensor="nope")
            raise AssertionError("expected an error")
        except ToolError as e:
            assert e.payload["code"] == "unknown_tensor"
    assert set(rep["induction_heads"]) == set(truth["induction_heads"])
