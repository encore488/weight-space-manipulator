"""Challenges and the API runner (driven by a scripted fake Anthropic client; no network, no API key)."""

from types import SimpleNamespace as NS

import pytest


def test_challenge_lifecycle_and_scoring(tk):
    lst = tk.call("challenge_list", {})["challenges"]
    assert {c["id"] for c in lst} >= {"induction-hunt", "fix-the-copier", "blind-surgeon", "mystery-finetune",
                                      "zoo-sprint"}
    assert tk.call("challenge_start", {"challenge_id": "induction-hunt", "agent_model": ""})["code"] == "bad_arg"
    r = tk.call("challenge_start", {"challenge_id": "induction-hunt", "agent_model": "m1"})
    sid = r["sessions"][0]["session_id"]
    assert tk.call("budget_status", {"session_id": sid})["access_level"] == "weights+activations"
    assert tk.call("run_forward", {"session_id": sid, "inputs": [[0, 1]]})["code"] == "access_denied"
    import json
    truth = json.loads((tk.cfg.subdir("fixtures") / "induction_tf" / "ground_truth.json").read_text())
    s = tk.call("challenge_submit", {"run_id": r["run_id"], "answer": {
        "induction_heads": truth["induction_heads"], "previous_token_heads": truth["previous_token_heads"]}})
    assert s["score"] == 1.0
    assert tk.call("challenge_submit", {"run_id": r["run_id"], "answer": {}})["code"] == "bad_state"
    board = tk.call("challenge_scores", {})["leaderboard"]
    assert board[0]["agent_model"] == "m1" and board[0]["harness"] == "mcp (self-reported)"


def test_blind_surgeon_scores_working_copy(tk):
    r = tk.call("challenge_start", {"challenge_id": "blind-surgeon", "agent_model": "m"})
    sid = r["sessions"][0]["session_id"]
    before = tk.challenges._copier_score(sid)[0]
    tk.call("edit_apply", {"session_id": sid, "rationale": "scale layer-1 head outputs back up", "operations": [
        {"op": "scale", "tensor": "blocks.1.attn.o_proj.weight", "factor": 10.0}]})
    s = tk.call("challenge_submit", {"run_id": r["run_id"], "answer": {}})
    assert before < 0.2 and s["score"] > 0.9 and s["usage"]["forward_passes"] == 0


def test_challenge_lock(tk):
    from wsp_toolkit.mcp_server.server import lock_to_challenge
    lock_to_challenge(tk, "fix-the-copier")
    assert tk.call("workspace_open", {"source": "fixture:induction_tf"})["code"] == "forbidden"
    r = tk.call("challenge_start", {"challenge_id": "fix-the-copier", "agent_model": "m"})
    assert r["locked"] is True


class FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return self.msg


class FakeClient:
    """Replays a scripted conversation; tool_use blocks may reference the run's session id lazily."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.beta = NS(messages=NS(stream=self.stream))

    def stream(self, **kw):
        self.calls.append(kw)
        step = self.script[len(self.calls) - 1]
        content = [b(kw) if callable(b) else b for b in step["content"]]
        usage = NS(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0, cache_read_input_tokens=500)
        return FakeStream(NS(content=content, stop_reason=step["stop"], usage=usage))


def _sid(kw):
    import json
    first = kw["messages"][0]["content"]
    return json.loads(first.split("\n", 1)[1])["sessions"][0]["session_id"]


def _run_id(kw):
    import json
    return json.loads(kw["messages"][0]["content"].split("\n", 1)[1])["run_id"]


def test_api_runner_with_fake_client(tk):
    from wsp_toolkit.agents.api_runner import run
    script = [
        {"stop": "tool_use", "content": [
            NS(type="text", text="Looking at head norms."),
            lambda kw: NS(type="tool_use", id="tu1", name="tensor_stats",
                          input={"session_id": _sid(kw), "tensor": "blocks.1.attn.o_proj.weight"}),
            NS(type="tool_use", id="tu2", name="tensor_stats", input={"tensor": 5})]},   # invalid input
        {"stop": "tool_use", "content": [
            lambda kw: NS(type="tool_use", id="tu3", name="edit_apply", input={
                "session_id": _sid(kw), "rationale": "restore layer-1 head outputs",
                "operations": [{"op": "scale", "tensor": "blocks.1.attn.o_proj.weight", "factor": 10.0}]})]},
        {"stop": "tool_use", "content": [
            lambda kw: NS(type="tool_use", id="tu4", name="challenge_submit",
                          input={"run_id": _run_id(kw), "answer": {}})]},
    ]
    fc = FakeClient(script)
    res = run(tk, "fix-the-copier", "claude-fable-5-2", client=fc, verbose=False)
    assert res["stop"] == "submitted" and res["score"] > 0.9 and not res["auto_submitted"]
    kw = fc.calls[0]
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"] == {"effort": "high"} and kw["cache_control"] == {"type": "ephemeral"}
    names = {t["name"] for t in kw["tools"]}
    assert "workspace_open" not in names and "challenge_submit" in names
    assert all(t["eager_input_streaming"] for t in kw["tools"])
    # the invalid tool input was answered with an error result, not executed
    second_user = fc.calls[1]["messages"][2]["content"]
    assert any(r.get("is_error") and "INVALID_JSON" in r["content"] for r in second_user)
    # history is append-only: each request extends the previous one
    assert fc.calls[2]["messages"][:3] == fc.calls[1]["messages"][:3]
    row = tk.challenges.runs("fix-the-copier")[0]
    assert row["harness"] == "wsp-agent-runner" and row["locked"] == 1
    assert row["usage"]["estimated_cost_usd"] > 0


def test_api_runner_refusal_and_limits(tk):
    from wsp_toolkit.agents.api_runner import run
    fc = FakeClient([{"stop": "refusal", "content": []}])
    res = run(tk, "induction-hunt", "claude-opus-5-5", client=fc, verbose=False)
    assert res["stop"] == "refusal" and res["auto_submitted"] and res["score"] == 0
    loop = [{"stop": "tool_use", "content": [lambda kw, i=i: NS(type="tool_use", id=f"t{i}", name="budget_status",
                                                            input={"session_id": _sid(kw)})]} for i in range(5)]
    res = run(tk, "induction-hunt", "claude-haiku-5-5", client=FakeClient(loop), max_turns=3, verbose=False)
    assert res["stop"] == "max_turns" and res["turns"] == 3


@pytest.mark.parametrize("cid", ["induction-hunt", "fix-the-copier"])
def test_challenge_prompt_cli(capsys, cid):
    from wsp_toolkit.agents.challenge_cli import main
    main(["prompt", cid, "--model", "claude-fable-5-2"])
    out = capsys.readouterr().out
    assert f"wsp mcp --challenge {cid}" in out and 'agent_model="claude-fable-5-2"' in out


def test_challenge_pages_render(tk):
    from fastapi.testclient import TestClient

    from wsp_toolkit.agents.challenge_baseline import run_all
    from wsp_toolkit.api.app import create_app
    run_all(tk, ["induction-hunt", "blind-surgeon"])
    c = TestClient(create_app(tk))
    for url in ["/", "/challenges", "/challenges/report.md"]:
        r = c.get(url)
        assert r.status_code == 200, url
    assert "scripted-baseline" in c.get("/challenges").text
    assert "| Find the induction heads |" in c.get("/challenges/report.md").text
