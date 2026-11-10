import json

import pytest

from wsp_toolkit.agents.clients import InProcessClient
from wsp_toolkit.agents.induction_finder import find_induction_heads
from wsp_toolkit.core.util import json_size


def test_workspace_open_hash_and_info(call, sid):
    info = call("workspace_info", session_id=sid)
    assert info["is_base"] and info["working_copy_hash"] == info["base_model_hash"]
    assert info["n_layers"] == 2 and info["is_language_model"]


def test_base_files_are_read_only(tk, sid):
    import os
    s = tk.sessions.get(sid)
    ws = tk.wsm.get(s["workspace_id"])
    f = os.path.join(ws.row["base_path"], "model.safetensors")
    assert not os.access(f, os.W_OK) or os.geteuid() == 0  # root ignores mode bits
    assert oct(os.stat(f).st_mode)[-3:] == "444"


def test_tensor_tools(call, sid):
    tl = call("tensor_list", session_id=sid)
    names = [t["name"] for t in tl["tensors"]]
    assert "blocks.1.attn.o_proj.weight" in names
    g = call("tensor_get", session_id=sid, tensor="blocks.0.attn.q_proj.weight", slice="0:2, 0:3")
    assert len(g["values"]) == 2 and len(g["values"][0]) == 3
    st = call("tensor_stats", session_id=sid, tensor="blocks.1.attn.o_proj.weight")
    assert len(st["head_norms"]) == 4
    sv = call("tensor_svd", session_id=sid, tensor="blocks.1.attn.o_proj.weight", k=3, head=1)
    assert len(sv["singular_values"]) == 3 and sv["components"][0]["left_direction"].startswith("dir_")
    ro = call("direction_readout", session_id=sid, direction={"id": sv["components"][0]["left_direction"]})
    assert len(ro["promotes"]) == 10
    cmp_ = call("tensor_compare", session_id=sid, tensor="blocks.0.attn.q_proj.weight",
                other_tensor="blocks.1.attn.q_proj.weight")
    assert -1 <= cmp_["cosine"] <= 1


def test_actionable_errors(call, sid):
    r = call("tensor_get", session_id=sid, tensor="blocks.0.attn.q_proj.weight", slice="0:100, 0:2")
    assert r["code"] == "slice_out_of_range" and "max" not in r["error"] or "0..63" in r["error"]
    r = call("tensor_get", session_id=sid, tensor="blocks.0.attn.q_proj.weigth")
    assert r["code"] == "unknown_tensor" and "blocks.0.attn.q_proj.weight" in r["hint"]
    r = call("tensor_get", session_id=sid, tensor="embed.weight")
    assert r["code"] == "too_large"


def test_weights_only_cannot_get_activations(call):
    s = call("workspace_open", source="fixture:induction_tf", access_level="weights_only")["session_id"]
    for tool, args in [("probe_activations", {"inputs": [[0, 1, 2]], "hooks": ["L0.resid_post"]}),
                       ("probe_attention", {"inputs": [[0, 1, 2]]}),
                       ("run_forward", {"inputs": [[0, 1, 2]]}),
                       ("direction_from_activations", {"positive_inputs": [[0, 1]], "negative_inputs": [[0, 2]],
                                                       "hook": "L0.resid_post"})]:
        r = call(tool, session_id=s, **args)
        assert r.get("code") == "access_denied", (tool, r)
    assert call("budget_status", session_id=s)["forward_passes"]["used"] == 0
    assert "values" in call("tensor_get", session_id=s, tensor="blocks.0.attn.q_proj.weight", slice="0:1, 0:1")


def test_activations_level_cannot_get_outputs(call):
    s = call("workspace_open", source="fixture:induction_tf", access_level="weights+activations")["session_id"]
    assert "heads" in call("probe_attention", session_id=s, inputs=[[0, 1, 2]])
    assert call("run_forward", session_id=s, inputs=[[0, 1, 2]])["code"] == "access_denied"
    # evaluation results are model outputs too: hidden without the 'outputs' capability
    assert call("suite_run_practice", session_id=s, suite_id="induction_practice", limit=2)["code"] == "access_denied"
    assert call("evaluate_heldout", session_id=s, suite_id="induction_heldout")["code"] == "access_denied"


def test_agent_cannot_raise_access_or_budget(tk, fixtures_dir, tmp_path):
    from conftest import make_toolkit
    tk2 = make_toolkit(tmp_path, fixtures_dir, session={"access_level": "weights_only", "forward_pass_budget": 5})
    r = tk2.call("workspace_open", {"source": "fixture:induction_tf", "access_level": "weights+activations+outputs"})
    assert r["code"] == "forbidden"
    r = tk2.call("workspace_open", {"source": "fixture:induction_tf", "budget": 100})
    assert r["code"] == "forbidden"


def test_budget_enforced_exactly(call):
    s = call("workspace_open", source="fixture:induction_tf", budget=3)["session_id"]
    assert call("run_forward", session_id=s, inputs=[[0, 1], [0, 2]])["budget"]["remaining"] == 1
    r = call("run_forward", session_id=s, inputs=[[0, 1], [0, 2]])
    assert r["code"] == "budget_exhausted"
    assert call("budget_status", session_id=s)["forward_passes"]["remaining"] == 1  # no partial spend
    assert call("probe_attention", session_id=s, inputs=[[0, 1]])["budget"]["remaining"] == 0
    assert call("run_forward", session_id=s, inputs=[[0]])["code"] == "budget_exhausted"


def test_no_tool_output_exceeds_cap(tk, call, sid):
    cap = tk.limits["max_response_bytes"]
    long_inputs = [[0] + list(range(1, 12)) + list(range(1, 12))] * 8
    outs = [
        call("probe_attention", session_id=sid, inputs=long_inputs, top_k_keys=8),
        call("probe_activations", session_id=sid, inputs=long_inputs, hooks=["L0.resid_post", "L1.resid_post", "L0.z"],
             positions="all", top_k=32),
        call("tensor_list", session_id=sid),
        call("tensor_svd", session_id=sid, tensor="embed.weight", k=32),
        call("tensor_stats", session_id=sid, tensor="embed.weight"),
        call("suite_list", session_id=sid),
        call("suite_run_practice", session_id=sid, suite_id="induction_practice"),
        call("session_log", session_id=sid),
    ]
    for o in outs:
        assert json_size(o) <= cap, (json_size(o), list(o)[:5])
    big = outs[0]
    assert big.get("truncated") and big["pages"] > 1
    for p in range(big["pages"]):
        pg = call("result_page", handle=big["handle"], page=p)
        assert json_size(pg) <= cap
    total = sum(len(call("result_page", handle=big["handle"], page=p)["items"]) for p in range(big["pages"]))
    assert total == big["total_items"]


def test_every_call_is_logged(tk, call, sid):
    n0 = len(tk.store.events(sid))
    call("tensor_list", session_id=sid)
    call("run_forward", session_id=sid, inputs=[[0, 1, 2]])
    evs = tk.store.events(sid)
    kinds = [(e["kind"], e["tool"]) for e in evs[n0:]]
    assert ("tool", "tensor_list") in kinds and ("forward", "run_forward") in kinds and ("tool", "run_forward") in kinds


def test_agent_locates_induction_heads(tk, fixtures_dir):
    """M1 acceptance: an agent using only the tools finds the induction / previous-token heads."""
    d, _ = fixtures_dir
    truth = json.loads((d / "induction_tf" / "ground_truth.json").read_text())
    rep = find_induction_heads(InProcessClient(tk))
    assert set(rep["induction_heads"]) == set(truth["induction_heads"])
    assert set(rep["previous_token_heads"]) == set(truth["previous_token_heads"])
    assert rep["forward_passes_used"] <= 10


def test_mlp_fixture_probe(call):
    s = call("workspace_open", source="fixture:mlp_parity3")["session_id"]
    r = call("run_forward", session_id=s, inputs=[[1, 0, 0], [1, 1, 0]])
    assert [x["argmax"] for x in r["results"]] == [1, 0]
    a = call("probe_activations", session_id=s, inputs=[[1, 0, 1]], hooks=["L0.hidden"])
    assert a["activations"][0]["units"] == 16
    assert call("direction_readout", session_id=s, direction={"tensor": "layers.0.weight", "row": 0})["code"] == \
        "unsupported"


@pytest.mark.parametrize("bad", [{"inputs": [[99]]}, {"inputs": []}, {"inputs": [list(range(30))]}])
def test_bad_inputs_rejected_without_charge(call, sid, bad):
    r = call("run_forward", session_id=sid, **bad)
    assert r["code"] in ("bad_input", "too_large")
    assert call("budget_status", session_id=sid)["forward_passes"]["used"] == 0
