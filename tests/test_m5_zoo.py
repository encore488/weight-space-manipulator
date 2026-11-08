import json
from pathlib import Path


def test_zoo_mini_build_list_submit(tk):
    r = tk.call("zoo_build", {"scale": "mini"}, actor="human")
    assert r["networks"] >= 10
    assert tk.call("zoo_build", {"scale": "mini"}, actor="agent")["code"] == "human_only"
    lst = tk.call("zoo_list_tasks", {})
    assert lst["total"] == r["networks"]
    xor_ids = {tk.zoo.id_for(f"mlpb-xor_01-s0-{f}") for f in ("silenced_input", "shifted_threshold", "backdoor")}
    net = next(i for i in lst["items"] if "repair" in i["task_types"] and i["net_id"] in xor_ids)
    assert all("xor" not in i["net_id"] and "backdoor" not in i["net_id"] for i in lst["items"])  # opaque ids
    t = tk.call("zoo_get_task", {"net_id": net["net_id"], "task_type": "identify"})
    assert "answers" not in json.dumps(t) and "flaw\": \"" not in json.dumps(t["card"])
    assert t["open_with"]["source"] == f"zoo:{net['net_id']}"
    s = tk.call("workspace_open", {"source": t["open_with"]["source"]})
    assert s["model_kind"] == "tiny_mlp"
    # no tool reveals held-out examples or answers
    assert tk.call("suite_examples", {"session_id": s["session_id"], "suite_id": "zoo_heldout"})["code"] == \
        "access_denied"
    res = tk.call("zoo_submit", {"net_id": net["net_id"], "task_type": "identify", "answer": {"label": "xor_01"},
                                 "submitter": "t"})
    if res["split"] == "public":
        assert res["score"] == 1.0
    else:
        assert "score" not in res
    bad = tk.call("zoo_submit", {"net_id": net["net_id"], "task_type": "localize", "answer": {"flaw": "gremlins"}})
    assert bad["code"] == "bad_answer"


def test_symmetry_copies_preserve_function_but_change_weights(tk):

    import torch

    from wsp_toolkit.core.workspace.adapters import load_path
    tk.call("zoo_build", {"scale": "mini"}, actor="human")
    rows = []
    for r in tk.store.all("SELECT id, path FROM zoo_nets"):
        a = json.loads((Path(r["path"]) / "answers.json").read_text())
        if a.get("symmetry_of"):
            rows.append({**r, "orig": tk.zoo.id_for(a["name"][:-4])})
    assert rows
    for r in rows[:4]:
        orig = tk.store.one("SELECT path FROM zoo_nets WHERE id=?", (r["orig"],))
        a, b = load_path(Path(orig["path"])), load_path(Path(r["path"]))
        x = torch.rand(64, a.config["d_in"]).round()
        assert torch.allclose(a.forward(x), b.forward(x), atol=1e-4)
        assert not torch.equal(a.state()["layers.0.weight"], b.state()["layers.0.weight"])


def test_zoo_baseline_agent_runs_all_task_types(tk, fixtures_dir, tmp_path):
    from conftest import make_toolkit

    from wsp_toolkit.agents.clients import InProcessClient
    from wsp_toolkit.agents.zoo_baseline import run_baseline
    tk2 = make_toolkit(tmp_path, fixtures_dir, session={"forward_pass_budget": 120})
    tk2.call("zoo_build", {"scale": "mini"}, actor="human")
    rep = run_baseline(InProcessClient(tk2))
    assert all(v > 0 for v in rep["submitted"].values())
    types = {r["task_type"] for r in rep["scores"]["rows"]}
    assert types == {"identify", "localize", "repair"}


def test_repair_scoring(tk):
    tk.call("zoo_build", {"scale": "mini"}, actor="human")
    nid = tk.zoo.id_for("mlpb-xor_01-s0-backdoor")
    good = tk.zoo.submit(nid, "repair", {"operations": [{"op": "zero", "tensor": "layers.0.weight",
                                                          "slice": ":, 4:8"}]}, "t", None)
    none = tk.zoo.submit(nid, "repair", {"operations": []}, "t", None)
    if good["split"] == "public":
        assert good["score"] > 0.9 and none["score"] == 0.0
