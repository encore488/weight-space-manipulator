import pytest

from wsp_toolkit.core.features.index import induction_fixture_corpus


@pytest.fixture
def indexed(tk):
    ws = tk.wsm.open("fixture:induction_tf")
    rep = tk.features.build(ws, ["L0.resid_post", "L1.resid_post"], [16, 48], induction_fixture_corpus(n=240),
                            steps=600)
    sid = tk.call("workspace_open", {"source": "fixture:induction_tf"})["session_id"]
    return rep, sid


def test_index_built_with_zoom_and_fvu(tk, indexed):
    rep, sid = indexed
    assert len(rep["dictionaries"]) == 4
    for d in rep["dictionaries"]:
        assert 0 <= d["unexplained_fraction"] <= 1
    dicts = tk.call("feature_dictionaries", {"session_id": sid})["dictionaries"]
    fine = [d for d in dicts if d["size"] == 48]
    assert all(d["parent_dict"] for d in fine)  # zoom tree: fine -> coarse
    f = tk.call("feature_get", {"feature_id": f"{fine[0]['id']}:0"})
    assert {"label", "fidelity", "top_examples", "neurons", "zoom_parent", "readout"} <= set(f)
    assert "decoder_direction" not in f


def test_three_query_modes_report_fidelity_and_unexplained(tk, indexed):
    _, sid = indexed
    s = tk.call("feature_search", {"query": "current token", "k": 5, "session_id": sid})
    assert s["results"] and all("fidelity" in r and "unexplained_fraction" in r for r in s["results"])
    loc = tk.call("features_at_location", {"session_id": sid, "tensor": "blocks.0.attn.o_proj.weight",
                                           "index": ":, 0:16"})
    assert loc["results"] and all("fidelity" in r and "unexplained_fraction" in r for r in loc["results"])
    rd = tk.call("features_at_location", {"session_id": sid, "tensor": "unembed.weight", "index": "5, :"})
    assert rd["results"][0]["how"].startswith("selected weights read")
    act = tk.call("features_active_on", {"session_id": sid, "inputs": [[0, 5, 9, 3, 5, 9, 3]]})
    h = act["results"][0]["hooks"]["L1.resid_post"]
    assert "unexplained_fraction_this_input" in h and "dictionary_unexplained_fraction" in h
    assert all("fidelity" in f for p in h["positions"] for f in p["features"])
    assert act["budget"]["used"] == 1
    nb = tk.call("feature_neighbors", {"feature_id": loc["results"][0]["feature_id"]})
    assert "unexplained_fraction" in nb and "similar_in_dictionary" in nb
    nf = tk.call("neuron_features", {"session_id": sid, "hook": "L0.resid_post", "unit": 3})
    assert nf["features"] and "encoder_weight_from_unit" in nf["features"][0]


def test_weights_only_cannot_query_behaviour(tk, indexed):
    s = tk.call("workspace_open", {"source": "fixture:induction_tf", "access_level": "weights_only"})["session_id"]
    assert tk.call("features_active_on", {"session_id": s, "inputs": [[0, 1]]})["code"] == "access_denied"
    assert "results" in tk.call("features_at_location", {"session_id": s, "tensor": "blocks.1.attn.o_proj.weight"})


def test_sensitive_feature_flags_patches(tk, indexed):
    _, sid = indexed
    fid = tk.call("feature_search", {"query": "current token", "k": 1, "session_id": sid})["results"][0]["feature_id"]
    tk.call("sensitive_register", {"kind": "feature", "target": fid, "reason": "pretend safety feature"},
            actor="human")
    assert tk.call("feature_get", {"feature_id": fid})["sensitive"]
    did = tk.call("feature_get", {"feature_id": fid})["direction_id"]
    pv = tk.call("edit_preview", {"session_id": sid, "operations": [
        {"op": "project_out_direction", "tensor": "embed.weight", "direction": {"id": did}, "side": "input"}]})
    assert any(f["kind"] == "feature" for f in pv["sensitive_flags"])
