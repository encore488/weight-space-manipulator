"""The real Qwen3.5-0.8B architecture (hybrid linear/full attention, gated q_proj, zero-centred RMSNorm, tied
embeddings, multimodal wrapper) shrunk to a few hundred K random parameters, built offline from a config that is
the published config.json with smaller dimensions. Verifies the adapter and the paper's edit target names."""

import json
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen3_5")


@pytest.fixture(scope="module")
def qwen_dir(tmp_path_factory):
    import torch
    cfgd = json.loads((Path(__file__).parent / "data" / "qwen3_5_tiny_config.json").read_text())
    cfg = transformers.AutoConfig.for_model(**cfgd)
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("qwen35_tiny")
    transformers.AutoModelForImageTextToText.from_config(cfg).save_pretrained(d)
    return d


def test_qwen35_adapter_and_paper_edit_target(tk, qwen_dir):
    o = tk.call("workspace_open", {"source": str(qwen_dir)})
    assert o["model_kind"] == "hf_causal_lm", o
    sid = o["session_id"]
    names = [t["name"] for t in tk.call("tensor_list", {"session_id": sid, "filter": "down_proj"})["tensors"]]
    assert "model.language_model.layers.3.mlp.down_proj.weight" in names  # same naming as the released edit
    tl = tk.call("tensor_list", {"session_id": sid, "filter": "embed_tokens|lm_head"})["tensors"]
    assert any(t.get("tied_to") for t in tl)
    att = tk.call("probe_attention", {"session_id": sid, "inputs": [[1, 2, 3, 4, 5]]})
    assert {h["head"].split(".")[0] for h in att["heads"]} == {"3"}  # only the full-attention layer has patterns
    # Qwen loop-fix style edit: remove a direction from 8 output columns of two down_projs
    d = tk.call("direction_from_activations", {"session_id": sid, "positive_inputs": [[1, 2, 3, 1, 2, 3]],
                                               "negative_inputs": [[4, 5, 6, 7, 8, 9]], "hook": "L2.resid_post"})
    ops = [{"op": "project_out_direction", "tensor": f"model.language_model.layers.{i}.mlp.down_proj.weight",
            "direction": {"id": d["direction"]}, "side": "output", "indices": "0:8"} for i in (1, 3)]
    p = tk.call("edit_apply", {"session_id": sid, "rationale": "loop-fix analogue on the real architecture",
                               "operations": ops})
    assert p.get("params_changed") == 2 * 64 * 8, p
    assert tk.call("patch_revert", {"session_id": sid, "patch_id": p["patch_id"]})["base_restored"]
