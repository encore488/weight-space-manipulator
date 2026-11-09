"""Hugging Face path, fully offline: a tiny randomly-initialised Llama + a word-level tokenizer built locally."""

import pytest

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

WORDS = ["<unk>", "<eos>", "the", "cat", "dog", "sat", "on", "mat", "a", "red", "blue", "big", "small", "ran", "and",
         "user", "assistant", ":"] + [f"w{i}" for i in range(46)]


@pytest.fixture(scope="module")
def hf_dir(tmp_path_factory):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny_llama")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      eos_token_id=1, bos_token_id=1, pad_token_id=1, tie_word_embeddings=False)
    LlamaForCausalLM(cfg).save_pretrained(d)
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")
    fast.chat_template = "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}assistant :"
    fast.save_pretrained(d)
    return d


def test_hf_open_inspect_probe_edit(tk, hf_dir):
    c = lambda tool, **a: tk.call(tool, a)  # noqa: E731
    o = c("workspace_open", source=str(hf_dir))
    assert o["model_kind"] == "hf_causal_lm", o
    sid = o["session_id"]
    assert o["n_layers"] == 2 and "L1.resid_post" in o["hook_points"] and "L0.mlp_hidden" in o["hook_points"]
    tl = c("tensor_list", session_id=sid, filter="layers.0.self_attn")
    heads = {t["name"].split(".")[-2]: t["heads"] for t in tl["tensors"] if "heads" in t}
    assert heads["q_proj"]["n_heads"] == 4 and heads["k_proj"]["n_heads"] == 2  # GQA
    r = c("run_forward", session_id=sid, inputs=["the cat sat on the mat"])
    assert r["results"][0]["tokens"][:2] == ["the", "cat"]
    a = c("probe_attention", session_id=sid, inputs=["the cat sat on the mat"], layers=[1])
    assert len(a["heads"]) == 4
    act = c("probe_activations", session_id=sid, inputs=["the dog ran"], hooks=["L0.mlp_hidden", "L1.resid_post"])
    assert {x["hook"] for x in act["activations"]} == {"L0.mlp_hidden", "L1.resid_post"}
    ro = c("direction_readout", session_id=sid, direction={"tensor": "model.layers.1.mlp.down_proj.weight", "col": 3})
    assert ro["through_final_norm_scale"] and len(ro["promotes"]) == 10
    g = c("generate", session_id=sid, inputs=["the cat"], settings={"max_new_tokens": 5, "presence_penalty": 1.5})
    assert g["results"][0]["new_tokens"] <= 5
    # the Qwen-style edit: project a direction out of 8 output columns of a down_proj
    d = c("direction_from_activations", session_id=sid, positive_inputs=["the cat sat", "a red cat"],
          negative_inputs=["the dog ran", "a blue dog"], hook="L1.resid_post")["direction"]
    p = c("edit_apply", session_id=sid, rationale="project the cat-vs-dog direction out of 8 down_proj columns",
          operations=[{"op": "project_out_direction", "tensor": "model.layers.1.mlp.down_proj.weight",
                       "direction": {"id": d}, "side": "output", "indices": "0:8"}])
    assert p["params_changed"] <= 32 * 8
    assert c("patch_revert", session_id=sid, patch_id=p["patch_id"])["base_restored"]


def test_batched_generation_matches_unbatched_greedy(tk, hf_dir):
    from pathlib import Path

    from wsp_toolkit.core.evals.scorers import GenerationSettings, score_examples
    from wsp_toolkit.core.workspace.adapters import load_path
    ad = load_path(Path(hf_dir))
    prompts = ["the cat", "a big red dog sat on the mat", "the dog ran and"]
    encs = [ad.encode(p) for p in prompts]
    settings = {"max_new_tokens": 6, "temperature": 0.0, "presence_penalty": 1.5}
    one = [ad.generate(e, settings)["new_ids"] for e in encs]
    batch = [g["new_ids"] for g in ad.generate_batch(encs, settings)]
    assert one == batch
    ex = [{"prompt": p, "target": "cat"} for p in prompts]
    s = GenerationSettings.from_dict({"max_new_tokens": 6})
    a = score_examples(ad, "gen_contains", ex, s, batch_size=1)
    b = score_examples(ad, "gen_contains", ex, s, batch_size=3)
    assert [x["score"] for x in a] == [x["score"] for x in b]
