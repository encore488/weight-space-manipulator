import torch

from wsp_toolkit.core.util import short


def _ws(tk, sid):
    return tk.wsm.get(tk.sessions.get(sid)["workspace_id"])


def test_apply_revert_exact(tk, call, sid):
    base = call("workspace_info", session_id=sid)["working_copy_hash"]
    p = call("edit_apply", session_id=sid, rationale="test scale",
             operations=[{"op": "scale", "tensor": "blocks.1.attn.o_proj.weight", "head": 2, "factor": 0.5}])
    assert p["params_changed"] == 64 * 16
    assert call("workspace_info", session_id=sid)["working_copy_hash"] != base
    r = call("patch_revert", session_id=sid, patch_id=p["patch_id"])
    assert r["base_restored"] and r["model_hash"] == base
    ws = _ws(tk, sid)
    assert ws.model_hash() == ws.base_hash  # full hash, not just prefix


def test_replay_on_fresh_base_reproduces_hash(tk, call, sid):
    ops = [{"op": "project_out_direction", "tensor": "blocks.0.attn.o_proj.weight",
            "direction": {"tensor": "unembed.weight", "row": 5}, "side": "output", "indices": "0:16"},
           {"op": "zero", "tensor": "blocks.1.attn.q_proj.weight", "slice": "0:2, :"},
           {"op": "set_slice", "tensor": "embed.weight", "slice": "3, 0:4", "values": [[0.1, 0.2, 0.3, 0.4]]},
           {"op": "add_low_rank", "tensor": "blocks.1.attn.v_proj.weight",
            "u": [{"tensor": "embed.weight", "row": 1}], "v": [{"tensor": "embed.weight", "row": 2}], "coeffs": [0.05]}]
    p = call("edit_apply", session_id=sid, operations=ops, rationale="multi-op")
    ws = _ws(tk, sid)
    final = ws.model_hash()
    # fresh process view: drop the cache and rebuild from base + stack
    tk.wsm._cache.clear()
    ws2 = tk.wsm.get(ws.id)
    assert ws2.model_hash() == final
    # apply the stored delta onto a freshly loaded base model
    from wsp_toolkit.core.workspace.workspace import Workspace
    fresh = Workspace(ws.row, tk.wsm.fresh_base(ws.id))
    assert fresh.model_hash() == ws.base_hash
    tk.patch_io.apply_delta(fresh, p["patch_id"])
    assert fresh.model_hash() == final


def test_project_out_removes_direction(tk, call, sid):
    ws = _ws(tk, sid)
    d = ws.tensor("unembed.weight")[7].clone()
    d = d / d.norm()
    call("edit_apply", session_id=sid, rationale="remove token-7 direction from cols 16:32",
         operations=[{"op": "project_out_direction", "tensor": "blocks.1.attn.o_proj.weight",
                      "direction": {"tensor": "unembed.weight", "row": 7}, "indices": "16:32"}])
    W = ws.tensor("blocks.1.attn.o_proj.weight")
    assert torch.allclose(d @ W[:, 16:32], torch.zeros(16), atol=1e-5)
    assert (d @ W[:, :16]).abs().max() > 1e-3  # untouched columns keep their component


def test_preview_does_not_modify(call, sid):
    h = call("workspace_info", session_id=sid)["working_copy_hash"]
    pv = call("edit_preview", session_id=sid, operations=[{"op": "zero", "tensor": "embed.weight", "slice": "0:2, :"}])
    assert pv["params_changed"] == 128 and not pv["applied"]
    assert call("workspace_info", session_id=sid)["working_copy_hash"] == h


def test_stack_cascade_and_rebase(tk, call, sid):
    ws = _ws(tk, sid)
    mk = lambda name, f: call("edit_apply", session_id=sid, rationale=f"scale {name}",  # noqa: E731
                              operations=[{"op": "scale", "tensor": name, "factor": f}])
    a = mk("blocks.0.attn.q_proj.weight", 0.9)
    h_a = ws.model_hash()
    b = mk("blocks.0.attn.k_proj.weight", 0.9)
    c = mk("blocks.0.attn.v_proj.weight", 0.9)
    assert tk.patch_io.row(c["patch_id"])["parent_id"] == b["patch_id"]
    # cascade: reverting the middle patch also reverts everything above it
    r = call("patch_revert", session_id=sid, patch_id=b["patch_id"])
    assert r["stack"] == [a["patch_id"]] and r["also_reverted"] == [c["patch_id"]]
    assert ws.model_hash() == h_a
    # rebase: re-apply b and c, then revert b in rebase mode; c is re-derived on top of a
    call("patch_reapply", session_id=sid, patch_id=b["patch_id"])
    call("patch_reapply", session_id=sid, patch_id=c["patch_id"])
    r = call("patch_revert", session_id=sid, patch_id=b["patch_id"], mode="rebase")
    assert len(r["rebased"]) == 1 and r["rebased"][0]["old"] == c["patch_id"]
    new_c = r["rebased"][0]["new"]
    assert new_c != c["patch_id"] and r["stack"] == [a["patch_id"], new_c]
    assert tk.patch_io.row(c["patch_id"])["status"] == "superseded"
    # v_proj is scaled, k_proj untouched
    base = tk.wsm.fresh_base(ws.id).state()
    assert torch.equal(ws.tensor("blocks.0.attn.k_proj.weight"), base["blocks.0.attn.k_proj.weight"])
    assert not torch.equal(ws.tensor("blocks.0.attn.v_proj.weight"), base["blocks.0.attn.v_proj.weight"])


def test_patch_manifest_and_diff(call, sid):
    p1 = call("edit_apply", session_id=sid, rationale="first scale",
              operations=[{"op": "scale", "tensor": "embed.weight", "slice": "0:4, :", "factor": 2.0}])
    p2 = call("edit_apply", session_id=sid, rationale="second scale",
              operations=[{"op": "scale", "tensor": "embed.weight", "slice": "2:6, :", "factor": 2.0}])
    m = call("patch_show", patch_id=p1["patch_id"])
    for k in ("patch_id", "base_model_hash", "parent_patch", "operations", "tensors", "params_changed", "pct_of_model",
              "rationale", "author", "created_at", "status", "evals"):
        assert k in m, k
    assert m["tensors"][0]["hash_before"] != m["tensors"][0]["hash_after"]
    d = call("patch_diff", patch_a=p1["patch_id"], patch_b=p2["patch_id"])
    assert d["shared"][0]["overlapping_elements"] == 2 * 64
    lst = call("patch_list", session_id=sid)
    assert lst["applied_stack"] == [p1["patch_id"], p2["patch_id"]]


def test_edit_errors(call, sid):
    assert call("edit_apply", session_id=sid, rationale="unknown op", operations=[{"op": "explode", "tensor": "embed.weight"}])[
        "code"] == "bad_op"
    assert call("edit_apply", session_id=sid, rationale="", operations=[{"op": "zero", "tensor": "embed.weight"}])[
        "code"] == "bad_arg"
    r = call("edit_apply", session_id=sid, rationale="too big",
             operations=[{"op": "set_slice", "tensor": "embed.weight", "slice": ":, :", "values": 0.0}])
    assert r["code"] == "too_large"
    r = call("edit_apply", session_id=sid, rationale="noop", operations=[{"op": "scale", "tensor": "embed.weight",
                                                                          "factor": 1.0}])
    assert r["code"] == "no_change"
    r = call("edit_apply", session_id=sid, rationale="graft", operations=[{"op": "graft_subspace", "tensor": "x"}])
    assert r["code"] in ("not_implemented", "policy_violation")


def test_hash_ids_are_short_prefixes(call, sid):
    info = call("workspace_info", session_id=sid)
    assert len(info["base_model_hash"]) == 16 and short("abc" * 10) == ("abc" * 10)[:16]
