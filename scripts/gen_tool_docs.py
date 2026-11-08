"""Generate docs/tools/<tool>.md for every tool: purpose, inputs (from the real signature), access & budget,
size limits, and an example call with its REAL (truncated) output, captured by running it against fixtures.

    python scripts/gen_tool_docs.py            # writes docs/tools/*.md and docs/tools/README.md
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsp_toolkit.config import Config  # noqa: E402
from wsp_toolkit.core.service import AGENT_TOOLS, HUMAN_ONLY, Toolkit  # noqa: E402

COST = {
    "run_forward": "needs `outputs`; 1 pass per input", "generate": "needs `outputs`; 1 pass per input",
    "probe_activations": "needs `activations`; 1 pass per input",
    "probe_attention": "needs `activations`; 1 pass per input",
    "direction_from_activations": "needs `activations`; 1 pass per input (positive + negative)",
    "suite_run_practice": "1 pass per example (× samples); per-example details need `outputs`",
    "evaluate_heldout": "no pass cost; rate-limited per session (session.heldout_rate_limit_per_hour); aggregate only",
    "run_baseline": "practice: 1 pass per example; held-out: counts toward the held-out rate limit",
    "features_active_on": "needs `activations`; 1 pass per input",
    "diff_report": "needs `outputs`; 2 passes per probe (both models)",
    "edit_apply": "no pass cost; runs regression suites automatically (system-run, not charged)",
    "patch_export": "blocked unless every patch in the chain is human-approved",
}
GROUP = {
    "M1 inspect & probe": ["workspace_open", "session_start", "workspace_info", "budget_status", "tensor_list",
                           "tensor_get", "tensor_stats", "tensor_svd", "tensor_compare", "direction_readout",
                           "direction_info", "direction_cosine", "run_forward", "generate", "probe_activations",
                           "direction_from_activations", "probe_attention", "result_page", "session_log"],
    "M2 edits & patches": ["edit_preview", "edit_apply", "patch_list", "patch_show", "patch_diff", "patch_revert",
                           "patch_reapply", "patch_export"],
    "M3 harness": ["suite_list", "suite_examples", "suite_run_practice", "evaluate_heldout", "run_baseline",
                   "compare_runs", "patch_test"],
    "M4 safety": ["sensitive_list", "patch_approve", "patch_reject", "sensitive_register"],
    "M5 zoo": ["zoo_list_tasks", "zoo_get_task", "zoo_submit", "zoo_scores", "zoo_build"],
    "M6 features": ["feature_search", "features_at_location", "neuron_features", "features_active_on", "feature_get",
                    "feature_neighbors", "feature_dictionaries"],
    "M7 audit": ["diff_models", "diff_report", "audit_get"],
}


def examples(tk, S, B, P, P2, RUN_A, RUN_B, FEAT, NET):
    o = {"op": "scale", "tensor": "blocks.1.attn.o_proj.weight", "head": 0, "factor": 0.9}
    return {
        "workspace_open": {"source": "fixture:induction_tf", "access_level": "weights+activations+outputs"},
        "session_start": {"workspace_id": tk.sessions.get(S)["workspace_id"], "budget": 10},
        "workspace_info": {"session_id": S}, "budget_status": {"session_id": S},
        "tensor_list": {"session_id": S, "filter": "blocks\\.1"},
        "tensor_get": {"session_id": S, "tensor": "blocks.1.attn.o_proj.weight", "slice": "0:2, 0:4"},
        "tensor_stats": {"session_id": S, "tensor": "blocks.1.attn.o_proj.weight"},
        "tensor_svd": {"session_id": S, "tensor": "blocks.1.attn.o_proj.weight", "head": 2, "k": 2},
        "tensor_compare": {"session_id": S, "tensor": "blocks.0.attn.q_proj.weight",
                           "other_tensor": "blocks.1.attn.q_proj.weight"},
        "direction_readout": {"session_id": S, "direction": {"tensor": "blocks.1.attn.o_proj.weight", "svd": 0,
                                                             "side": "left"}, "k": 5},
        "direction_info": {"session_id": S, "direction": {"token": "t5", "matrix": "unembed"}},
        "direction_cosine": {"session_id": S, "a": {"token": "t5"}, "b": {"token": "t6"}},
        "run_forward": {"session_id": S, "inputs": ["<bos> t3 t5 t7 t3 t5"], "top_k": 3},
        "generate": {"session_id": S, "inputs": [[0, 3, 5, 7, 3]], "settings": {"max_new_tokens": 4}},
        "probe_activations": {"session_id": S, "inputs": [[0, 3, 5, 7, 3, 5]], "hooks": ["L1.resid_post"], "top_k": 4},
        "direction_from_activations": {"session_id": S, "positive_inputs": [[0, 3, 5, 3, 5]],
                                       "negative_inputs": [[0, 3, 5, 9, 11]], "hook": "L1.resid_post"},
        "probe_attention": {"session_id": S, "inputs": [[0, 3, 5, 7, 3, 5]], "layers": [1], "heads": [2],
                            "top_k_keys": 2},
        "result_page": None, "session_log": {"session_id": S},
        "edit_preview": {"session_id": S, "operations": [o]},
        "edit_apply": {"session_id": S, "operations": [{**o, "factor": 0.95}],
                       "rationale": "example: shrink head 1.0 output by 5%"},
        "patch_list": {"session_id": S}, "patch_show": {"patch_id": P},
        "patch_diff": {"patch_a": P, "patch_b": P2},
        "patch_revert": {"session_id": S, "patch_id": P2}, "patch_reapply": {"session_id": S, "patch_id": P2},
        "patch_export": {"patch_id": P}, "patch_test": {"patch_id": P},
        "suite_list": {"session_id": S}, "suite_examples": {"session_id": S, "suite_id": "induction_practice"},
        "suite_run_practice": {"session_id": S, "suite_id": "induction_practice", "limit": 2},
        "evaluate_heldout": {"session_id": S, "suite_id": "induction_heldout", "limit": 8},
        "run_baseline": {"session_id": S, "suite_id": "induction_practice", "limit": 2, "settings": {"seed": 1}},
        "compare_runs": {"run_a": RUN_A, "run_b": RUN_B},
        "sensitive_list": {"session_id": S},
        "patch_approve": {"patch_id": P, "reviewer": "alice", "note": "looks fine"},
        "patch_reject": {"patch_id": P2, "reviewer": "alice", "note": "not needed"},
        "sensitive_register": {"kind": "tensor", "target": "unembed.weight", "reason": "example"},
        "zoo_list_tasks": {"family": "mlp_compiled"}, "zoo_get_task": {"net_id": NET, "task_type": "identify"},
        "zoo_submit": {"net_id": NET, "task_type": "identify", "answer": {"label": "and_01"}, "submitter": "docs"},
        "zoo_scores": {"submitter": "docs"}, "zoo_build": None,
        "feature_search": {"query": "previous token", "k": 3},
        "features_at_location": {"session_id": S, "tensor": "blocks.0.attn.o_proj.weight", "index": ":, 16:32", "k": 2},
        "neuron_features": {"session_id": S, "hook": "L0.resid_post", "unit": 3, "k": 2},
        "features_active_on": {"session_id": S, "inputs": [[0, 5, 9, 3, 5, 9]], "hook": "L1.resid_post", "k": 1},
        "feature_get": {"feature_id": FEAT}, "feature_neighbors": {"feature_id": FEAT, "k": 2},
        "feature_dictionaries": {"session_id": S},
        "diff_models": {"session_id": S, "other_session_id": B},
        "diff_report": {"session_id": S, "other_session_id": B, "n_probes": 8}, "audit_get": None,
    }


def trunc(obj, n=1800):
    s = json.dumps(obj, indent=1)
    return s if len(s) <= n else s[:n] + "\n  ... (truncated for the docs)"


def main():
    tmp = Path(tempfile.mkdtemp())
    tk = Toolkit(Config.load(overrides={"paths": {"data_dir": str(tmp)}, "policy": {"max_param_change_pct": 20.0},
                                        "session": {"forward_pass_budget": 200}}))
    call = lambda n, a, actor="agent": tk.call(n, a, actor=actor)  # noqa: E731
    S = call("workspace_open", {"source": "fixture:induction_tf"})["session_id"]
    B = call("workspace_open", {"source": "fixture:induction_tf_broken"})["session_id"]
    P = call("edit_apply", {"session_id": S, "rationale": "doc patch A", "operations": [
        {"op": "scale", "tensor": "blocks.1.attn.o_proj.weight", "head": 1, "factor": 0.9}]})["patch_id"]
    P2 = call("edit_apply", {"session_id": S, "rationale": "doc patch B", "operations": [
        {"op": "scale", "tensor": "blocks.1.attn.o_proj.weight", "head": 1, "factor": 0.8}]})["patch_id"]
    RUN_A = call("suite_run_practice", {"session_id": B, "suite_id": "induction_practice", "limit": 4})["run_id"]
    RUN_B = call("suite_run_practice", {"session_id": S, "suite_id": "induction_practice", "limit": 4})["run_id"]
    tk.call("zoo_build", {"scale": "mini"}, actor="human")
    NET = call("zoo_list_tasks", {"family": "mlp_compiled"})["items"][0]["net_id"]
    from wsp_toolkit.core.features.index import induction_fixture_corpus
    tk.features.build(tk.wsm.get(tk.sessions.get(S)["workspace_id"]), ["L0.resid_post", "L1.resid_post"], [32],
                      induction_fixture_corpus(240), steps=600)
    FEAT = call("feature_search", {"query": "previous token", "k": 1})["results"][0]["feature_id"]
    ex = examples(tk, S, B, P, P2, RUN_A, RUN_B, FEAT, NET)
    out_dir = ROOT / "docs" / "tools"
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = tk.limits["max_response_bytes"]
    index = ["# Tool reference", "", f"Every tool returns JSON capped at `limits.max_response_bytes` (default {cap} "
             "bytes); bigger results come back as `{truncated, handle, pages, preview}` — fetch the rest with "
             "[`result_page`](result_page.md). Errors are `{error, code, hint?}` (over MCP they set `isError`).", ""]
    for group, names in GROUP.items():
        index += [f"## {group}", "", "| Tool | Purpose | Who |", "|---|---|---|"]
        for name in names:
            fn = tk._tools[name]
            hints = typing.get_type_hints(fn)
            doc = getattr(fn, "_wsp_doc", "") or AGENT_TOOLS.get(name, "")
            who = "human only (UI / HTTP API / CLI)" if name in HUMAN_ONLY else "agents (MCP) and humans"
            index.append(f"| [`{name}`]({name}.md) | {doc} | {who} |")
            L = [f"# `{name}`", "", doc, "", f"**Who can call it:** {who}.", ""]
            if name in COST:
                L += [f"**Access / budget:** {COST[name]}.", ""]
            L += ["## Inputs", "", "| Name | Type | Default |", "|---|---|---|"]
            for p in inspect.signature(fn).parameters.values():
                if p.name in ("self", "actor"):
                    continue
                t = hints.get(p.name, typing.Any)
                tn = getattr(t, "__name__", None) or str(t).replace("typing.", "")
                dflt = "required" if p.default is inspect.Parameter.empty else f"`{json.dumps(p.default)}`"
                L.append(f"| `{p.name}` | `{tn}` | {dflt} |")
            L += ["", "## Size limits", "", f"Response capped at {cap} bytes (paged via `result_page` if larger)."]
            if name in ("tensor_get", "edit_apply", "edit_preview"):
                L.append(f"Raw value slices are capped at `limits.max_slice_elements` = "
                         f"{tk.limits['max_slice_elements']} elements.")
            if name in ("run_forward", "probe_activations", "probe_attention", "generate", "features_active_on"):
                L.append(f"At most `limits.max_batch_inputs` = {tk.limits['max_batch_inputs']} inputs per call and "
                         f"`limits.max_seq_len` = {tk.limits['max_seq_len']} tokens per input.")
            args = ex.get(name)
            if name == "result_page":
                big = call("probe_attention", {"session_id": S, "inputs": [[0] + list(range(1, 12)) * 2] * 6})
                args = {"handle": big.get("handle", "res_..."), "page": 1}
            if name == "audit_get":
                aid = call("diff_report", {"session_id": S, "other_session_id": B, "n_probes": 4})["audit_id"]
                args = {"audit_id": aid}
            L += ["", "## Example", ""]
            if args is None:
                L += ["```bash", f"wsp zoo build --scale full   # human-only; via HTTP: POST /api/tools/{name} is refused",
                      "```"]
            else:
                actor = "human" if name in HUMAN_ONLY else "agent"
                res = call(name, args, actor)
                L += ["Call (MCP tool arguments, or `wsp call " + name + " '<json>'`):", "", "```json",
                      json.dumps(args, indent=1), "```", "", "Result (real output from the induction fixture):", "",
                      "```json", trunc(res), "```"]
            (out_dir / f"{name}.md").write_text("\n".join(L) + "\n")
        index.append("")
    index += ["## Direction specs", "", "Many tools take a `direction`: see "
              "[`wsp_toolkit/core/directions.py`](../../wsp_toolkit/core/directions.py) — `{\"id\": \"dir_…\"}`, "
              "`{\"tensor\", \"row\"|\"col\"|\"svd\"}`, `{\"token\", \"matrix\"}`, `{\"vector\"}`, `{\"combine\"}`.", "",
              "## Edit operations", "", "`{\"op\": \"project_out_direction\", \"tensor\", \"direction\", "
              "\"side\": \"output\"|\"input\", \"indices\"?, \"strength\"?}` · `{\"op\": \"scale\", \"tensor\", "
              "\"head\"|\"slice\"?, \"factor\"}` · `{\"op\": \"zero\", ...}` · `{\"op\": \"set_slice\", \"tensor\", "
              "\"slice\", \"values\"}` · `{\"op\": \"add_low_rank\", \"tensor\", \"u\": [dir], \"v\": [dir], "
              "\"coeffs\"?}` or explicit `U`/`V` · `graft_subspace` (stub)."]
    (out_dir / "README.md").write_text("\n".join(index) + "\n")
    print(f"wrote {len(list(out_dir.glob('*.md')))} files to {out_dir}")


if __name__ == "__main__":
    main()
