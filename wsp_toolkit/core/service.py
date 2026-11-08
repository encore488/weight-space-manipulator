"""The Toolkit service: every agent- or human-facing operation lives here.

The MCP server and the HTTP API are thin wrappers that call ``Toolkit.call(tool, args, actor=...)``.
``call`` does the cross-cutting work in one place:
  - looks up the session and enforces access level / budget (inside the individual tool),
  - converts exceptions into actionable error dicts,
  - caps the response size (large results become a summary + a pageable handle),
  - appends an event to the session log.
"""

from __future__ import annotations

import json
import math
import shutil
import traceback
from collections.abc import Callable
from typing import Any

import torch

from ..config import Config
from .db import Store
from .directions import DirectionStore, resolve
from .evals.harness import EvalHarness
from .inspect import tensors as T
from .patch.patches import PatchIO, PatchManager
from .probe import probe as P
from .safety.policy import PolicyEngine
from .util import WSPError, canonical_json, json_size, now, rnd, short
from .workspace.workspace import SessionManager, WorkspaceManager

AGENT_TOOLS: dict[str, str] = {}   # name -> one-line description (filled by @tool)
HUMAN_ONLY = {"patch_approve", "patch_reject", "sensitive_register", "zoo_build"}


def tool(name: str, doc: str, human_only: bool = False):
    def deco(fn: Callable):
        fn._wsp_tool = name
        fn._wsp_doc = doc
        if not human_only:
            AGENT_TOOLS[name] = doc
        return fn
    return deco


class Toolkit:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.load()
        self.store = Store(self.cfg.data_dir / "wsp.sqlite")
        self.dirs = DirectionStore(self.store)
        self.wsm = WorkspaceManager(self.store, self.cfg)
        self.patch_io = PatchIO(self.store, self.cfg.subdir("patches"))
        self.wsm._patch_io = self.patch_io
        self.sessions = SessionManager(self.store, self.cfg)
        self.policy = PolicyEngine(self.cfg, self.store, self.dirs)
        self.limits = dict(self.cfg.raw["limits"])
        self.patches = PatchManager(self.store, self.patch_io, self.wsm, self.dirs, self.policy, self.limits)
        self.harness = EvalHarness(self.store, self.cfg, self.wsm)
        self.sig = int(self.limits["float_sig_digits"])
        self._tools = {}
        for attr in dir(self):
            fn = getattr(self, attr, None)
            if callable(fn) and hasattr(fn, "_wsp_tool"):
                self._tools[fn._wsp_tool] = fn
        from . import extras  # zoo / features / diff tools register themselves
        extras.register(self)

    # ================================================================== dispatch
    def call(self, name: str, args: dict | None = None, actor: str = "agent") -> dict:
        args = dict(args or {})
        sid = args.get("session_id")
        fn = self._tools.get(name)
        try:
            if fn is None:
                raise WSPError(f"unknown tool {name!r}", code="unknown_tool", hint=f"tools: {sorted(self._tools)}")
            if name in HUMAN_ONLY and actor != "human":
                raise WSPError(f"{name} is reserved for human reviewers (UI / HTTP API)", code="human_only")
            result = fn(actor=actor, **args)
            ok = True
        except WSPError as e:
            result, ok = e.to_dict(), False
        except TypeError as e:
            result, ok = {"error": f"bad arguments for {name}: {e}", "code": "bad_args"}, False
        except Exception as e:  # noqa: BLE001 - never crash the server; report actionably
            result = {"error": f"{type(e).__name__}: {e}", "code": "internal_error",
                      "trace_tail": traceback.format_exc().splitlines()[-3:]}
            ok = False
        result = self._cap(result)
        log_sid = sid or (result.get("session_id") if isinstance(result, dict) else None)
        self.store.log_event(session_id=log_sid,
                             actor=actor, kind="tool", tool=name, args=_loggable(args),
                             result_summary=_summarise(result), result=result, ok=ok)
        return result

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    # ================================================================== size capping
    def _cap(self, result: Any) -> Any:
        cap = int(self.limits["max_response_bytes"])
        if json_size(result) <= cap:
            return result
        handle = self.store.put_result(result)
        path, items = _largest_list(result, cap)
        if path is not None:
            per_page = max(1, len(items))
            while per_page > 1 and json_size(items[:per_page]) > cap * 0.6:
                per_page = max(1, per_page // 2)
            shrunk = _replace_at(result, path, items[:per_page])
            preview = shrunk if json_size(shrunk) <= cap * 0.8 else {"note": "preview omitted (too large)"}
            return {"truncated": True, "handle": handle, "paged_field": ".".join(map(str, path)),
                    "total_items": len(items), "items_per_page": per_page,
                    "pages": math.ceil(len(items) / per_page), "page": 0, "preview": preview,
                    "hint": "call result_page(handle, page) for more"}
        text = canonical_json(result)
        chunk = int(cap * 0.7)
        return {"truncated": True, "handle": handle, "pages": math.ceil(len(text) / chunk), "page": 0,
                "text_chunk": text[:chunk], "hint": "call result_page(handle, page) for more JSON text"}

    @tool("result_page", "Fetch page N of a large result that was returned as a handle.")
    def result_page(self, handle: str, page: int = 0, actor: str = "agent", session_id: str | None = None) -> dict:
        full = self.store.get_result(handle)
        if full is None:
            raise WSPError(f"unknown handle {handle!r}", code="not_found")
        cap = int(self.limits["max_response_bytes"])
        path, items = _largest_list(full, cap)
        if path is not None:
            per_page = max(1, len(items))
            while per_page > 1 and json_size(items[:per_page]) > cap * 0.6:
                per_page = max(1, per_page // 2)
            pages = math.ceil(len(items) / per_page)
            if not 0 <= page < pages:
                raise WSPError(f"page {page} out of range 0..{pages - 1}", code="bad_arg")
            return {"handle": handle, "page": page, "pages": pages, "paged_field": ".".join(map(str, path)),
                    "items": items[page * per_page:(page + 1) * per_page]}
        text = canonical_json(full)
        chunk = int(cap * 0.7)
        pages = math.ceil(len(text) / chunk)
        if not 0 <= page < pages:
            raise WSPError(f"page {page} out of range 0..{pages - 1}", code="bad_arg")
        return {"handle": handle, "page": page, "pages": pages, "text_chunk": text[page * chunk:(page + 1) * chunk]}

    # ================================================================== helpers
    def _session(self, session_id: str) -> tuple[dict, Any]:
        s = self.sessions.get(session_id)
        if s["closed_at"]:
            raise WSPError(f"session {session_id} is closed", code="closed")
        return s, self.wsm.get(s["workspace_id"])

    def _charge(self, s: dict, n_inputs: int, what: str) -> dict:
        """Charge the budget for running n_inputs inputs (see session.inputs_per_pass)."""
        per = max(1, int(self.cfg.get("session.inputs_per_pass", 1)))
        n = math.ceil(n_inputs / per) if n_inputs > 0 else 0
        s2 = self.sessions.charge(s["id"], n)
        self.store.log_event(session_id=s["id"], actor="system", kind="forward", tool=what,
                             args={"passes": n, "inputs": n_inputs}, result_summary=f"{n} forward pass(es) for "
                             f"{n_inputs} input(s); {s2['budget_remaining']} remaining")
        return s2

    def _budget(self, s: dict) -> dict:
        s = self.sessions.get(s["id"])
        return {"used": s["budget_used"], "total": s["budget_total"], "remaining": s["budget_remaining"]}

    # ================================================================== M1: workspaces
    @tool("workspace_open", "Open a model (fixture:<name>, zoo:<id>, hf:<repo>, or a local path) into a new "
          "workspace with a working copy, and start a budgeted session on it.")
    def workspace_open(self, source: str, label: str | None = None, access_level: str | None = None,
                       budget: int | None = None, confirm_download: bool = False, agent_model: str | None = None,
                       actor: str = "agent") -> dict:
        if agent_model and agent_model in (self.cfg.get("policy.forbidden_models") or []):
            raise WSPError("refusing: an agent may never open its own weights", code="forbidden")
        if agent_model and source.removeprefix("hf:").lower() == agent_model.lower():
            raise WSPError("refusing: an agent may never open its own weights for editing (hard rule)",
                           code="forbidden")
        ws = self.wsm.open(source, label, confirm_download)
        s = self.sessions.create(ws.id, actor=actor, access_level=access_level, budget=budget, label=label,
                                 agent_model=agent_model)
        return {"workspace_id": ws.id, "session_id": s["id"], "access_level": s["access_level"],
                "capabilities": s["capabilities"], "budget": self._budget(s), **self._model_summary(ws)}

    @tool("session_start", "Start a new budgeted session on an existing workspace.")
    def session_start(self, workspace_id: str, access_level: str | None = None, budget: int | None = None,
                      label: str | None = None, agent_model: str | None = None, actor: str = "agent") -> dict:
        ws = self.wsm.get(workspace_id)
        s = self.sessions.create(ws.id, actor=actor, access_level=access_level, budget=budget, label=label,
                                 agent_model=agent_model)
        return {"workspace_id": ws.id, "session_id": s["id"], "access_level": s["access_level"],
                "capabilities": s["capabilities"], "budget": self._budget(s)}

    def _model_summary(self, ws) -> dict:
        ad = ws.adapter
        info = {"model_kind": ad.kind, "base_model_hash": short(ws.base_hash), "n_params": ws.n_params(),
                "n_tensors": len(ws.tensors()), "n_layers": ad.n_layers(), "is_language_model": ad.is_lm,
                "hook_points": [h.alias for h in ad.hook_specs()][:80]}
        if ad.is_lm:
            info["vocab_size"] = ad.vocab_size()
        cfg = {k: v for k, v in ad.config.items() if k not in ("token_names",)}
        if "token_names" in ad.config:
            info["token_names_sample"] = ad.config["token_names"][:8]
        info["config"] = cfg
        return info

    @tool("workspace_info", "Describe the session's model, working-copy state (applied patches, hash) and budget.")
    def workspace_info(self, session_id: str, actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        return {"workspace_id": ws.id, "source": ws.row["source"], "access_level": s["access_level"],
                "capabilities": s["capabilities"], "budget": self._budget(s),
                "applied_patches": list(ws.stack), "working_copy_hash": short(ws.model_hash()),
                "is_base": not ws.stack, **self._model_summary(ws)}

    @tool("budget_status", "Remaining forward-pass budget and held-out evaluation allowance for the session.")
    def budget_status(self, session_id: str, actor: str = "agent") -> dict:
        s = self.sessions.get(session_id)
        lim = int(self.cfg.get("session.heldout_rate_limit_per_hour"))
        used = self.harness.heldout_count_last_hour(session_id)
        return {"session_id": session_id, "forward_passes": self._budget(s),
                "heldout_evals_remaining_this_hour": max(0, lim - used), "access_level": s["access_level"]}

    # ================================================================== M1: tensors
    @tool("tensor_list", "List tensors with shapes, dtypes, layer grouping and head layout. Paged; regex filter.")
    def tensor_list(self, session_id: str, filter: str | None = None, offset: int = 0, actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        return T.tensor_list(ws, filter, offset, int(self.limits["page_size"]))

    @tool("tensor_get", "Raw values of a small tensor slice (numpy-style 'a:b, c:d'; optional head). Size-capped.")
    def tensor_get(self, session_id: str, tensor: str, slice: str | None = None, head: int | None = None,
                   actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        return T.tensor_get(ws, tensor, slice, int(self.limits["max_slice_elements"]), self.sig, head)

    @tool("tensor_stats", "Summary statistics of a tensor or slice: norms, distribution, sparsity, row/col/head norms.")
    def tensor_stats(self, session_id: str, tensor: str, slice: str | None = None, head: int | None = None,
                     actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        return T.tensor_stats(ws, tensor, slice, self.sig, head)

    @tool("tensor_svd", "Top-k singular values/vectors of a 2-D tensor, slice or head. Each singular vector is "
          "stored as a reusable direction id.")
    def tensor_svd(self, session_id: str, tensor: str, k: int = 4, slice: str | None = None, head: int | None = None,
                   actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        return T.tensor_svd(ws, tensor, k, slice, self.sig, self.dirs, head)

    @tool("tensor_compare", "Compare two tensors (same or another session's workspace, e.g. base vs patched).")
    def tensor_compare(self, session_id: str, tensor: str, other_tensor: str | None = None,
                       other_session_id: str | None = None, slice: str | None = None, against_base: bool = False,
                       actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        if against_base:
            from .workspace.workspace import Workspace
            base_ws = Workspace(ws.row, self.wsm.fresh_base(ws.id))
            return T.tensor_compare(base_ws, tensor, ws, other_tensor or tensor, slice, self.sig)
        ws_b = self._session(other_session_id)[1] if other_session_id else ws
        return T.tensor_compare(ws, tensor, ws_b, other_tensor or tensor, slice, self.sig)

    @tool("direction_readout", "Project a direction onto the vocabulary: which tokens it promotes / suppresses.")
    def direction_readout(self, session_id: str, direction: dict, k: int = 10, final_norm: bool = True,
                          actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        return T.direction_readout(ws, direction, self.dirs, min(int(k), 50), self.sig, final_norm)

    @tool("direction_info", "Resolve a direction spec: dimension, norm-1 vector top entries, and a stored id.")
    def direction_info(self, session_id: str, direction: dict, actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        v, desc = resolve(ws, direction, self.dirs)
        did = self.dirs.put(v, {"description": desc}, ws.id)
        top = torch.topk(v.abs(), min(8, v.numel())).indices
        return {"id": did, "description": desc, "dim": v.numel(), "top_entries": [[int(i), rnd(v[i])] for i in top]}

    @tool("direction_cosine", "Cosine similarity between two directions of the same dimension.")
    def direction_cosine(self, session_id: str, a: dict, b: dict, actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        va, da = resolve(ws, a, self.dirs)
        vb, db_ = resolve(ws, b, self.dirs)
        if va.numel() != vb.numel():
            raise WSPError(f"dims differ: {va.numel()} vs {vb.numel()}", code="shape_mismatch")
        return {"a": da, "b": db_, "cosine": rnd(torch.dot(va, vb).item())}

    # ================================================================== M1: probing
    def _encode(self, ws, inputs):
        return P.encode_all(ws.adapter, inputs, int(self.limits["max_batch_inputs"]), int(self.limits["max_seq_len"]))

    @tool("run_forward", "Run the model on inputs; returns top-k next-token predictions (LMs) or outputs. "
          "Needs 'outputs' access; costs 1 pass per input.")
    def run_forward(self, session_id: str, inputs: list, top_k: int = 5, positions: Any = "last",
                    actor: str = "agent") -> dict:
        s = self.sessions.require(session_id, "outputs")
        _, ws = self._session(session_id)
        encs = self._encode(ws, inputs)
        self._charge(s, len(encs), "run_forward")
        with ws.lock:
            res = P.run_forward(ws.adapter, encs, min(int(top_k), 20), positions, self.sig)
        return {"results": res, "budget": self._budget(s)}

    @tool("generate", "Generate a continuation for each input (LMs). Needs 'outputs' access; 1 pass per input.")
    def generate(self, session_id: str, inputs: list, settings: dict | None = None, actor: str = "agent") -> dict:
        from .evals.scorers import GenerationSettings, detect_loop
        s = self.sessions.require(session_id, "outputs")
        _, ws = self._session(session_id)
        encs = self._encode(ws, inputs)
        gs = GenerationSettings.from_dict(settings)
        self._charge(s, len(encs), "generate")
        out = []
        with ws.lock:
            for enc in encs:
                g = ws.adapter.generate(enc, gs.to_dict())
                out.append({"text": g["text"][-1500:], "new_tokens": len(g["new_ids"]),
                            "hit_token_limit": g["hit_token_limit"], "looped": detect_loop(g["new_ids"])})
        return {"results": out, "settings": gs.to_dict(), "budget": self._budget(s)}

    @tool("probe_activations", "Capture activations at hook points for inputs; returns summaries (norms, top units). "
          "Needs 'activations' access; 1 pass per input.")
    def probe_activations(self, session_id: str, inputs: list, hooks: list[str], positions: Any = "last",
                          top_k: int = 8, save_mean: bool = False, actor: str = "agent") -> dict:
        s = self.sessions.require(session_id, "activations")
        _, ws = self._session(session_id)
        encs = self._encode(ws, inputs)
        self._charge(s, len(encs), "probe_activations")
        with ws.lock:
            res = P.probe_activations(ws.adapter, encs, hooks, positions, min(int(top_k), 32), self.sig, self.dirs,
                                      ws.id, save_mean)
        res["budget"] = self._budget(s)
        return res

    @tool("direction_from_activations", "Difference of mean activations (positive minus negative inputs) at a hook, "
          "stored as a direction. Needs 'activations' access; 1 pass per input.")
    def direction_from_activations(self, session_id: str, positive_inputs: list, negative_inputs: list, hook: str,
                                   positions: Any = "last", actor: str = "agent") -> dict:
        s = self.sessions.require(session_id, "activations")
        _, ws = self._session(session_id)
        pe, ne = self._encode(ws, positive_inputs), self._encode(ws, negative_inputs)
        self._charge(s, len(pe) + len(ne), "direction_from_activations")
        with ws.lock:
            res = P.direction_from_activations(ws.adapter, pe, ne, hook, positions, self.sig, self.dirs, ws.id)
        res["budget"] = self._budget(s)
        return res

    @tool("probe_attention", "Attention-pattern summaries per head: entropy, attention by relative offset, top keys "
          "per query. Needs 'activations' access; 1 pass per input.")
    def probe_attention(self, session_id: str, inputs: list, layers: list[int] | None = None,
                        heads: list[int] | None = None, top_k_keys: int = 3, query_positions: Any = None,
                        actor: str = "agent") -> dict:
        s = self.sessions.require(session_id, "activations")
        _, ws = self._session(session_id)
        encs = self._encode(ws, inputs)
        self._charge(s, len(encs), "probe_attention")
        with ws.lock:
            res = P.probe_attention(ws.adapter, encs, layers, heads, min(int(top_k_keys), 8), query_positions, self.sig)
        res["budget"] = self._budget(s)
        return res

    # ================================================================== M2: edits & patches
    @tool("edit_preview", "Show what a list of edit operations would change (tensors, #params, % of model, policy "
          "and sensitive-component flags) without applying.")
    def edit_preview(self, session_id: str, operations: list[dict], actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        return self.patches.preview(ws, operations, s)

    @tool("edit_apply", "Apply edit operations to the working copy as a new draft patch (with rationale). "
          "Regression suites run automatically.")
    def edit_apply(self, session_id: str, operations: list[dict], rationale: str, actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        res = self.patches.apply(ws, operations, rationale, author=f"{actor}:{s.get('label') or s['id']}", session=s)
        self.store.log_event(session_id=s["id"], actor=actor, kind="edit", tool="edit_apply",
                             args={"patch_id": res["patch_id"]}, result_summary=f"applied {res['patch_id']} "
                             f"({res['params_changed']} params)")
        if self.cfg.get("policy.auto_regression_on_apply"):
            try:
                res["regression"] = self._test_patch(ws, res["patch_id"], session_id=s["id"])["summary"]
            except Exception as e:  # noqa: BLE001 - the patch IS applied; report the test failure, don't mask it
                res["regression"] = {"error": f"{type(e).__name__}: {e}",
                                     "note": "patch applied but required suites could not run; status stays 'draft'"}
        return res

    @tool("patch_list", "List patches (optionally for this session's workspace or by status).")
    def patch_list(self, session_id: str | None = None, status: str | None = None, all_workspaces: bool = False,
                   actor: str = "agent") -> dict:
        wid = None
        if session_id and not all_workspaces:
            wid = self.sessions.get(session_id)["workspace_id"]
        items = self.patches.list(wid, status)
        stack = json.loads(self.store.one("SELECT stack_json FROM workspaces WHERE id=?", (wid,))["stack_json"]) \
            if wid else []
        return {"patches": items, "applied_stack": stack}

    @tool("patch_show", "Full manifest of a patch: operations, per-tensor hashes, params changed, rationale, "
          "linked evaluations, status, approvals.")
    def patch_show(self, patch_id: str, session_id: str | None = None, actor: str = "agent") -> dict:
        out = self.patches.show(patch_id)
        if actor != "human":
            for r in out.get("eval_runs", []):
                r.pop("details", None)
        return out

    @tool("patch_diff", "Compare two patches: tensors touched by each, overlapping elements, delta cosine.")
    def patch_diff(self, patch_a: str, patch_b: str, session_id: str | None = None, actor: str = "agent") -> dict:
        return self.patches.diff(patch_a, patch_b)

    @tool("patch_revert", "Revert an applied patch. mode='cascade' (default) also reverts patches above it; "
          "'rebase' re-derives the patches above it.")
    def patch_revert(self, session_id: str, patch_id: str, mode: str = "cascade", actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        return self.patches.revert(ws, patch_id, mode, author=f"{actor}:{s['id']}", session=s)

    @tool("patch_reapply", "Re-apply an existing patch (whose parent is the current top of the stack).")
    def patch_reapply(self, session_id: str, patch_id: str, actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        r = self.patch_io.row(patch_id)
        top = ws.stack[-1] if ws.stack else None
        if r["parent_id"] != top or r["workspace_id"] != ws.id:
            raise WSPError(f"patch {patch_id} has parent {r['parent_id']} but the current top is {top}",
                           code="not_applicable", hint="revert to the parent first, or re-create the edit")
        with ws.lock:
            self.patch_io.apply_delta(ws, patch_id)
            ws.stack.append(patch_id)
            self.wsm.set_stack(ws, ws.stack)
        return {"stack": ws.stack, "model_hash": short(ws.model_hash())}

    @tool("patch_test", "Run the required suites (regression etc.) on parent vs patched; marks the patch 'tested'.")
    def patch_test(self, patch_id: str, session_id: str | None = None, actor: str = "agent") -> dict:
        r = self.patch_io.row(patch_id)
        ws = self.wsm.get(r["workspace_id"])
        return self._test_patch(ws, patch_id, session_id)["summary"]

    def _test_patch(self, ws, patch_id: str, session_id: str | None) -> dict:
        r = self.patch_io.row(patch_id)
        if r["status"] in ("approved", "rejected", "superseded"):
            raise WSPError(f"patch is {r['status']}; cannot re-test", code="bad_state")
        chain = self.patches.chain(patch_id)
        parent_stack = chain[:-1]
        required = list(self.cfg.get("policy.required_suites") or [])
        suites = self.harness.suites_for(ws)
        ids = []
        for req in required:
            if req == "regression":
                ids += [s["id"] for s in suites if s["split"] == "regression"]
            elif any(s["id"] == req for s in suites):
                ids.append(req)
        if not ids:
            ids = [s["id"] for s in suites if s["split"] in ("regression", "practice") and "examples" in s][:2]
        runs, summary = [], []
        for sid in ids:
            before = self.harness.run(ws, sid, kind="regression", stack=parent_stack, session_id=session_id)
            after = self.harness.run(ws, sid, kind="regression", stack=chain, session_id=session_id)
            runs += [before["run_id"], after["run_id"]]
            mb, ma = before["metrics"], after["metrics"]
            summary.append({"suite": sid, "split": before["split"], "before": rnd(mb.get("score") or 0),
                            "after": rnd(ma.get("score") or 0),
                            "delta": rnd((ma.get("score") or 0) - (mb.get("score") or 0))})
        self.patches.link_runs(patch_id, runs)
        if r["status"] == "draft":
            self.patches.set_status(patch_id, "tested")
        self.store.log_event(session_id=session_id, actor="system", kind="eval", tool="patch_test",
                             args={"patch_id": patch_id}, result_summary=json.dumps(summary)[:500])
        return {"summary": {"patch_id": patch_id, "status": "tested", "suites": summary,
                            "side_effects": [x for x in summary if x["delta"] is not None and x["delta"] < -0.02]},
                "runs": runs}

    @tool("patch_export", "Export the patched model (base + patch chain) to safetensors. Blocked unless every patch "
          "in the chain is approved by a human.")
    def patch_export(self, patch_id: str, session_id: str | None = None, actor: str = "agent") -> dict:
        chain = self.patches.chain(patch_id)
        for pid in chain:
            r = self.patch_io.row(pid)
            if r["status"] != "approved":
                raise WSPError(f"export blocked: patch {pid} in the chain is {r['status']!r}, not 'approved'",
                               code="export_blocked", hint="a human must approve it in the UI after it is tested")
            appr = self.store.one("SELECT * FROM approvals WHERE patch_id=? AND decision='approved' ORDER BY id DESC",
                                  (pid,))
            if not appr:
                raise WSPError(f"export blocked: no logged human approval for {pid}", code="export_blocked")
        r = self.patch_io.row(patch_id)
        ws = self.wsm.get(r["workspace_id"])
        from .workspace.workspace import Workspace
        fresh = Workspace(ws.row, self.wsm.fresh_base(ws.id))
        for pid in chain:
            self.patch_io.apply_delta(fresh, pid)
        out = self.cfg.subdir("exports") / patch_id
        out.mkdir(parents=True, exist_ok=True)
        ad = fresh.adapter
        if ad.kind == "hf_causal_lm":
            ad.model.save_pretrained(out)
            if ad.tokenizer is not None:
                ad.tokenizer.save_pretrained(out)
        else:
            from safetensors.torch import save_file
            save_file({k: v.contiguous() for k, v in ad.state().items()}, str(out / "model.safetensors"))
            src_cfg = self.store.one("SELECT base_path FROM workspaces WHERE id=?", (ws.id,))["base_path"]
            from pathlib import Path
            if (Path(src_cfg) / "config.json").exists():
                shutil.copy(Path(src_cfg) / "config.json", out / "config.json")
        final_hash = fresh.model_hash()
        manifest = {"exported_patch": patch_id, "chain": chain, "base_model_hash": ws.base_hash,
                    "model_hash": final_hash, "exported_at": now(), "exported_by": actor,
                    "approvals": [self.store.all("SELECT decision,reviewer,ts,note FROM approvals WHERE patch_id=?",
                                                 (p,)) for p in chain]}
        (out / "wsp_manifest.json").write_text(json.dumps(manifest, indent=2))
        return {"exported_to": str(out), "model_hash": short(final_hash), "chain": chain}

    # ================================================================== M4: human-only approval
    @tool("patch_approve", "Approve a tested patch (human only).", human_only=True)
    def patch_approve(self, patch_id: str, reviewer: str, note: str = "", acknowledge_flags: bool = False,
                      actor: str = "human", session_id: str | None = None) -> dict:
        return self._decide(patch_id, "approved", reviewer, note, acknowledge_flags)

    @tool("patch_reject", "Reject a patch (human only).", human_only=True)
    def patch_reject(self, patch_id: str, reviewer: str, note: str = "", actor: str = "human",
                     session_id: str | None = None) -> dict:
        return self._decide(patch_id, "rejected", reviewer, note, False)

    def _decide(self, pid: str, decision: str, reviewer: str, note: str, ack: bool) -> dict:
        if not reviewer or not reviewer.strip():
            raise WSPError("reviewer name is required", code="bad_arg")
        r = self.patch_io.row(pid)
        if decision == "approved":
            if r["status"] != "tested":
                raise WSPError(f"only 'tested' patches can be approved (this one is {r['status']!r})",
                               code="bad_state", hint="run patch_test first")
            if r["flags"] and not ack:
                raise WSPError(f"patch touches {len(r['flags'])} sensitive component(s); approval requires "
                               "explicit acknowledgement", code="needs_ack")
        self.store.execute("INSERT INTO approvals(patch_id,decision,reviewer,note,acknowledged_flags,ts) "
                           "VALUES (?,?,?,?,?,?)", (pid, decision, reviewer.strip(), note, int(ack), now()))
        self.patches.set_status(pid, decision)
        self.store.log_event(session_id=r["session_id"], actor="human", kind="approval", tool=f"patch_{decision}",
                             args={"patch_id": pid, "reviewer": reviewer}, result_summary=f"{reviewer} {decision} "
                             f"{pid}")
        return {"patch_id": pid, "status": decision, "reviewer": reviewer, "ts": now()}

    @tool("sensitive_register", "Mark a tensor, direction or feature as safety-relevant (human only).",
          human_only=True)
    def sensitive_register(self, kind: str, target: str, reason: str, model_hash: str = "*", reviewer: str = "human",
                           actor: str = "human", session_id: str | None = None) -> dict:
        return self.policy.register(model_hash, kind, target, reason, reviewer)

    @tool("sensitive_list", "List registered sensitive components for this session's model.")
    def sensitive_list(self, session_id: str | None = None, actor: str = "agent") -> dict:
        mh = None
        if session_id:
            mh = self._session(session_id)[1].base_hash
        return {"entries": [{k: e[k] for k in ("id", "kind", "target", "reason", "model_hash")}
                            for e in self.policy.entries(mh)]}

    # ================================================================== M3: harness
    @tool("suite_list", "List test suites for this model: practice (examples visible), held-out and regression "
          "(names and sizes only).")
    def suite_list(self, session_id: str, actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        return {"suites": self.harness.list_suites(ws)}

    @tool("suite_examples", "Page through the examples of a PRACTICE suite (held-out / regression examples are never "
          "returned).")
    def suite_examples(self, session_id: str, suite_id: str, offset: int = 0, actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        suite = self.harness.get_suite(ws, suite_id)
        if suite["split"] != "practice":
            raise WSPError(f"{suite_id} is a {suite['split']} suite; its examples are not available to agents",
                           code="access_denied")
        n = int(self.limits["page_size"])
        ex = suite["examples"]
        return {"suite": suite_id, "total": len(ex), "offset": offset, "examples": ex[offset:offset + n],
                "next_offset": offset + n if offset + n < len(ex) else None}

    @tool("suite_run_practice", "Run a practice suite on the current working copy. Costs 1 pass per example (x samples). "
          "Per-example results need 'outputs' access.")
    def suite_run_practice(self, session_id: str, suite_id: str, limit: int | None = None,
                           settings: dict | None = None, actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        suite = self.harness.get_suite(ws, suite_id)
        if suite["split"] != "practice":
            raise WSPError(f"{suite_id} is a {suite['split']} suite; only practice suites run here",
                           code="access_denied", hint="use evaluate_heldout for aggregate held-out scores")
        n = self.harness.passes_needed(ws, suite_id, settings, limit)
        self._charge(s, n, "suite_run_practice")
        run = self.harness.run(ws, suite_id, kind="practice", settings=settings, limit=limit, session_id=s["id"])
        out = self.harness.agent_view(run, "outputs" in s["capabilities"])
        out["budget"] = self._budget(s)
        return out

    @tool("evaluate_heldout", "Aggregate score on a held-out suite for the current working copy. Never returns "
          "examples; rate-limited; logged.")
    def evaluate_heldout(self, session_id: str, suite_id: str, limit: int | None = None,
                         settings: dict | None = None, actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        suite = self.harness.get_suite(ws, suite_id)
        if suite["split"] != "heldout":
            raise WSPError(f"{suite_id} is not a held-out suite", code="bad_arg")
        lim = int(self.cfg.get("session.heldout_rate_limit_per_hour"))
        if actor == "agent" and self.harness.heldout_count_last_hour(s["id"]) >= lim:
            raise WSPError(f"held-out rate limit reached ({lim} per hour for this session)", code="rate_limited",
                           hint="use practice suites to iterate; save held-out runs for final checks")
        run = self.harness.run(ws, suite_id, kind="heldout", settings=settings, limit=limit, session_id=s["id"])
        return {"run_id": run["run_id"], "suite": suite_id, "metrics": {k: rnd(v) if isinstance(v, float) else v
                                                                        for k, v in run["metrics"].items()},
                "settings": run["settings"], "on_patch": run["patch_id"], "model_hash": run["model_hash"]}

    @tool("run_baseline", "Run a non-edit baseline: the BASE model on a practice or held-out suite with different "
          "generation settings (e.g. temperature, repetition_penalty). Aggregate only for held-out.")
    def run_baseline(self, session_id: str, suite_id: str, settings: dict, limit: int | None = None,
                     actor: str = "agent") -> dict:
        s, ws = self._session(session_id)
        suite = self.harness.get_suite(ws, suite_id)
        if suite["split"] == "practice":
            self._charge(s, self.harness.passes_needed(ws, suite_id, settings, limit), "run_baseline")
        elif suite["split"] == "heldout":
            lim = int(self.cfg.get("session.heldout_rate_limit_per_hour"))
            if actor == "agent" and self.harness.heldout_count_last_hour(s["id"]) >= lim:
                raise WSPError("held-out rate limit reached", code="rate_limited")
            self.store.log_event(session_id=s["id"], actor=actor, kind="eval", tool="evaluate_heldout",
                                 args={"suite_id": suite_id, "baseline": True}, result_summary="baseline heldout run")
        run = self.harness.run(ws, suite_id, kind="baseline", settings=settings, limit=limit, session_id=s["id"],
                               stack=[])
        out = self.harness.agent_view(run, suite["split"] == "practice" and "outputs" in s["capabilities"])
        out["model"] = "base"
        return out

    @tool("compare_runs", "Per-metric comparison of two evaluation runs (with settings-mismatch warnings).")
    def compare_runs(self, run_a: str, run_b: str, session_id: str | None = None, actor: str = "agent") -> dict:
        return self.harness.compare(run_a, run_b)

    # ================================================================== misc
    @tool("session_log", "The session's event log (what has been done, in order).")
    def session_log(self, session_id: str, after_id: int = 0, actor: str = "agent") -> dict:
        evs = self.store.events(session_id, after_id, 100)
        return {"events": [{"id": e["id"], "ts": e["ts"], "actor": e["actor"], "kind": e["kind"], "tool": e["tool"],
                            "ok": bool(e["ok"]), "summary": (e["result_summary"] or "")[:160]} for e in evs]}


# ====================================================================== helpers

def _loggable(args: dict) -> dict:
    out = {}
    for k, v in args.items():
        s = canonical_json(v)
        out[k] = v if len(s) <= 2000 else {"_truncated_json": s[:2000]}
    return out


def _summarise(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)[:300]
    if "error" in result:
        return f"ERROR {result.get('code')}: {result['error']}"[:500]
    keys = []
    for k, v in result.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            keys.append(f"{k}={v}")
        elif isinstance(v, list):
            keys.append(f"{k}=[{len(v)}]")
        elif isinstance(v, dict):
            keys.append(f"{k}={{...}}")
        if len(", ".join(keys)) > 400:
            break
    return ", ".join(keys)[:500]


def _largest_list(obj: Any, cap: int | None = None) -> tuple[tuple | None, list]:
    """Path to the list (len > 1) with the largest serialized size, anywhere in obj. If a cap is given and even
    a single item of that list is too big to page, returns (None, [])."""
    found: list[tuple[int, tuple, list]] = []

    def walk(o, path):
        if isinstance(o, list):
            if len(o) > 1:
                found.append((json_size(o), path, o))
            for i, v in enumerate(o):
                if isinstance(v, (list, dict)):
                    walk(v, path + (i,))
        elif isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (list, dict)):
                    walk(v, path + (k,))

    walk(obj, ())
    if not found:
        return None, []
    found.sort(key=lambda x: -x[0])
    for _, path, items in found:
        if cap is None or max(json_size(x) for x in items) <= cap * 0.6:
            return path, items
    return None, []


def _replace_at(obj: Any, path: tuple, value: Any) -> Any:
    if not path:
        return value
    if isinstance(obj, dict):
        return {k: (_replace_at(v, path[1:], value) if k == path[0] else v) for k, v in obj.items()}
    return [(_replace_at(v, path[1:], value) if i == path[0] else v) for i, v in enumerate(obj)]
