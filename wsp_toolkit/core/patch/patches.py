"""Patches: named, reversible, reviewable weight edits ("git for weights").

A patch = JSON manifest (in SQLite) + a safetensors delta file holding, per changed tensor, the flat indices of
changed elements and their exact old and new values. Applying writes the new values; reverting writes the old
ones. Because values are written (not added), apply -> revert is bit-exact and replaying a patch on a freshly
loaded base reproduces the same hash.

Stacking semantics (see docs/design/patch_semantics.md):
- A workspace has an ordered stack of applied patches; each patch records its parent (the patch below it).
- revert(P, mode="cascade")  pops P and every patch above it.            (default; always exact)
- revert(P, mode="rebase")   pops P and everything above, then re-derives the patches above P from their
                             operation lists on the new state. They get new IDs (parent changed); the originals
                             are marked "superseded".
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..db import Store
from ..edit.ops import compute
from ..util import WSPError, json_hash, now, rnd, short, tensor_hash

STATUSES = ["draft", "tested", "approved", "rejected", "superseded"]


def _flat_diff(old: torch.Tensor, new: torch.Tensor):
    o, n = old.reshape(-1), new.reshape(-1)
    if o.dtype in (torch.bfloat16, torch.float16):
        changed = o.view(torch.int16) != n.view(torch.int16)
    else:
        changed = o != n
    idx = changed.nonzero().flatten()
    return idx, o[idx].clone(), n[idx].clone()


class PatchIO:
    """Low-level delta application, used by the workspace manager to rebuild working copies."""

    def __init__(self, store: Store, patch_dir: Path):
        self.store = store
        self.dir = patch_dir
        self._cache: dict[str, dict] = {}

    def row(self, pid: str) -> dict:
        r = self.store.one("SELECT * FROM patches WHERE id=?", (pid,))
        if not r:
            raise WSPError(f"unknown patch {pid!r}", code="not_found", hint="use patch_list")
        r["manifest"] = json.loads(r["manifest_json"])
        r["flags"] = json.loads(r["flagged_json"])
        return r

    def delta(self, pid: str) -> dict[str, torch.Tensor]:
        if pid not in self._cache:
            self._cache[pid] = load_file(str(self.dir / f"{pid}.safetensors"))
        return self._cache[pid]

    def _write(self, ws, pid: str, which: str) -> None:
        man = self.row(pid)["manifest"]
        d = self.delta(pid)
        expect_before = "hash_before" if which == "new" else "hash_after"
        expect_after = "hash_after" if which == "new" else "hash_before"
        live = ws.tensors()
        for t in man["tensors"]:
            name = t["name"]
            cur = ws.tensor_hash(name)
            if cur != t[expect_before]:
                raise WSPError(f"cannot {'apply' if which == 'new' else 'revert'} {pid}: tensor {name} hash "
                               f"{short(cur)} != expected {short(t[expect_before])}", code="hash_mismatch")
        with torch.no_grad():
            for t in man["tensors"]:
                name = t["name"]
                flat = live[name].view(-1)
                flat[d[f"{name}::idx"]] = d[f"{name}::{which}"]
                ws.invalidate([name])
        for t in man["tensors"]:
            if ws.tensor_hash(t["name"]) != t[expect_after]:
                raise WSPError(f"post-{which} hash check failed for {t['name']} in {pid}", code="hash_mismatch")

    def apply_delta(self, ws, pid: str) -> None:
        self._write(ws, pid, "new")

    def revert_delta(self, ws, pid: str) -> None:
        self._write(ws, pid, "old")


class PatchManager:
    def __init__(self, store: Store, io: PatchIO, wsm, dirs, policy, limits: dict):
        self.store, self.io, self.wsm, self.dirs, self.policy, self.limits = store, io, wsm, dirs, policy, limits

    # ------------------------------------------------------------------ compute
    def _compute_all(self, ws, ops: list[dict]):
        if not isinstance(ops, list) or not ops:
            raise WSPError("operations must be a non-empty list of op objects", code="bad_op")
        if len(ops) > 32:
            raise WSPError("at most 32 operations per patch", code="too_large")
        originals: dict[str, torch.Tensor] = {}
        resolved = []
        live = ws.tensors()
        try:
            for op in ops:
                newvals, r = compute(ws, op, self.dirs, self.limits)
                for name, val in newvals.items():
                    if name not in originals:
                        originals[name] = live[name].detach().clone()
                    with torch.no_grad():  # apply temporarily so later ops see earlier ones
                        live[name].copy_(val)
                    ws.invalidate([name])
                resolved.append(r)
            finals = {n: live[n].detach().clone() for n in originals}
        finally:
            with torch.no_grad():
                for n, v in originals.items():
                    live[n].copy_(v)
                    ws.invalidate([n])
        return originals, finals, resolved

    def _summary(self, ws, originals, finals) -> tuple[list[dict], dict, int]:
        rows, deltas, total = [], {}, 0
        for name in sorted(originals):
            old, new = originals[name], finals[name]
            idx, ov, nv = _flat_diff(old, new)
            if idx.numel() == 0:
                continue
            d = (new.float() - old.float())
            deltas[name] = d
            on = old.float().norm().item()
            rows.append({"name": name, "changed": int(idx.numel()), "numel": old.numel(),
                         "delta_norm": rnd(d.norm()), "rel_change": rnd(d.norm().item() / on if on else 0),
                         "max_abs_delta": rnd(d.abs().max())})
            total += int(idx.numel())
        return rows, deltas, total

    def preview(self, ws, ops: list[dict], session: dict | None = None) -> dict:
        with ws.lock:
            originals, finals, resolved = self._compute_all(ws, ops)
            rows, deltas, total = self._summary(ws, originals, finals)
        n = ws.n_params()
        pct = 100.0 * total / n if n else 0
        violations = self.policy.check_patch(session, ops, pct)
        flags = self.policy.sensitive_flags(ws, deltas, resolved)
        return {"would_change": rows, "params_changed": total, "pct_of_model": rnd(pct),
                "policy_violations": violations, "sensitive_flags": flags, "operations": resolved,
                "applied": False}

    # ------------------------------------------------------------------ apply
    def apply(self, ws, ops: list[dict], rationale: str, author: str, session: dict | None = None) -> dict:
        if not rationale or len(rationale.strip()) < 3:
            raise WSPError("a rationale is required: explain why this edit should fix the problem", code="bad_arg")
        with ws.lock:
            originals, finals, resolved = self._compute_all(ws, ops)
            rows, deltas, total = self._summary(ws, originals, finals)
            if not rows:
                raise WSPError("these operations change no values", code="no_change")
            n = ws.n_params()
            pct = 100.0 * total / n if n else 0
            violations = self.policy.check_patch(session, ops, pct)
            if violations:
                raise WSPError("patch blocked by policy: " + "; ".join(violations), code="policy_violation")
            flags = self.policy.sensitive_flags(ws, deltas, resolved)
            parent = ws.stack[-1] if ws.stack else None
            tensors, tensor_payload = [], {}
            for r in rows:
                name = r["name"]
                idx, ov, nv = _flat_diff(originals[name], finals[name])
                tensor_payload[f"{name}::idx"] = idx.to(torch.int64)
                tensor_payload[f"{name}::old"] = ov.contiguous()
                tensor_payload[f"{name}::new"] = nv.contiguous()
                tensors.append({**r, "hash_before": tensor_hash(originals[name]),
                                "hash_after": tensor_hash(finals[name])})
            before = ws.model_hash()
            ident = {"workspace_id": ws.id, "base_model_hash": ws.base_hash, "parent_patch": parent, "operations": resolved,
                     "tensors": [(t["name"], t["hash_before"], t["hash_after"]) for t in tensors]}
            pid = "p_" + json_hash(ident)[:16]
            manifest = {
                "patch_id": pid, "workspace_id": ws.id, "base_model_hash": ws.base_hash, "parent_patch": parent,
                "model_hash_before": before, "operations": resolved, "tensors": tensors,
                "params_changed": total, "pct_of_model": rnd(pct, 6), "rationale": rationale.strip(),
                "author": author, "created_at": now(), "evals": [],
            }
            exists = self.store.one("SELECT id FROM patches WHERE id=?", (pid,))
            if not exists:
                save_file(tensor_payload, str(self.io.dir / f"{pid}.safetensors"),
                          metadata={"patch_id": pid, "base_model_hash": ws.base_hash})
            self.io._cache.pop(pid, None)
            if not exists:
                self.store.execute(
                    "INSERT INTO patches(id,workspace_id,base_hash,parent_id,status,manifest_json,delta_path,"
                    "flagged_json,author,session_id,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, ws.id, ws.base_hash, parent, "draft", "{}", str(self.io.dir / f"{pid}.safetensors"),
                     json.dumps(flags), author, session["id"] if session else None, now()))
            self.store.execute("UPDATE patches SET manifest_json=? WHERE id=?", (json.dumps(manifest), pid))
            self.io.apply_delta(ws, pid)
            ws.stack.append(pid)
            self.wsm.set_stack(ws, ws.stack)
            after = ws.model_hash()
            manifest["model_hash_after"] = after
            self.store.execute("UPDATE patches SET manifest_json=? WHERE id=?", (json.dumps(manifest), pid))
        return {"patch_id": pid, "status": "draft", "params_changed": total, "pct_of_model": rnd(pct),
                "tensors": [{k: t[k] for k in ("name", "changed", "rel_change")} for t in tensors],
                "model_hash_after": short(after), "sensitive_flags": flags, "stack": list(ws.stack)}

    # ------------------------------------------------------------------ revert
    def revert(self, ws, pid: str, mode: str = "cascade", author: str = "agent", session: dict | None = None) -> dict:
        with ws.lock:
            if pid not in ws.stack:
                raise WSPError(f"patch {pid} is not applied in workspace {ws.id}; applied stack: {ws.stack}",
                               code="not_applied")
            pos = ws.stack.index(pid)
            above = ws.stack[pos + 1:]
            for p in reversed(ws.stack[pos:]):
                self.io.revert_delta(ws, p)
                ws.stack.pop()
            self.wsm.set_stack(ws, ws.stack)
            rebased = []
            if mode == "rebase":
                for old in above:
                    r = self.io.row(old)
                    m = r["manifest"]
                    res = self.apply(ws, m["operations"], f"[rebased from {old} after reverting {pid}] {m['rationale']}",
                                     author, session)
                    self.store.execute("UPDATE patches SET status='superseded' WHERE id=?", (old,))
                    rebased.append({"old": old, "new": res["patch_id"]})
            elif mode != "cascade":
                raise WSPError("mode must be 'cascade' or 'rebase'", code="bad_arg")
            return {"reverted": pid, "also_reverted": [] if mode == "rebase" else above, "rebased": rebased,
                    "stack": list(ws.stack), "model_hash": short(ws.model_hash()),
                    "base_restored": not ws.stack and ws.model_hash() == ws.base_hash}

    # ------------------------------------------------------------------ queries
    def list(self, workspace_id: str | None = None, status: str | None = None) -> list[dict]:
        q, args = "SELECT id,workspace_id,parent_id,status,author,created_at,flagged_json,manifest_json FROM patches", []
        conds = []
        if workspace_id:
            conds.append("workspace_id=?")
            args.append(workspace_id)
        if status:
            conds.append("status=?")
            args.append(status)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY created_at DESC"
        out = []
        for r in self.store.all(q, args):
            m = json.loads(r["manifest_json"])
            out.append({"patch_id": r["id"], "workspace_id": r["workspace_id"], "parent": r["parent_id"],
                        "status": r["status"], "author": r["author"], "created_at": r["created_at"],
                        "params_changed": m.get("params_changed"), "pct_of_model": m.get("pct_of_model"),
                        "ops": [o.get("op") for o in m.get("operations", [])],
                        "flagged": len(json.loads(r["flagged_json"])) > 0,
                        "rationale": (m.get("rationale") or "")[:200]})
        return out

    def show(self, pid: str) -> dict:
        r = self.io.row(pid)
        m = r["manifest"]
        approvals = self.store.all("SELECT decision,reviewer,note,acknowledged_flags,ts FROM approvals WHERE patch_id=?"
                                   " ORDER BY id", (pid,))
        ev = m.get("evals", [])
        marks = ",".join("?" * len(ev))
        runs = self.store.all(f"SELECT id,suite_id,split,kind,metrics_json,patch_id FROM runs WHERE id IN ({marks})",
                              ev) if ev else []
        return {**m, "status": r["status"], "flags": r["flags"], "approvals": approvals,
                "eval_runs": [{"run_id": x["id"], "suite": x["suite_id"], "split": x["split"], "kind": x["kind"],
                               "on": "patched" if x["patch_id"] else "base",
                               "metrics": json.loads(x["metrics_json"])} for x in runs]}

    def chain(self, pid: str) -> list[str]:
        out, cur = [], pid
        while cur:
            out.append(cur)
            cur = self.io.row(cur)["parent_id"]
        return list(reversed(out))

    def diff(self, a: str, b: str) -> dict:
        ma, mb = self.io.row(a)["manifest"], self.io.row(b)["manifest"]
        da, db_ = self.io.delta(a), self.io.delta(b)
        ta = {t["name"] for t in ma["tensors"]}
        tb = {t["name"] for t in mb["tensors"]}
        shared = []
        for name in sorted(ta & tb):
            ia, ib = set(da[f"{name}::idx"].tolist()), set(db_[f"{name}::idx"].tolist())
            union = sorted(ia | ib)
            u = torch.tensor(union)
            va = torch.zeros(len(union))
            vb = torch.zeros(len(union))
            pos = {j: k for k, j in enumerate(union)}
            for idxs, old, new, v in [(da[f"{name}::idx"], da[f"{name}::old"], da[f"{name}::new"], va),
                                      (db_[f"{name}::idx"], db_[f"{name}::old"], db_[f"{name}::new"], vb)]:
                for j, o, n in zip(idxs.tolist(), old.float().tolist(), new.float().tolist()):
                    v[pos[j]] = n - o
            cos = torch.nn.functional.cosine_similarity(va, vb, dim=0).item() if u.numel() else 0
            shared.append({"tensor": name, "elements_a": len(ia), "elements_b": len(ib),
                           "overlapping_elements": len(ia & ib), "delta_cosine": rnd(cos)})
        return {"a": a, "b": b, "only_in_a": sorted(ta - tb), "only_in_b": sorted(tb - ta), "shared": shared,
                "same_base": ma["base_model_hash"] == mb["base_model_hash"],
                "ops_a": [o["op"] for o in ma["operations"]], "ops_b": [o["op"] for o in mb["operations"]]}

    def set_status(self, pid: str, status: str) -> None:
        if status not in STATUSES:
            raise WSPError(f"bad status {status}", code="bad_arg")
        self.store.execute("UPDATE patches SET status=? WHERE id=?", (status, pid))

    def link_runs(self, pid: str, run_ids: list[str]) -> None:
        r = self.io.row(pid)
        m = r["manifest"]
        m["evals"] = list(dict.fromkeys(m.get("evals", []) + run_ids))
        self.store.execute("UPDATE patches SET manifest_json=? WHERE id=?", (json.dumps(m), pid))
