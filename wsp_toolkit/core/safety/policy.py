"""Policy limits and the sensitive-component registry.

Policy (config [policy] + optional per-session overrides that can only tighten):
- max_param_change_pct per patch
- allowed_primitives
Sensitive registry entries (table ``registry``):
- kind "tensor":    target = exact tensor name or a regex prefixed with "re:"
- kind "direction": target = a direction id; a patch is flagged if any changed tensor's delta has a top singular
                    vector with |cosine| >= policy.sensitive_direction_cosine to it (when dims match), or if a
                    project_out/add_low_rank op uses a direction that close to it
- kind "feature":   target = a feature id (M6); resolved to its decoder direction and treated like "direction"
Flagged patches need explicit acknowledgement at approval time.
"""

from __future__ import annotations

import json
import re
import uuid

import torch

from ..db import Store
from ..util import WSPError, now, rnd


class PolicyEngine:
    def __init__(self, cfg, store: Store, dirs):
        self.cfg = cfg
        self.store = store
        self.dirs = dirs

    def effective(self, session: dict | None) -> dict:
        base = {"max_param_change_pct": float(self.cfg.get("policy.max_param_change_pct")),
                "allowed_primitives": list(self.cfg.get("policy.allowed_primitives"))}
        if session:
            over = json.loads(session.get("policy_json") or "{}")
            if "max_param_change_pct" in over:
                base["max_param_change_pct"] = min(base["max_param_change_pct"], float(over["max_param_change_pct"]))
            if "allowed_primitives" in over:
                base["allowed_primitives"] = [p for p in base["allowed_primitives"] if p in over["allowed_primitives"]]
        return base

    def check_patch(self, session: dict | None, ops: list[dict], pct: float, n_params: int | None = None) -> list[str]:
        pol = self.effective(session)
        v = []
        for op in ops:
            if op.get("op") not in pol["allowed_primitives"]:
                v.append(f"primitive {op.get('op')!r} not allowed in this session (allowed: {pol['allowed_primitives']})")
        min_params = int(self.cfg.get("policy.pct_limit_min_params", 0) or 0)
        exempt = n_params is not None and n_params < min_params
        if pct > pol["max_param_change_pct"] and not exempt:
            v.append(f"patch changes {pct:.4g}% of parameters, above the limit of {pol['max_param_change_pct']}%")
        return v

    # ------------------------------------------------------------ registry
    def register(self, model_hash: str, kind: str, target: str, reason: str, created_by: str) -> dict:
        if kind not in ("tensor", "direction", "feature"):
            raise WSPError("kind must be tensor | direction | feature", code="bad_arg")
        rid = "sens_" + uuid.uuid4().hex[:10]
        self.store.execute("INSERT INTO registry(id,model_hash,kind,target,reason,created_by,created_at) "
                           "VALUES (?,?,?,?,?,?,?)", (rid, model_hash, kind, target, reason, created_by, now()))
        if kind == "feature":
            self.store.execute("UPDATE features SET sensitive=1 WHERE id=?", (target,))
        return {"id": rid, "kind": kind, "target": target, "model_hash": model_hash}

    def entries(self, model_hash: str | None = None) -> list[dict]:
        if model_hash:
            return self.store.all("SELECT * FROM registry WHERE model_hash IN (?, '*') ORDER BY created_at",
                                  (model_hash,))
        return self.store.all("SELECT * FROM registry ORDER BY created_at")

    def _direction_vec(self, entry: dict) -> torch.Tensor | None:
        if entry["kind"] == "direction":
            return self.dirs.get(entry["target"])[0]
        if entry["kind"] == "feature":
            row = self.store.one("SELECT record_json FROM features WHERE id=?", (entry["target"],))
            if row:
                rec = json.loads(row["record_json"])
                return torch.tensor(rec["decoder_direction"]) if "decoder_direction" in rec else None
        return None

    def sensitive_flags(self, ws, deltas: dict[str, torch.Tensor], resolved_ops: list[dict]) -> list[dict]:
        flags = []
        thresh = float(self.cfg.get("policy.sensitive_direction_cosine"))
        for e in self.entries(ws.base_hash):
            if e["kind"] == "tensor":
                t = e["target"]
                hit = [n for n in deltas if (re.search(t[3:], n) if t.startswith("re:") else n == t)]
                if hit:
                    flags.append({"registry_id": e["id"], "kind": "tensor", "target": t, "tensors": hit,
                                  "reason": e["reason"]})
                continue
            v = self._direction_vec(e)
            if v is None:
                continue
            v = v / v.norm()
            best = 0.0
            where = None
            for name, d in deltas.items():
                if d.dim() != 2:
                    continue
                for axis in (0, 1):
                    if d.shape[axis] != v.numel():
                        continue
                    M = d if axis == 0 else d.T
                    if M.norm() == 0:
                        continue
                    # fraction of the delta's energy that lies along v
                    frac = ((v @ M).norm() / M.norm()).item()
                    if frac > best:
                        best, where = frac, f"{name} (axis {axis})"
            for op in resolved_ops:
                for key in ("direction",):
                    if isinstance(op.get(key), dict) and "id" in op[key]:
                        dv = self.dirs.get(op[key]["id"])[0]
                        if dv.numel() == v.numel():
                            c = abs(torch.nn.functional.cosine_similarity(dv, v, dim=0).item())
                            if c > best:
                                best, where = c, f"op direction {op[key]['id']}"
            if best >= thresh:
                flags.append({"registry_id": e["id"], "kind": e["kind"], "target": e["target"],
                              "alignment": rnd(best), "where": where, "reason": e["reason"]})
        return flags
