"""Edit primitives. Each primitive maps (workspace, op dict) -> {tensor name: new full tensor}; nothing is
written here. The patch layer diffs the result against the current values and stores only what changed.

Axis convention: for an nn.Linear weight W of shape [out, in], axis 0 indexes outputs (rows) and axis 1 indexes
inputs (columns). ``side: "output"`` is shorthand for ``axis: 0``; ``side: "input"`` for ``axis: 1``.
"""

from __future__ import annotations

import torch

from ..directions import DirectionStore, resolve
from ..util import WSPError, parse_slice

PRIMITIVES = ["project_out_direction", "scale", "zero", "set_slice", "add_low_rank", "graft_subspace"]


def _region(ws, op: dict, t: torch.Tensor) -> tuple:
    """Index tuple selecting the op's region (head and/or slice); default = whole tensor."""
    name = op["tensor"]
    idx: list = [slice(None)] * t.dim()
    if op.get("head") is not None:
        lay = ws.adapter.head_layout(name)
        if lay is None:
            raise WSPError(f"{name} has no head layout; use 'slice' instead", code="bad_op")
        h = int(op["head"])
        if not 0 <= h < lay.n_heads:
            raise WSPError(f"head {h} out of range 0..{lay.n_heads - 1}", code="bad_op")
        idx[lay.axis] = slice(h * lay.d_head, (h + 1) * lay.d_head)
    if op.get("slice"):
        if op.get("head") is not None:
            raise WSPError("give either 'head' or 'slice', not both", code="bad_op")
        idx = list(parse_slice(op["slice"], tuple(t.shape), name))
    return tuple(idx)


def _indices(spec, n: int, what: str) -> torch.Tensor:
    if spec is None:
        return torch.arange(n)
    if isinstance(spec, str):
        sl = parse_slice(spec, (n,), what)[0]
        return torch.arange(sl.start, sl.stop)
    idx = torch.tensor([int(i) for i in spec], dtype=torch.long)
    if idx.numel() and (idx.min() < 0 or idx.max() >= n):
        raise WSPError(f"{what} indices must be in 0..{n - 1}", code="bad_op")
    return idx


def compute(ws, op: dict, dirs: DirectionStore, limits: dict) -> tuple[dict[str, torch.Tensor], dict]:
    """Return ({tensor: new values}, resolved op for the manifest)."""
    kind = op.get("op")
    if kind not in PRIMITIVES:
        raise WSPError(f"unknown edit primitive {kind!r}; available: {PRIMITIVES}", code="bad_op")
    if kind == "graft_subspace":
        raise WSPError("graft_subspace is not implemented yet (design note in docs/design/graft_subspace.md)",
                       code="not_implemented")
    if "tensor" not in op:
        raise WSPError(f"{kind} needs a 'tensor' field", code="bad_op")
    name = op["tensor"]
    cur = ws.tensor(name)
    new = cur.detach().clone()
    resolved = dict(op)
    work = new.float() if new.dtype != torch.float32 else new

    if kind in ("scale", "zero"):
        reg = _region(ws, op, cur)
        factor = 0.0 if kind == "zero" else float(op.get("factor", 1.0))
        work[reg] = work[reg] * factor
        resolved["factor"] = factor

    elif kind == "set_slice":
        if not op.get("slice"):
            raise WSPError("set_slice needs 'slice'", code="bad_op")
        reg = _region(ws, op, cur)
        target = work[reg]
        if target.numel() > limits["max_slice_elements"]:
            raise WSPError(f"set_slice region has {target.numel()} elements, cap is {limits['max_slice_elements']}",
                           code="too_large", hint="use scale/zero/project_out_direction for large regions")
        vals = torch.tensor(op["values"], dtype=torch.float32)
        try:
            work[reg] = vals.expand_as(target) if vals.dim() == 0 else vals.reshape(target.shape)
        except RuntimeError as e:
            raise WSPError(f"values shape {list(vals.shape)} does not match region shape {list(target.shape)}",
                           code="bad_op") from e

    elif kind == "project_out_direction":
        if work.dim() != 2:
            raise WSPError(f"{name} has shape {list(cur.shape)}; project_out_direction needs a 2-D weight",
                           code="bad_op")
        axis = int(op.get("axis", {"output": 0, "input": 1}.get(op.get("side", "output"), 0)))
        d, desc = resolve(ws, op["direction"], dirs)
        if d.numel() != work.shape[axis]:
            raise WSPError(f"direction has dim {d.numel()} but {name} axis {axis} has size {work.shape[axis]}",
                           code="shape_mismatch", hint="use side='output' (axis 0, rows) or 'input' (axis 1, cols)")
        other = 1 - axis
        idx = _indices(op.get("indices"), work.shape[other], "indices")
        s = float(op.get("strength", 1.0))
        if axis == 0:
            sub = work[:, idx]
            work[:, idx] = sub - s * torch.outer(d, d @ sub)
        else:
            sub = work[idx, :]
            work[idx, :] = sub - s * torch.outer(sub @ d, d)
        did = dirs.put(d, {"description": desc}, ws.id)
        resolved.update({"direction": {"id": did}, "direction_description": desc, "axis": axis,
                         "n_indices": int(idx.numel()), "strength": s})

    elif kind == "add_low_rank":
        if work.dim() != 2:
            raise WSPError("add_low_rank needs a 2-D weight", code="bad_op")
        coeffs = op.get("coeffs")
        if "U" in op:
            U = torch.tensor(op["U"], dtype=torch.float32)
            V = torch.tensor(op["V"], dtype=torch.float32)
            if U.dim() == 1:
                U, V = U[:, None], V[:, None]
        else:
            us = [resolve(ws, s, dirs)[0] for s in op["u"]]
            vs = [resolve(ws, s, dirs)[0] for s in op["v"]]
            if len(us) != len(vs):
                raise WSPError("u and v need the same number of directions", code="bad_op")
            U, V = torch.stack(us, 1), torch.stack(vs, 1)
            resolved["u"] = [{"id": dirs.put(u, {"description": "low-rank u"}, ws.id)} for u in us]
            resolved["v"] = [{"id": dirs.put(v, {"description": "low-rank v"}, ws.id)} for v in vs]
        k = U.shape[1]
        if k > limits["max_low_rank"]:
            raise WSPError(f"rank {k} exceeds max_low_rank {limits['max_low_rank']}", code="too_large")
        if U.shape[0] != work.shape[0] or V.shape[0] != work.shape[1]:
            raise WSPError(f"U must be [{work.shape[0]}, k] and V [{work.shape[1]}, k]; got {list(U.shape)} and "
                           f"{list(V.shape)}", code="shape_mismatch")
        c = torch.tensor(coeffs if coeffs is not None else [1.0] * k, dtype=torch.float32)
        work += (U * c) @ V.T
        resolved["rank"] = k

    if work is not new:
        new = work.to(cur.dtype)
    return {name: new}, resolved
