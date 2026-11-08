"""Weight inspection: listing, slicing, statistics, SVD, comparison and vocabulary readout.

All functions return small JSON-able dicts; floats are rounded to a few significant digits.
"""

from __future__ import annotations

import math
import re

import torch

from ..directions import DirectionStore, resolve
from ..util import WSPError, parse_slice, rnd, rnd_list, rnd_nested, slice_numel, slice_to_str


def _select(ws, name: str, slice_spec: str | None = None, head: int | None = None) -> tuple[torch.Tensor, str]:
    t = ws.tensor(name)
    region = "all"
    if head is not None:
        lay = ws.adapter.head_layout(name)
        if lay is None:
            raise WSPError(f"{name} is not an attention projection with a head layout", code="bad_arg",
                           hint="head selection works on q/k/v/o_proj weights")
        if not 0 <= head < lay.n_heads:
            raise WSPError(f"head {head} out of range 0..{lay.n_heads - 1}", code="bad_arg")
        a, b = head * lay.d_head, (head + 1) * lay.d_head
        t = t[a:b] if lay.axis == 0 else t[:, a:b]
        region = f"head {head} ({'rows' if lay.axis == 0 else 'cols'} {a}:{b})"
    if slice_spec:
        sl = parse_slice(slice_spec, tuple(t.shape), name if head is None else f"{name}[head {head}]")
        t = t[sl]
        region = (region + " / " if head is not None else "") + slice_to_str(sl)
    return t, region


def tensor_list(ws, filter: str | None = None, offset: int = 0, limit: int = 50) -> dict:
    st = ws.tensors()
    tied = ws.adapter.tied_groups()
    names = sorted(st)
    if filter:
        try:
            rx = re.compile(filter)
        except re.error as e:
            raise WSPError(f"bad regex {filter!r}: {e}", code="bad_arg") from e
        names = [n for n in names if rx.search(n)]
    groups: dict[str, int] = {}
    for n in names:
        g = ws.adapter.layer_of(n)
        groups[g] = groups.get(g, 0) + st[n].numel()
    page = names[offset:offset + limit]
    items = []
    for n in page:
        t = st[n]
        it = {"name": n, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "numel": t.numel(),
              "group": ws.adapter.layer_of(n)}
        lay = ws.adapter.head_layout(n)
        if lay:
            it["heads"] = {"n_heads": lay.n_heads, "d_head": lay.d_head, "axis": "rows" if lay.axis == 0 else "cols"}
        if n in tied:
            it["tied_to"] = tied[n]
        items.append(it)
    return {"total_tensors": len(names), "offset": offset, "returned": len(items),
            "next_offset": offset + len(items) if offset + len(items) < len(names) else None,
            "params_by_group": groups, "tensors": items}


def tensor_get(ws, name: str, slice_spec: str | None, max_elements: int, sig: int, head: int | None = None) -> dict:
    t, region = _select(ws, name, slice_spec, head)
    n = t.numel()
    if n > max_elements:
        raise WSPError(f"slice {region} of {name} has {n} elements, above the cap of {max_elements}",
                       code="too_large", hint=f"request a smaller slice (shape {list(t.shape)}), or use tensor_stats / "
                       "tensor_svd for summaries")
    return {"tensor": name, "region": region, "shape": list(t.shape), "values": rnd_nested(t, sig)}


def _summary(t: torch.Tensor, sig: int) -> dict:
    x = t.detach().float().flatten()
    n = x.numel()
    absx = x.abs()
    mx = float(absx.max()) if n else 0.0
    q = torch.quantile(x[torch.randperm(n, generator=torch.Generator().manual_seed(0))[:100000]] if n > 100000 else x,
                       torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    std = float(x.std()) if n > 1 else 0.0
    mean = float(x.mean())
    hist = torch.histc(x, bins=16, min=float(x.min()), max=float(x.max())) if n else torch.zeros(16)
    return {
        "numel": n, "mean": rnd(mean, sig), "std": rnd(std, sig), "min": rnd(x.min(), sig), "max": rnd(x.max(), sig),
        "abs_mean": rnd(absx.mean(), sig), "fro_norm": rnd(x.norm(), sig),
        "frac_exact_zero": rnd(float((x == 0).float().mean()), sig),
        "frac_small": rnd(float((absx < 1e-3 * mx).float().mean()) if mx else 1.0, sig),
        "frac_outliers_6sd": rnd(float(((x - mean).abs() > 6 * std).float().mean()) if std else 0.0, sig),
        "quantiles": dict(zip(["p1", "p5", "p25", "p50", "p75", "p95", "p99"], rnd_list(q.tolist(), sig))),
        "histogram": {"min": rnd(x.min(), sig), "max": rnd(x.max(), sig), "counts": [int(c) for c in hist.tolist()]},
    }


def _norm_summary(norms: torch.Tensor, label: str, k: int, sig: int) -> dict:
    order = torch.argsort(norms, descending=True)
    return {
        f"{label}_norm_mean": rnd(norms.mean(), sig), f"{label}_norm_std": rnd(norms.std() if norms.numel() > 1 else 0, sig),
        f"top_{label}s": [{"index": int(i), "norm": rnd(norms[i], sig)} for i in order[:k]],
        f"bottom_{label}s": [{"index": int(i), "norm": rnd(norms[i], sig)} for i in order[-k:].flip(0)],
    }


def tensor_stats(ws, name: str, slice_spec: str | None, sig: int, head: int | None = None, k: int = 5) -> dict:
    t, region = _select(ws, name, slice_spec, head)
    out = {"tensor": name, "region": region, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
           "stats": _summary(t, sig)}
    tf = t.float()
    if t.dim() == 2:
        out["rows"] = _norm_summary(tf.norm(dim=1), "row", k, sig)
        out["cols"] = _norm_summary(tf.norm(dim=0), "col", k, sig)
    lay = ws.adapter.head_layout(name)
    if lay and head is None and not slice_spec:
        full = ws.tensor(name).float()
        hn = [full[h * lay.d_head:(h + 1) * lay.d_head].norm() if lay.axis == 0 else
              full[:, h * lay.d_head:(h + 1) * lay.d_head].norm() for h in range(lay.n_heads)]
        out["head_norms"] = rnd_list(hn, sig)
    return out


def _svd(t: torch.Tensor, k: int):
    m = min(t.shape)
    if m > 1024 and k < m // 4:
        U, S, V = torch.svd_lowrank(t, q=min(m, k + 16), niter=4)
        total = float((t * t).sum())
        return U, S, V.T, total, True
    U, S, Vh = torch.linalg.svd(t, full_matrices=False)
    return U, S, Vh, float((S * S).sum()), False


def tensor_svd(ws, name: str, k: int, slice_spec: str | None, sig: int, dirs: DirectionStore,
               head: int | None = None, top_entries: int = 6) -> dict:
    t, region = _select(ws, name, slice_spec, head)
    if t.dim() != 2:
        raise WSPError(f"{name} region has shape {list(t.shape)}; SVD needs 2-D", code="bad_arg")
    t = t.float()
    k = max(1, min(k, min(t.shape), 32))
    U, S, Vh, total, approx = _svd(t, k)
    full_S = S
    energy = (S[:k] ** 2).sum().item() / total if total else 0.0
    out = {"tensor": name, "region": region, "shape": list(t.shape), "approximate": approx,
           "singular_values": rnd_list(S[:k].tolist(), sig), "energy_top_k": rnd(energy, sig)}
    if not approx:
        p = (full_S ** 2) / (full_S ** 2).sum() if total else full_S
        ent = -(p[p > 0] * p[p > 0].log()).sum().item()
        out["effective_rank"] = rnd(math.exp(ent), sig)
        out["stable_rank"] = rnd(total / float(full_S[0] ** 2) if total else 0, sig)
    comps = []
    for i in range(min(k, S.numel())):
        u, v = U[:, i], Vh[i]
        ui = torch.argsort(u.abs(), descending=True)[:top_entries]
        vi = torch.argsort(v.abs(), descending=True)[:top_entries]
        left_id = dirs.put(u, {"description": f"left singular vector {i} of {name} [{region}]", "space": "rows",
                               "tensor": name}, ws.id)
        right_id = dirs.put(v, {"description": f"right singular vector {i} of {name} [{region}]", "space": "cols",
                                "tensor": name}, ws.id)
        comps.append({"index": i, "sigma": rnd(S[i], sig), "left_direction": left_id, "right_direction": right_id,
                      "left_top_entries": [[int(j), rnd(u[j], sig)] for j in ui],
                      "right_top_entries": [[int(j), rnd(v[j], sig)] for j in vi]})
    out["components"] = comps
    return out


def tensor_compare(ws_a, name_a: str, ws_b, name_b: str, slice_spec: str | None, sig: int, k: int = 4) -> dict:
    a, region = _select(ws_a, name_a, slice_spec)
    b, _ = _select(ws_b, name_b, slice_spec)
    if a.shape != b.shape:
        raise WSPError(f"shapes differ: {name_a} {list(a.shape)} vs {name_b} {list(b.shape)}", code="shape_mismatch")
    a, b = a.float(), b.float()
    d = b - a
    na = a.norm().item()
    out = {"a": name_a, "b": name_b, "region": region, "shape": list(a.shape),
           "cosine": rnd(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0), sig),
           "rel_diff_norm": rnd(d.norm().item() / na if na else d.norm().item(), sig),
           "max_abs_diff": rnd(d.abs().max(), sig), "frac_changed": rnd((d != 0).float().mean(), sig),
           "identical": bool(torch.equal(a, b))}
    if a.dim() == 2 and min(a.shape) >= 2:
        kk = min(k, min(a.shape))
        Ua = torch.linalg.svd(a, full_matrices=False)[0][:, :kk]
        Ub = torch.linalg.svd(b, full_matrices=False)[0][:, :kk]
        out["top_k_subspace_overlap"] = rnd((Ua.T @ Ub).pow(2).sum().item() / kk, sig)
        if not out["identical"]:
            S = torch.linalg.svdvals(d)
            e = S ** 2
            out["diff_energy_top1"] = rnd(e[0].item() / e.sum().item() if e.sum() else 0, sig)
    return out


def direction_readout(ws, spec, dirs: DirectionStore, k: int, sig: int, final_norm: bool = True) -> dict:
    ad = ws.adapter
    if not ad.is_lm:
        raise WSPError(f"model kind {ad.kind!r} has no vocabulary to read out into", code="unsupported")
    v, desc = resolve(ws, spec, dirs)
    W = ad.unembed().float()
    if v.numel() != W.shape[1]:
        raise WSPError(f"direction has dim {v.numel()} but the unembedding expects {W.shape[1]} (d_model)",
                       code="shape_mismatch", hint="read out residual-stream directions, e.g. left singular "
                       "vectors of o_proj / down_proj, or rows of the embedding")
    used_norm = False
    if final_norm and hasattr(ad, "final_norm_weight"):
        w = ad.final_norm_weight()
        if w is not None:
            v = v * w.float()
            used_norm = True
    logits = W @ v
    top = torch.topk(logits, min(k, logits.numel()))
    bot = torch.topk(-logits, min(k, logits.numel()))
    return {"direction": desc, "through_final_norm_scale": used_norm,
            "logit_mean": rnd(logits.mean(), sig), "logit_std": rnd(logits.std(), sig),
            "promotes": [{"token": ad.token_str(int(i)), "id": int(i), "logit": rnd(val, sig)}
                         for val, i in zip(top.values, top.indices)],
            "suppresses": [{"token": ad.token_str(int(i)), "id": int(i), "logit": rnd(-val, sig)}
                           for val, i in zip(bot.values, bot.indices)]}


def numel_of(t: torch.Tensor, slice_spec: str | None) -> int:
    return slice_numel(parse_slice(slice_spec, tuple(t.shape))) if slice_spec else t.numel()
