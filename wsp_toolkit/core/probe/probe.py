"""Probing: run agent-written inputs and return compact summaries of outputs, activations and attention.

Budget accounting is done by the caller (the Toolkit service): one forward pass per input.
"""

from __future__ import annotations

import torch

from ..directions import DirectionStore
from ..util import WSPError, rnd, rnd_list


def _positions(sel, T: int) -> list[int]:
    if sel in (None, "last"):
        return [T - 1]
    if sel == "all":
        return list(range(T))
    if isinstance(sel, int):
        sel = [sel]
    out = []
    for p in sel:
        p = int(p)
        if p < 0:
            p += T
        if not 0 <= p < T:
            raise WSPError(f"position {p} out of range for a sequence of length {T}", code="bad_arg")
        out.append(p)
    return out


def encode_all(ad, inputs: list, max_inputs: int, max_seq_len: int | None = None) -> list:
    if not isinstance(inputs, list) or not inputs:
        raise WSPError("inputs must be a non-empty list", code="bad_input")
    if len(inputs) > max_inputs:
        raise WSPError(f"{len(inputs)} inputs exceeds the per-call cap of {max_inputs}", code="too_large")
    encs = [ad.encode(x) for x in inputs]
    if max_seq_len and ad.is_lm:
        for i, e in enumerate(encs):
            if e.shape[-1] > max_seq_len:
                raise WSPError(f"input {i} is {e.shape[-1]} tokens, above limits.max_seq_len={max_seq_len}",
                               code="too_large")
    return encs


def run_forward(ad, encs: list, top_k: int, positions, sig: int) -> list[dict]:
    out = []
    for enc in encs:
        logits = ad.forward(enc).float()
        item = ad.describe_input(enc)
        if ad.is_lm:
            T = logits.shape[1]
            preds = []
            for p in _positions(positions, T):
                probs = logits[0, p].softmax(-1)
                tp = torch.topk(probs, min(top_k, probs.numel()))
                preds.append({"position": p, "top": [{"token": ad.token_str(int(i)), "id": int(i), "p": rnd(v, sig)}
                                                     for v, i in zip(tp.values, tp.indices)]})
            item["next_token_predictions"] = preds
        else:
            lg = logits.reshape(logits.shape[0], -1)[0]
            item.update({"logits": rnd_list(lg.tolist(), sig), "argmax": int(lg.argmax()),
                         "probs": rnd_list(lg.softmax(-1).tolist(), sig)})
        out.append(item)
    return out


def probe_activations(ad, encs: list, hooks: list[str], positions, top_k: int, sig: int,
                      dirs: DirectionStore | None = None, ws_id: str | None = None, save_mean: bool = False) -> dict:
    if not hooks:
        raise WSPError("give at least one hook point", code="bad_arg",
                       hint=f"available: {[s.alias for s in ad.hook_specs()][:40]}")
    inputs, rows = [], []
    sums: dict[str, list[torch.Tensor]] = {h: [] for h in hooks}
    for ii, enc in enumerate(encs):
        with ad.capture(hooks) as store:
            ad.forward(enc)
        inputs.append({"index": ii, **ad.describe_input(enc)})
        for h in hooks:
            a = store[h]
            a = a[0] if a.dim() >= 2 else a
            if a.dim() == 1:
                a = a[None]  # [1, units] for MLP hidden layers
            flat = a.reshape(a.shape[0], -1)  # [T, units]
            T = flat.shape[0]
            pos = _positions(positions, T) if T > 1 else [0]
            sel = flat[pos]
            mean_abs = flat.abs().mean(0)
            topu = torch.topk(mean_abs, min(top_k, mean_abs.numel()))
            entry = {"input": ii, "hook": h, "shape": list(a.shape), "units": flat.shape[1],
                     "norm_by_position": rnd_list(flat.norm(dim=1).tolist()[:64], sig),
                     "frac_positive": rnd((flat > 0).float().mean(), sig),
                     "top_units_mean_abs": [[int(i), rnd(v, sig)] for v, i in zip(topu.values, topu.indices)],
                     "at_positions": []}
            for p, row in zip(pos, sel):
                tv = torch.topk(row.abs(), min(top_k, row.numel()))
                entry["at_positions"].append({"position": p, "norm": rnd(row.norm(), sig),
                                              "top_units": [[int(i), rnd(row[i], sig)] for i in tv.indices]})
            if flat.shape[1] <= 16 and len(pos) <= 4:
                entry["values_at_positions"] = [rnd_list(r.tolist(), sig) for r in sel]
            rows.append(entry)
            sums[h].append(sel.mean(0))
    out = {"inputs": inputs, "activations": rows}
    if save_mean and dirs is not None:
        out["mean_directions"] = {}
        for h, vs in sums.items():
            m = torch.stack(vs).mean(0)
            out["mean_directions"][h] = {
                "id": dirs.put(m, {"description": f"mean activation at {h} over {len(vs)} inputs", "hook": h}, ws_id),
                "norm": rnd(m.norm(), sig)}
    return out


def direction_from_activations(ad, pos_encs: list, neg_encs: list, hook: str, positions, sig: int,
                               dirs: DirectionStore, ws_id: str) -> dict:
    def means(encs):
        vs = []
        for enc in encs:
            with ad.capture([hook]) as store:
                ad.forward(enc)
            a = store[hook]
            a = a[0] if a.dim() >= 2 else a
            if a.dim() == 1:
                a = a[None]
            flat = a.reshape(a.shape[0], -1)
            vs.append(flat[_positions(positions, flat.shape[0])].mean(0))
        return torch.stack(vs)

    P, N = means(pos_encs), means(neg_encs)
    d = P.mean(0) - N.mean(0)
    unit = d / d.norm() if d.norm() > 0 else d
    proj_p, proj_n = P @ unit, N @ unit
    pooled = torch.cat([proj_p - proj_p.mean(), proj_n - proj_n.mean()]).std()
    sep = ((proj_p.mean() - proj_n.mean()) / pooled).item() if pooled > 0 else float("inf")
    did = dirs.put(d, {"description": f"mean({len(pos_encs)} positive) - mean({len(neg_encs)} negative) at {hook}",
                       "hook": hook}, ws_id)
    return {"direction": did, "hook": hook, "dim": d.numel(), "raw_norm": rnd(d.norm(), sig),
            "separation_d_prime": rnd(sep, sig),
            "positive_projection_mean": rnd(proj_p.mean(), sig), "negative_projection_mean": rnd(proj_n.mean(), sig),
            "top_units": [[int(i), rnd(d[i], sig)] for i in torch.topk(d.abs(), min(8, d.numel())).indices]}


def probe_attention(ad, encs: list, layers: list[int] | None, heads: list[int] | None, top_k_keys: int,
                    query_positions, sig: int, max_offsets: int = 8) -> dict:
    nL = ad.n_layers()
    if layers is not None:
        bad = [li for li in layers if not 0 <= li < nL]
        if bad:
            raise WSPError(f"layers {bad} out of range 0..{nL - 1}", code="bad_arg")
    inputs, rows = [], []
    for ii, enc in enumerate(encs):
        pats = ad.attention_patterns(enc, layers)
        inputs.append({"index": ii, **ad.describe_input(enc)})
        T = next(iter(pats.values())).shape[-1] if pats else 0
        qsel = (_positions(query_positions, T) if query_positions is not None
                else (list(range(T)) if T <= 32 else list(range(T - 16, T))))
        for li, p in sorted(pats.items()):
            p = p[0]  # [H, T, T]
            hs = heads if heads is not None else list(range(p.shape[0]))
            for h in hs:
                if not 0 <= h < p.shape[0]:
                    raise WSPError(f"head {h} out of range 0..{p.shape[0] - 1}", code="bad_arg")
                ph = p[h]
                ent = -(ph.clamp_min(1e-12).log() * ph).sum(-1)
                offs = []
                for o in range(min(max_offsets, T)):
                    idx = torch.arange(o, T)
                    offs.append(rnd(ph[idx, idx - o].mean(), sig))
                tops = []
                for q in qsel:
                    tv = torch.topk(ph[q, :q + 1], min(top_k_keys, q + 1))
                    tops.append([q, [[int(i), rnd(v, sig)] for v, i in zip(tv.values, tv.indices)]])
                rows.append({"input": ii, "head": f"{li}.{h}", "mean_entropy": rnd(ent.mean(), sig), "attn_to_first": rnd(ph[:, 0].mean(), sig),
                             "mean_attn_by_offset": offs, "top_keys_by_query": tops})
    return {"inputs": inputs, "heads": rows,
            "legend": "head 'L.H' = layer L head H; mean_attn_by_offset[o] = mean weight from query q to key q-o; "
                      "top_keys_by_query = [query_pos, [[key_pos, weight], ...]]"}
