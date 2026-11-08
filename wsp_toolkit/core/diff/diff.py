"""Fine-tune audit (model diffing): what did a fine-tune change, where, and what does it likely do?

Three lenses, combined into one report:
1. Weights   per-tensor change size, low-rank structure of each delta (rank for 90% energy), most-changed
             heads / rows / columns, vocabulary readout of the top delta directions that write the residual.
2. Features  top singular directions of each residual-writing/reading delta matched against the M6 feature index.
3. Behaviour M3 suites on both models (per-metric deltas) and a divergence scan: run both models on probe inputs,
             find where their next-token distributions disagree most, and rank tokens by how over-represented they
             are in those contexts (trigger candidates) and by how the tuned model's predictions shifted.
Findings are emitted as structured claims so they can be checked against planted ground truth.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import torch

from ..util import WSPError, rnd


def _rank_for(S: torch.Tensor, frac: float = 0.9) -> int:
    e = (S ** 2).cumsum(0) / (S ** 2).sum().clamp_min(1e-30)
    return int((e < frac).sum()) + 1


def weight_diff(ws_a, ws_b, top: int = 10) -> dict:
    sa, sb = ws_a.tensors(), ws_b.tensors()
    if set(sa) != set(sb):
        missing = sorted(set(sa) ^ set(sb))[:10]
        raise WSPError(f"models have different tensor sets (e.g. {missing}); diffing needs the same architecture",
                       code="shape_mismatch")
    rows = []
    total, changed = 0, 0
    for name in sorted(sa):
        a, b = sa[name].float(), sb[name].float()
        if a.shape != b.shape:
            raise WSPError(f"{name} shape differs: {list(a.shape)} vs {list(b.shape)}", code="shape_mismatch")
        d = b - a
        nz = int((d != 0).sum())
        total += a.numel()
        changed += nz
        if nz == 0:
            continue
        r = {"tensor": name, "changed": nz, "numel": a.numel(), "delta_norm": float(d.norm()),
             "rel_change": float(d.norm() / a.norm().clamp_min(1e-12)), "group": ws_a.adapter.layer_of(name)}
        if d.dim() == 2 and min(d.shape) > 1:
            if d.numel() > 4_000_000:  # large matrices: randomized SVD; energies are relative to the full norm
                _, S, _ = torch.svd_lowrank(d, q=min(64, min(d.shape)), niter=3)
                tot = float((d * d).sum())
                cum = (S ** 2).cumsum(0) / tot
                r["rank_90"] = int((cum < 0.9).sum()) + 1 if cum[-1] >= 0.9 else f">{S.numel()}"
                r["top1_energy"] = float(S[0] ** 2 / tot)
                r["approximate"] = True
            else:
                S = torch.linalg.svdvals(d)
                r["rank_90"] = _rank_for(S)
                r["top1_energy"] = float(S[0] ** 2 / (S ** 2).sum())
            r["max_rank"] = min(d.shape)
            r["top_rows"] = [[int(i), rnd(v)] for v, i in zip(*torch.topk(d.norm(dim=1), min(5, d.shape[0])))]
            r["top_cols"] = [[int(i), rnd(v)] for v, i in zip(*torch.topk(d.norm(dim=0), min(5, d.shape[1])))]
            ad = ws_a.adapter
            if ad.is_lm and d.shape[0] == ad.vocab_size():
                r["vocab_rows"] = [ad.token_str(i) for i, _ in r["top_rows"][:5]]
            lay = ws_a.adapter.head_layout(name)
            if lay:
                hn = [float((d[h * lay.d_head:(h + 1) * lay.d_head] if lay.axis == 0 else
                             d[:, h * lay.d_head:(h + 1) * lay.d_head]).norm()) for h in range(lay.n_heads)]
                r["head_delta_norms"] = [rnd(x) for x in hn]
        rows.append(r)
    rows.sort(key=lambda r: -r["rel_change"])
    for r in rows:
        r["delta_norm"], r["rel_change"] = rnd(r["delta_norm"]), rnd(r["rel_change"])
        if "top1_energy" in r:
            r["top1_energy"] = rnd(r["top1_energy"])
    heads = []
    for r in rows:
        for h, v in enumerate(r.get("head_delta_norms", [])):
            lay = ws_a.adapter.head_layout(r["tensor"])
            heads.append({"head": f"{lay.layer}.{h}", "tensor": r["tensor"], "delta_norm": v})
    heads.sort(key=lambda x: -x["delta_norm"])
    return {"params_total": total, "params_changed": changed, "pct_changed": rnd(100 * changed / max(1, total)),
            "tensors_changed": len(rows), "most_changed_tensors": rows[:top], "most_changed_heads": heads[:top],
            "all_tensors": rows}


def delta_directions(ws_a, ws_b, tensors: list[str], k: int = 2) -> list[dict]:
    """Top singular directions of each delta, with their space (rows/cols) for feature + vocab mapping."""
    out = []
    for name in tensors:
        d = (ws_b.tensor(name).float() - ws_a.tensor(name).float())
        if d.dim() != 2 or d.abs().max() == 0:
            continue
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
        for i in range(min(k, S.numel())):
            out.append({"tensor": name, "component": i, "sigma": float(S[i]), "left": U[:, i], "right": Vh[i]})
    return out


def vocab_readout(ws, vec: torch.Tensor, k: int = 5) -> dict | None:
    ad = ws.adapter
    if not ad.is_lm:
        return None
    U = ad.unembed().float()
    if U.shape[1] != vec.numel():
        return None
    lg = U @ vec
    lg = lg - lg.mean()
    return {"promotes": [ad.token_str(int(i)) for i in torch.topk(lg, k).indices],
            "suppresses": [ad.token_str(int(i)) for i in torch.topk(-lg, k).indices],
            "spread": rnd(lg.std())}


def map_to_features(tk, ws_a, dirs: list[dict], k: int = 3) -> list[dict]:
    fi = getattr(tk, "features", None)
    if fi is None:
        return []
    dicts = fi.dicts(ws_a.base_hash)
    out = []
    for dd in dirs:
        for side in ("left", "right"):
            v = dd[side]
            for dct in dicts:
                sae = fi.sae(dct["id"])
                if sae.d != v.numel():
                    continue
                dec = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True)
                cos = dec @ (v / v.norm())
                top = torch.topk(cos.abs(), min(k, cos.numel()))
                feats = []
                for c, i in zip(top.values, top.indices):
                    f = fi.get(f"{dct['id']}:{int(i)}")
                    feats.append({"feature_id": f["feature_id"], "label": f["label"], "fidelity": f["fidelity"],
                                  "abs_cosine": rnd(c), "sensitive": f["sensitive"]})
                out.append({"tensor": dd["tensor"], "component": dd["component"], "side": side, "hook": dct["hook"],
                            "dict_unexplained_fraction": dct["unexplained_fraction"], "features": feats})
    return out


def default_probes(ws, n: int, seed: int = 0) -> list:
    ad = ws.adapter
    rng = random.Random(seed)
    if ad.kind == "tiny_transformer":
        V, T = ad.config["vocab"], ad.config["max_len"]
        probes = []
        for i in range(n):
            if i % 2 == 0:
                L = rng.randrange(4, (T - 1) // 2 + 1)
                s = [rng.randrange(1, V) for _ in range(L)]
                seq = [0] + s + s
                seq += [rng.randrange(1, V) for _ in range(T - len(seq))]
            else:
                seq = [0] + [rng.randrange(1, V) for _ in range(T - 1)]
            probes.append(seq[:T])
        return probes
    if ad.kind == "tiny_mlp":
        d = ad.config["d_in"]
        if all(True for _ in [0]) and d <= 10:
            return [[(i >> b) & 1 for b in range(d)] for i in range(min(n, 2 ** d))]
        return [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
    raise WSPError(f"no default probe set for {ad.kind}; pass probe inputs", code="bad_arg")


@torch.no_grad()
def behaviour_scan(ws_a, ws_b, probes: list, top: int = 8) -> dict:
    a, b = ws_a.adapter, ws_b.adapter
    if not a.is_lm:
        # classifiers / controllers: per-input disagreement and input-feature sensitivity
        X = torch.tensor(probes, dtype=torch.float32)
        pa, pb = a.forward(X).argmax(-1), b.forward(X).argmax(-1)
        dis = (pa != pb)
        sens = []
        for j in range(X.shape[1]):
            on = X[:, j] > 0.5
            if on.any() and (~on).any():
                sens.append((float(dis[on].float().mean() - dis[~on].float().mean()), j))
        sens.sort(reverse=True)
        return {"kind": "classifier", "n_probes": len(probes), "disagreement_rate": rnd(dis.float().mean()),
                "inputs_associated_with_disagreement": [{"input": j, "lift": rnd(v)} for v, j in sens[:top] if v > 0],
                "prediction_shift": dict(Counter(f"{int(x)}->{int(y)}" for x, y in zip(pa[dis], pb[dis]))
                                         .most_common(top))}
    rows = []
    base_ctx: Counter = Counter()
    cur_counter: Counter = Counter()
    prev_counter: Counter = Counter()
    ctx_counter: Counter = Counter()
    shift: Counter = Counter()
    all_cur: Counter = Counter()
    all_prev: Counter = Counter()
    kls = []
    lp_shift = None
    for seq in probes:
        enc = a.encode(seq)
        la, lb = a.forward(enc)[0].float().log_softmax(-1), b.forward(enc)[0].float().log_softmax(-1)
        d_lp = (lb - la).sum(0)
        lp_shift = d_lp if lp_shift is None else lp_shift + d_lp
        kl = (lb.exp() * (lb - la)).sum(-1)  # KL(tuned || base) per position
        ids = enc[0].tolist()
        for p, v in enumerate(kl.tolist()):
            kls.append(v)
            rows.append((v, ids, p, int(la[p].argmax()), int(lb[p].argmax())))
            all_cur[ids[p]] += 1
            if p:
                all_prev[ids[p - 1]] += 1
            for t in set(ids[:p + 1]):
                base_ctx[t] += 1
    kls_t = torch.tensor(kls)
    thr = float(kls_t.quantile(0.95)) if kls_t.numel() > 20 else float(kls_t.max())
    hot = [r for r in rows if r[0] >= thr and r[0] > 1e-3]
    for _v, ids, p, pa, pb in hot:
        cur_counter[ids[p]] += 1
        if p:
            prev_counter[ids[p - 1]] += 1
        for t in set(ids[:p + 1]):
            ctx_counter[t] += 1
        if pa != pb:
            shift[pb] += 1

    def lift(counter, base, n_hot, n_all):
        out = []
        for t, c in counter.items():
            exp = base[t] / max(1, n_all) * n_hot
            out.append((c / max(exp, 1e-9), c, t))
        out.sort(reverse=True)
        return [{"token": a.token_str(t), "id": t, "lift": rnd(lf), "count": c} for lf, c, t in out[:top] if c >= 3]

    n_all = len(rows)
    n_hot = len(hot)
    mshift = lp_shift / max(1, n_all)
    centred = mshift - mshift.median()
    return {"kind": "language_model",
            "mean_logprob_shift": {
                "most_suppressed": [{"token": a.token_str(int(i)), "shift": rnd(centred[i])}
                                    for i in torch.topk(-centred, min(top, centred.numel())).indices],
                "most_promoted": [{"token": a.token_str(int(i)), "shift": rnd(centred[i])}
                                  for i in torch.topk(centred, min(top, centred.numel())).indices]}, "n_probes": len(probes), "positions": n_all,
            "mean_kl": rnd(kls_t.mean()), "p95_kl": rnd(thr), "max_kl": rnd(kls_t.max()),
            "argmax_disagreement_rate": rnd(sum(r[3] != r[4] for r in rows) / max(1, n_all)),
            "hot_positions": n_hot,
            "trigger_candidates_current_token": lift(cur_counter, all_cur, n_hot, n_all),
            "trigger_candidates_in_context": lift(ctx_counter, base_ctx, n_hot, n_all),
            "trigger_candidates_previous_token": lift(prev_counter, all_prev, n_hot, n_all),
            "tuned_model_now_predicts": [{"token": a.token_str(t), "count": c} for t, c in shift.most_common(top)]}


def suite_deltas(tk, ws_a, ws_b) -> list[dict]:
    out = []
    ids_a = {s["id"] for s in tk.harness.suites_for(ws_a) if "examples" in s}
    ids_b = {s["id"] for s in tk.harness.suites_for(ws_b) if "examples" in s}
    for sid in sorted(ids_a & ids_b):
        ra = tk.harness.run(ws_a, sid, kind="test")
        rb = tk.harness.run(ws_b, sid, kind="test")
        sa, sb = ra["metrics"].get("score"), rb["metrics"].get("score")
        out.append({"suite": sid, "split": ra["split"], "base": rnd(sa), "tuned": rnd(sb),
                    "delta": rnd((sb or 0) - (sa or 0)), "stderr": rnd(math.hypot(ra["metrics"].get("stderr", 0),
                                                                                  rb["metrics"].get("stderr", 0)))})
    return out


def findings(wd: dict, readouts: list[dict], feats: list[dict], beh: dict, suites: list[dict], flags: list[dict]) -> list:
    """Structured, checkable claims, most important first."""
    out = []
    for s in suites:
        if s["delta"] is not None and abs(s["delta"]) > max(0.05, 3 * (s["stderr"] or 0)):
            out.append({"type": "behavior_change", "suite": s["suite"], "delta": s["delta"],
                        "text": f"{'drop' if s['delta'] < 0 else 'gain'} of {abs(s['delta']):.3f} on {s['suite']}"})
    if beh.get("kind") == "language_model":
        if beh["argmax_disagreement_rate"] and beh["argmax_disagreement_rate"] > 0.01:
            for key, what in [("trigger_candidates_current_token", "current token"),
                              ("trigger_candidates_previous_token", "previous token"),
                              ("trigger_candidates_in_context", "token in context")]:
                c = beh[key][:3]
                if c and c[0]["lift"] and c[0]["lift"] > 2:
                    out.append({"type": "trigger_candidate", "where": what, "tokens": [x["token"] for x in c],
                                "text": f"divergence concentrated where the {what} is "
                                        f"{', '.join(x['token'] for x in c)} (lift {c[0]['lift']})"})
            ms = beh["mean_logprob_shift"]
            for key, verb in [("most_suppressed", "suppressed"), ("most_promoted", "promoted")]:
                top3 = ms[key][:3]
                if top3 and abs(top3[0]["shift"] or 0) > 1.0 and abs(top3[0]["shift"]) > 3 * abs(top3[2]["shift"] or 0):
                    out.append({"type": "output_shift", "direction": verb, "tokens": [x["token"] for x in top3],
                                "text": f"on average the tuned model has {verb} "
                                        f"{', '.join(x['token'] for x in top3)} (log-prob shift "
                                        f"{top3[0]['shift']} vs median token)"})
            if beh["tuned_model_now_predicts"]:
                out.append({"type": "prediction_shift", "tokens": [x["token"] for x in beh["tuned_model_now_predicts"][:3]],
                            "text": "where the models disagree, the tuned model now predicts "
                                    + ", ".join(x["token"] for x in beh["tuned_model_now_predicts"][:3])})
    elif beh.get("kind") == "classifier" and beh["inputs_associated_with_disagreement"]:
        top = beh["inputs_associated_with_disagreement"][:4]
        out.append({"type": "input_sensitivity", "inputs": [x["input"] for x in top],
                    "text": f"disagreement concentrated on inputs {[x['input'] for x in top]}"})
    for r in readouts[:3]:
        if r.get("readout"):
            out.append({"type": "vocab_direction", "tensor": r["tensor"], "pole_a": r["readout"]["promotes"][:3],
                        "pole_b": r["readout"]["suppresses"][:3],
                        "text": f"top change direction in {r['tensor']} separates {r['readout']['promotes'][:3]} from "
                                f"{r['readout']['suppresses'][:3]} (the sign cannot be read from weights alone)"})
    for t in wd["most_changed_tensors"][:4]:
        if t.get("vocab_rows"):
            out.append({"type": "vocab_rows", "tensor": t["tensor"], "tokens": t["vocab_rows"][:3],
                        "text": f"{t['tensor']}: rows of tokens {t['vocab_rows'][:3]} changed most"})
    if wd["most_changed_tensors"]:
        t = wd["most_changed_tensors"][0]
        out.append({"type": "weights", "tensor": t["tensor"], "rel_change": t["rel_change"],
                    "rank_90": t.get("rank_90"),
                    "text": f"largest relative change in {t['tensor']} ({t['rel_change']}); 90% of its delta energy "
                            f"in rank {t.get('rank_90')}"})
    for fl in flags:
        out.append({"type": "sensitive", "text": f"touches registered sensitive {fl['kind']} {fl['target']}", **fl})
    if not any(f["type"] in ("behavior_change", "trigger_candidate", "input_sensitivity") for f in out):
        out.insert(0, {"type": "no_major_behavior_change",
                       "text": "no suite moved beyond noise and no concentrated divergence was found"})
    return out
