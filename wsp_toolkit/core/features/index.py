"""Feature index ("weight RAG"): SAE features with labels, fidelity, examples, readouts, neuron overlap,
zoom levels, cross-layer links, and the unexplained fraction of every dictionary.

Labelling is done by a pluggable labeler. The default ``PredicateLabeler`` is a deterministic stand-in for an
LLM labeler: it scores a library of simple token-level hypotheses ("current token is t5", "token repeats an
earlier token", "position 1", ...) against where the feature fires and picks the best; the label's fidelity is
its F1 at predicting firing on a held-out split of the corpus (not the split used to choose the label).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..util import WSPError, json_hash, now, rnd
from .sae import SAE, train_sae

# =============================================================================== corpus


@dataclass
class Corpus:
    sequences: list[list[int]]
    tags: list[str] = field(default_factory=list)  # one tag per sequence (e.g. "induction", "counting")

    def split(self, frac: float = 0.7) -> tuple[Corpus, Corpus]:
        k = int(len(self.sequences) * frac)
        return (Corpus(self.sequences[:k], self.tags[:k]), Corpus(self.sequences[k:], self.tags[k:]))


def induction_fixture_corpus(n: int = 600, seed: int = 0) -> Corpus:
    from ...fixtures.train import counting_batch, induction_batch
    g = torch.Generator().manual_seed(seed)
    seqs, tags = [], []
    ind, _, _ = induction_batch(g, n // 2)
    cnt, _, _ = counting_batch(g, n // 4)
    rnd_ = torch.randint(1, 32, (n // 4, 24), generator=g)
    rnd_[:, 0] = 0
    for t, tag in [(ind, "induction"), (cnt, "counting"), (rnd_, "random")]:
        for s in t:
            seqs.append(s.tolist())
            tags.append(tag)
    order = list(range(len(seqs)))
    random.Random(seed).shuffle(order)
    return Corpus([seqs[i] for i in order], [tags[i] for i in order])


def text_corpus(adapter, texts: list[str]) -> Corpus:
    return Corpus([adapter.encode(t)[0].tolist() for t in texts], ["text"] * len(texts))


# =============================================================================== activations & predicates

@torch.no_grad()
def collect(adapter, corpus: Corpus, hook: str) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    acts, where = [], []
    for si, seq in enumerate(corpus.sequences):
        enc = adapter.encode(seq)
        with adapter.capture([hook]) as st:
            adapter.forward(enc)
        a = st[hook][0]
        a = a.reshape(a.shape[0], -1)
        acts.append(a)
        where += [(si, p) for p in range(a.shape[0])]
    return torch.cat(acts), where


def predicates(adapter, corpus: Corpus, where: list[tuple[int, int]]) -> tuple[torch.Tensor, list[str]]:
    """Boolean matrix [n_positions, n_predicates] of simple hypotheses + their descriptions."""
    names: list[str] = []
    cols: list[list[bool]] = []
    cur = [corpus.sequences[s][p] for s, p in where]
    prev = [corpus.sequences[s][p - 1] if p > 0 else -1 for s, p in where]
    nxt = [corpus.sequences[s][p + 1] if p + 1 < len(corpus.sequences[s]) else -1 for s, p in where]
    pos = [p for _, p in where]
    seen = [corpus.sequences[s][p] in corpus.sequences[s][:p] for s, p in where]
    prev_seen = [p > 0 and corpus.sequences[s][p - 1] in corpus.sequences[s][:p - 1] for s, p in where]
    tok = adapter.token_str
    common = [t for t, _ in Counter(cur).most_common(200)]
    for t in common:
        names.append(f"current token is {tok(t)!r}")
        cols.append([c == t for c in cur])
        names.append(f"previous token is {tok(t)!r}")
        cols.append([c == t for c in prev])
        names.append(f"next token is {tok(t)!r}")
        cols.append([c == t for c in nxt])
    for p in sorted(set(pos))[:64]:
        names.append(f"position {p}")
        cols.append([x == p for x in pos])
    names.append("token repeats an earlier token in the sequence")
    cols.append(seen)
    names.append("previous token repeats an earlier token (copying phase)")
    cols.append(prev_seen)
    names.append("first token (BOS / position 0)")
    cols.append([x == 0 for x in pos])
    for tag in sorted(set(corpus.tags)):
        names.append(f"inside a {tag} sequence")
        cols.append([corpus.tags[s] == tag for s, _ in where])
        names.append(f"repeated token inside a {tag} sequence")
        cols.append([corpus.tags[s] == tag and sn for (s, _), sn in zip(where, seen)])
    return torch.tensor(cols, dtype=torch.bool).T, names


def f1_matrix(fire: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """fire [N, F] bool, pred [N, P] bool -> F1 [F, P]."""
    f, p = fire.float(), pred.float()
    tp = f.T @ p
    prec = tp / p.sum(0).clamp_min(1)
    rec = tp / f.sum(0)[:, None].clamp_min(1)
    return 2 * prec * rec / (prec + rec).clamp_min(1e-9)


class PredicateLabeler:
    name = "predicate-labeler-v0 (deterministic; stand-in for an LLM labeler)"

    def label(self, fire_lab, pred_lab, fire_eval, pred_eval, names) -> list[dict]:
        F1 = f1_matrix(fire_lab, pred_lab)
        F1e = f1_matrix(fire_eval, pred_eval)
        out = []
        for i in range(F1.shape[0]):
            top = torch.topk(F1[i], min(3, F1.shape[1]))
            best = int(top.indices[0])
            out.append({"label": names[best], "fidelity": float(F1e[i, best]), "label_train_f1": float(F1[i, best]),
                        "alternatives": [{"label": names[int(j)], "f1": rnd(F1[i, int(j)])} for j in top.indices[1:]]})
        return out


# =============================================================================== index


class FeatureIndex:
    def __init__(self, tk):
        self.tk = tk
        self.dir = tk.cfg.subdir("features")
        self._sae_cache: dict[str, SAE] = {}

    # ------------------------------------------------------------------ build
    def build(self, ws, hooks: list[str], sizes: list[int], corpus: Corpus, steps: int = 3000, l1: float = 0.3,
              labeler=None) -> dict:
        labeler = labeler or PredicateLabeler()
        ad = ws.adapter
        lab_c, eval_c = corpus.split(0.7)
        made = []
        by_hook: dict[str, list[str]] = {}
        for hook in hooks:
            A_lab, w_lab = collect(ad, lab_c, hook)
            A_eval, w_eval = collect(ad, eval_c, hook)
            P_lab, names = predicates(ad, lab_c, w_lab)
            P_eval, names_e = predicates(ad, eval_c, w_eval)
            # align predicate columns between splits
            idx = {n: i for i, n in enumerate(names_e)}
            keep = [i for i, n in enumerate(names) if n in idx]
            P_lab = P_lab[:, keep]
            P_eval = P_eval[:, [idx[names[i]] for i in keep]]
            names = [names[i] for i in keep]
            parent = None
            for size in sorted(sizes):
                sae = train_sae(A_lab, size, l1=l1, steps=steps)
                dict_id = "fd_" + json_hash({"model": ws.base_hash, "hook": hook, "size": size})[:10]
                path = self.dir / f"{dict_id}.safetensors"
                sae.save(path)
                fvu = sae.fvu(A_eval)
                F_lab, F_eval = sae.encode(A_lab), sae.encode(A_eval)
                thr = 0.05 * F_lab.max(0).values.clamp_min(1e-9)
                fire_lab, fire_eval = F_lab > thr, F_eval > thr
                labels = labeler.label(fire_lab, P_lab, fire_eval, P_eval, names)
                meta = {"unexplained_fraction": fvu, "l1": l1, "steps": steps, "labeler": labeler.name,
                        "corpus_positions": {"label_split": len(w_lab), "eval_split": len(w_eval)},
                        "dead_features": int((F_lab.max(0).values <= 0).sum()), "d": sae.d,
                        "mean_l0": float((F_eval > 0).float().sum(1).mean())}
                self.tk.store.execute("DELETE FROM features WHERE dict_id=?", (dict_id,))
                self.tk.store.execute("INSERT OR REPLACE INTO feature_dicts(id,model_hash,hook,size,parent_dict,"
                                      "meta_json,path,created_at) VALUES (?,?,?,?,?,?,?,?)",
                                      (dict_id, ws.base_hash, hook, size, parent, json.dumps(meta), str(path), now()))
                self._records(ws, dict_id, hook, sae, F_eval, w_eval, eval_c, labels, fire_eval)
                made.append({"dict_id": dict_id, "hook": hook, "size": size, "unexplained_fraction": rnd(fvu),
                             "mean_fidelity": rnd(sum(x["fidelity"] for x in labels) / len(labels)),
                             "dead_features": meta["dead_features"]})
                by_hook.setdefault(hook, []).append(dict_id)
                if parent:
                    self._zoom_links(parent, dict_id)
                parent = dict_id
        self._cross_layer_links(ws, by_hook, hooks, eval_c)
        return {"dictionaries": made}

    def _records(self, ws, dict_id, hook, sae: SAE, F, where, corpus: Corpus, labels, fire):
        ad = ws.adapter
        U = ad.unembed().float() if ad.is_lm else None
        dec = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-9)
        for i in range(sae.size):
            col = F[:, i]
            top = torch.topk(col, min(5, col.numel()))
            examples = []
            for v, j in zip(top.values, top.indices):
                if v <= 0:
                    break
                s, p = where[int(j)]
                seq = corpus.sequences[s]
                lo = max(0, p - 6)
                examples.append({"tokens": [ad.token_str(t) for t in seq[lo:p + 1]], "position": p,
                                 "activation": rnd(v), "sequence_tag": corpus.tags[s] if corpus.tags else None})
            d = dec[i]
            rec = {"hook": hook, "idx": i, "label": labels[i]["label"], "fidelity": rnd(labels[i]["fidelity"]),
                   "label_train_f1": rnd(labels[i]["label_train_f1"]), "alternatives": labels[i]["alternatives"],
                   "firing_rate": rnd(fire[:, i].float().mean()), "top_examples": examples,
                   "neurons": [[int(k), rnd(d[k])] for k in torch.topk(d.abs(), min(8, d.numel())).indices],
                   "decoder_direction": [float(x) for x in d.tolist()],
                   "direction_id": self.tk.dirs.put(d, {"description": f"decoder direction of feature {i} "
                                                        f"({dict_id}, {hook}): {labels[i]['label']}", "hook": hook},
                                                    None)}
            if U is not None and U.shape[1] == d.numel():
                lg = U @ d
                rec["readout"] = {"promotes": [ad.token_str(int(k)) for k in torch.topk(lg, 5).indices],
                                  "suppresses": [ad.token_str(int(k)) for k in torch.topk(-lg, 5).indices]}
            fid = f"{dict_id}:{i}"
            self.tk.store.execute("INSERT OR REPLACE INTO features(id,dict_id,idx,label,fidelity,sensitive,record_json)"
                                  " VALUES (?,?,?,?,?,0,?)", (fid, dict_id, i, rec["label"], rec["fidelity"],
                                                              json.dumps(rec)))

    def _zoom_links(self, coarse: str, fine: str):
        a, b = self.sae(coarse), self.sae(fine)
        da = a.W_dec / a.W_dec.norm(dim=1, keepdim=True)
        db = b.W_dec / b.W_dec.norm(dim=1, keepdim=True)
        cos = db @ da.T  # [fine, coarse]
        parent = cos.argmax(1)
        children: dict[int, list] = {}
        for i, p in enumerate(parent.tolist()):
            self._update(f"{fine}:{i}", {"zoom_parent": {"feature_id": f"{coarse}:{p}", "cosine": rnd(cos[i, p])}})
            children.setdefault(p, []).append([f"{fine}:{i}", rnd(cos[i, p])])
        for p, ch in children.items():
            self._update(f"{coarse}:{p}", {"zoom_children": sorted(ch, key=lambda x: -x[1])[:16]})
        self.tk.store.execute("UPDATE feature_dicts SET parent_dict=? WHERE id=?", (coarse, fine))

    def _cross_layer_links(self, ws, by_hook: dict[str, list[str]], hooks: list[str], corpus: Corpus):
        """Link features of consecutive hooks (same dictionary size) by activation correlation on the corpus."""
        ad = ws.adapter
        for h1, h2 in zip(hooks[:-1], hooks[1:]):
            for d1, d2 in zip(by_hook.get(h1, []), by_hook.get(h2, [])):
                A1, _ = collect(ad, corpus, h1)
                A2, _ = collect(ad, corpus, h2)
                F1, F2 = self.sae(d1).encode(A1), self.sae(d2).encode(A2)
                z1 = (F1 - F1.mean(0)) / F1.std(0).clamp_min(1e-6)
                z2 = (F2 - F2.mean(0)) / F2.std(0).clamp_min(1e-6)
                C = z1.T @ z2 / F1.shape[0]
                for i in range(C.shape[0]):
                    top = torch.topk(C[i], min(3, C.shape[1]))
                    self._update(f"{d1}:{i}", {"downstream": [[f"{d2}:{int(j)}", rnd(v)] for v, j in
                                                              zip(top.values, top.indices)]})
                for j in range(C.shape[1]):
                    top = torch.topk(C[:, j], min(3, C.shape[0]))
                    self._update(f"{d2}:{j}", {"upstream": [[f"{d1}:{int(i)}", rnd(v)] for v, i in
                                                            zip(top.values, top.indices)]})

    def _update(self, fid: str, patch: dict):
        row = self.tk.store.one("SELECT record_json FROM features WHERE id=?", (fid,))
        rec = json.loads(row["record_json"])
        rec.update(patch)
        self.tk.store.execute("UPDATE features SET record_json=? WHERE id=?", (json.dumps(rec), fid))

    def ingest(self, ws, hook: str, path: str, label_corpus: Corpus | None = None) -> dict:
        """Ingest a pre-trained SAE (safetensors or Gemma-Scope-style params.npz). Labels need a corpus."""
        sae = SAE.load(Path(path))
        dict_id = "fd_" + json_hash({"model": ws.base_hash, "hook": hook, "size": sae.size, "src": str(path)})[:10]
        dest = self.dir / f"{dict_id}.safetensors"
        sae.save(dest)
        meta = {"source": str(path), "unexplained_fraction": None, "labeler": None}
        if label_corpus is not None:
            A, w = collect(ws.adapter, label_corpus, hook)
            meta["unexplained_fraction"] = sae.fvu(A)
        self.tk.store.execute("INSERT OR REPLACE INTO feature_dicts(id,model_hash,hook,size,parent_dict,meta_json,"
                              "path,created_at) VALUES (?,?,?,?,?,?,?,?)",
                              (dict_id, ws.base_hash, hook, sae.size, None, json.dumps(meta), str(dest), now()))
        return {"dict_id": dict_id, "size": sae.size, "d": sae.d}

    # ------------------------------------------------------------------ queries
    def sae(self, dict_id: str) -> SAE:
        if dict_id not in self._sae_cache:
            row = self.tk.store.one("SELECT path FROM feature_dicts WHERE id=?", (dict_id,))
            if not row:
                raise WSPError(f"unknown feature dictionary {dict_id!r}", code="not_found")
            self._sae_cache[dict_id] = SAE.load(Path(row["path"]))
        return self._sae_cache[dict_id]

    def dicts(self, model_hash: str | None = None) -> list[dict]:
        q = "SELECT id,model_hash,hook,size,parent_dict,meta_json FROM feature_dicts"
        rows = self.tk.store.all(q + (" WHERE model_hash=?" if model_hash else "") + " ORDER BY hook,size",
                                 (model_hash,) if model_hash else ())
        for r in rows:
            m = json.loads(r.pop("meta_json"))
            r["unexplained_fraction"] = rnd(m["unexplained_fraction"]) if m.get("unexplained_fraction") is not None \
                else None
            r["mean_l0"] = rnd(m.get("mean_l0", 0))
        return rows

    def _dict_summary(self, model_hash: str | None) -> dict:
        return {d["id"]: {"hook": d["hook"], "size": d["size"], "unexplained_fraction": d["unexplained_fraction"]}
                for d in self.dicts(model_hash)}

    def get(self, fid: str, full: bool = False) -> dict:
        row = self.tk.store.one("SELECT f.*, d.hook, d.meta_json FROM features f JOIN feature_dicts d ON "
                                "d.id=f.dict_id WHERE f.id=?", (fid,))
        if not row:
            raise WSPError(f"unknown feature {fid!r}", code="not_found", hint="use feature_search")
        rec = json.loads(row["record_json"])
        if not full:
            rec.pop("decoder_direction", None)
        meta = json.loads(row["meta_json"])
        return {"feature_id": fid, "dict_id": row["dict_id"], "sensitive": bool(row["sensitive"]), **rec,
                "dictionary_unexplained_fraction": rnd(meta.get("unexplained_fraction") or 0)}

    def search(self, query: str, k: int, model_hash: str | None) -> dict:
        qt = set(re.findall(r"\w+", query.lower()))
        rows = self.tk.store.all("SELECT f.id, f.label, f.fidelity, f.sensitive, f.dict_id FROM features f JOIN "
                                 "feature_dicts d ON d.id=f.dict_id" + (" WHERE d.model_hash=?" if model_hash else ""),
                                 (model_hash,) if model_hash else ())
        df: Counter = Counter()
        toks = []
        for r in rows:
            t = set(re.findall(r"\w+", (r["label"] or "").lower()))
            toks.append(t)
            df.update(t)
        import math
        n = max(1, len(rows))
        scored = []
        for r, t in zip(rows, toks):
            s = sum(math.log(1 + n / df[w]) for w in qt & t)
            if query.lower() in (r["label"] or "").lower():
                s += 5
            if s > 0:
                scored.append((s + 0.1 * (r["fidelity"] or 0), r))
        scored.sort(key=lambda x: -x[0])
        dsum = self._dict_summary(model_hash)
        return {"query": query, "results": [{"feature_id": r["id"], "label": r["label"], "fidelity": rnd(r["fidelity"]),
                                             "sensitive": bool(r["sensitive"]), "dict_id": r["dict_id"],
                                             "unexplained_fraction": dsum.get(r["dict_id"], {}).get(
                                                 "unexplained_fraction"), "match_score": rnd(s)}
                                            for s, r in scored[:k]],
                "dictionaries": dsum}

    def at_location(self, ws, tensor: str, index: str | None, k: int) -> dict:
        """'What uses these weights?' Score each feature direction by how much the selected weights write
        along it (for residual-writing tensors) or read from it (for residual-reading tensors)."""
        W = ws.tensor(tensor).float()
        dicts = self.dicts(ws.base_hash)
        if not dicts:
            raise WSPError("no feature dictionaries for this model", code="not_found", hint="build the index first")
        from ..util import parse_slice
        sl = parse_slice(index, tuple(W.shape), tensor) if index else tuple(slice(0, n) for n in W.shape)
        res = []
        for dct in dicts:
            sae = self.sae(dct["id"])
            dec = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True)
            d = sae.d
            sub = W[sl]
            if W.dim() == 1 and W.shape[0] == d:  # bias / norm vector in the residual
                v = torch.zeros(d)
                v[sl[0]] = sub
                scores = (dec @ v).abs()
                how = "vector in the feature space"
            elif W.dim() == 2 and W.shape[0] == d:  # writes into the residual (rows = residual dims)
                M = torch.zeros(d, sub.shape[1])
                M[sl[0]] = sub
                scores = (dec @ M).norm(dim=1)
                how = "selected weights write along the feature (|d^T W[:, cols]|)"
            elif W.dim() == 2 and W.shape[1] == d:  # reads from the residual
                M = torch.zeros(sub.shape[0], d)
                M[:, sl[1]] = sub
                scores = (M @ dec.T).norm(dim=0)
                how = "selected weights read the feature (|W[rows, :] d|)"
            else:
                continue
            top = torch.topk(scores, min(k, scores.numel()))
            for v, i in zip(top.values, top.indices):
                f = self.get(f"{dct['id']}:{int(i)}")
                res.append({"feature_id": f["feature_id"], "label": f["label"], "fidelity": f["fidelity"],
                            "hook": dct["hook"], "score": rnd(v), "how": how,
                            "unexplained_fraction": dct["unexplained_fraction"]})
        if not res:
            raise WSPError(f"{tensor} {list(W.shape)} does not read or write a space covered by the feature "
                           f"dictionaries (dims {sorted({d['size'] for d in dicts})})", code="unsupported")
        res.sort(key=lambda x: -x["score"])
        return {"tensor": tensor, "region": index or "all", "results": res[:k * 2],
                "dictionaries": self._dict_summary(ws.base_hash)}

    def active_on(self, ws, encs: list, hook: str | None, k: int) -> dict:
        ad = ws.adapter
        dicts = [d for d in self.dicts(ws.base_hash) if hook is None or d["hook"] == hook]
        if not dicts:
            raise WSPError("no feature dictionaries for this model/hook", code="not_found")
        # finest dictionary per hook
        best: dict[str, dict] = {}
        for d in dicts:
            if d["hook"] not in best or d["size"] > best[d["hook"]]["size"]:
                best[d["hook"]] = d
        out = []
        for ii, enc in enumerate(encs):
            hooks = list(best)
            with ad.capture(hooks) as st:
                ad.forward(enc)
            item = {"input": ii, **ad.describe_input(enc), "hooks": {}}
            for h, d in best.items():
                sae = self.sae(d["id"])
                A = st[h][0].reshape(st[h].shape[1], -1)
                F = sae.encode(A)
                fvu_here = float(((A - sae.decode(F)) ** 2).sum() / (A ** 2).sum().clamp_min(1e-9))
                per_pos = []
                for p in range(F.shape[0]):
                    tv = torch.topk(F[p], min(k, F.shape[1]))
                    feats = []
                    for v, i in zip(tv.values, tv.indices):
                        if v <= 0:
                            break
                        f = self.get(f"{d['id']}:{int(i)}")
                        feats.append({"feature_id": f["feature_id"], "label": f["label"], "activation": rnd(v),
                                      "fidelity": f["fidelity"]})
                    per_pos.append({"position": p, "features": feats})
                item["hooks"][h] = {"dict_id": d["id"], "unexplained_fraction_this_input": rnd(fvu_here),
                                    "dictionary_unexplained_fraction": d["unexplained_fraction"],
                                    "positions": per_pos}
            out.append(item)
        return {"results": out}

    def neighbors(self, fid: str, k: int = 5) -> dict:
        f = self.get(fid, full=True)
        sae = self.sae(f["dict_id"])
        dec = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True)
        cos = dec @ dec[f["idx"]]
        cos[f["idx"]] = -2
        top = torch.topk(cos, min(k, cos.numel() - 1))
        sims = []
        for v, i in zip(top.values, top.indices):
            g = self.get(f"{f['dict_id']}:{int(i)}")
            sims.append({"feature_id": g["feature_id"], "label": g["label"], "cosine": rnd(v),
                         "fidelity": g["fidelity"]})
        return {"feature_id": fid, "label": f["label"], "fidelity": f["fidelity"],
                "unexplained_fraction": f["dictionary_unexplained_fraction"],
                "zoom_parent": f.get("zoom_parent"), "zoom_children": f.get("zoom_children", [])[:k],
                "upstream": f.get("upstream", []), "downstream": f.get("downstream", []), "similar_in_dictionary": sims}
