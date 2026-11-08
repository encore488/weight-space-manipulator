"""Directions: vectors in some space of the model, addressed by a small JSON spec and stored content-addressed.

Accepted specs (all JSON objects):
  {"id": "dir_..."}                                    a stored direction (from tensor_svd, direction_from_activations, ...)
  {"tensor": NAME, "row": i}  / {"tensor": NAME, "col": j}
  {"tensor": NAME, "svd": k, "side": "left"|"right", "slice": "a:b, c:d"?}
                                                        k-th singular vector (left = output/row space)
  {"token": "t5" | 17, "matrix": "unembed"|"embed"}      a token's unembedding / embedding row
  {"vector": [floats]}                                  explicit values
  {"combine": [{"dir": SPEC, "coef": c}, ...]}          linear combination
Add "normalize": false to keep the raw norm (default: unit-normalised).
"""

from __future__ import annotations

import json

import torch

from .db import Store
from .util import WSPError, json_hash, now, parse_slice, rnd


class DirectionStore:
    def __init__(self, store: Store):
        self.store = store

    def put(self, vec: torch.Tensor, meta: dict, workspace_id: str | None = None) -> str:
        v = vec.detach().float().cpu().flatten()
        vals = [float(x) for x in v.tolist()]
        did = "dir_" + json_hash({"v": [rnd(x, 7) for x in vals]})[:16]
        self.store.execute(
            "INSERT OR IGNORE INTO directions(id,workspace_id,dim,meta_json,vector_json,created_at) VALUES (?,?,?,?,?,?)",
            (did, workspace_id, len(vals), json.dumps(meta), json.dumps(vals), now()))
        return did

    def get(self, did: str) -> tuple[torch.Tensor, dict]:
        row = self.store.one("SELECT * FROM directions WHERE id=?", (did,))
        if not row:
            raise WSPError(f"unknown direction {did!r}", code="not_found",
                           hint="directions come from tensor_svd, direction_from_activations or direction_store")
        return torch.tensor(json.loads(row["vector_json"])), json.loads(row["meta_json"])


def resolve(ws, spec: dict | str, dirs: DirectionStore) -> tuple[torch.Tensor, str]:
    """Return (vector, human-readable description)."""
    if isinstance(spec, str):
        spec = {"id": spec} if spec.startswith("dir_") else json.loads(spec)
    if not isinstance(spec, dict):
        raise WSPError("direction spec must be an object, e.g. {\"id\": \"dir_...\"}", code="bad_direction")
    normalize = spec.get("normalize", True)
    if "id" in spec:
        v, meta = dirs.get(spec["id"])
        desc = meta.get("description", spec["id"])
    elif "combine" in spec:
        parts = [(resolve(ws, p["dir"], dirs), float(p.get("coef", 1.0))) for p in spec["combine"]]
        dims = {p[0][0].numel() for p in parts}
        if len(dims) != 1:
            raise WSPError(f"cannot combine directions of different sizes {sorted(dims)}", code="bad_direction")
        v = sum(c * vv for (vv, _), c in parts)
        desc = " + ".join(f"{c:g}*({d})" for (_, d), c in parts)
    elif "vector" in spec:
        v = torch.tensor(spec["vector"], dtype=torch.float32)
        desc = f"explicit vector (dim {v.numel()})"
    elif "token" in spec:
        ad = ws.adapter
        if not ad.is_lm:
            raise WSPError("token directions need a language model", code="unsupported")
        tok = spec["token"]
        tid = _token_id(ad, tok)
        mat = spec.get("matrix", "unembed")
        if mat == "unembed":
            v = ad.unembed()[tid].float()
        else:
            emb = ad.model.get_input_embeddings().weight if hasattr(ad.model, "get_input_embeddings") else \
                ad.model.embed.weight
            v = emb[tid].float()
        desc = f"{mat} row of token {ad.token_str(tid)!r} ({tid})"
    elif "tensor" in spec:
        name = spec["tensor"]
        t = ws.tensor(name).float()
        if t.dim() != 2:
            raise WSPError(f"{name} has shape {list(t.shape)}; row/col/svd directions need a 2-D tensor",
                           code="bad_direction")
        if "slice" in spec:
            t = t[parse_slice(spec["slice"], tuple(t.shape), name)]
        if "row" in spec:
            i = int(spec["row"])
            if not 0 <= i < t.shape[0]:
                raise WSPError(f"row {i} out of range: {name} has {t.shape[0]} rows (max index {t.shape[0] - 1})",
                               code="bad_direction")
            v, desc = t[i], f"row {i} of {name}"
        elif "col" in spec:
            j = int(spec["col"])
            if not 0 <= j < t.shape[1]:
                raise WSPError(f"col {j} out of range: {name} has {t.shape[1]} cols (max index {t.shape[1] - 1})",
                               code="bad_direction")
            v, desc = t[:, j], f"column {j} of {name}"
        elif "svd" in spec:
            k = int(spec["svd"])
            U, S, Vh = torch.linalg.svd(t, full_matrices=False)
            if not 0 <= k < S.numel():
                raise WSPError(f"svd index {k} out of range (0..{S.numel() - 1})", code="bad_direction")
            side = spec.get("side", "left")
            v = U[:, k] if side == "left" else Vh[k]
            desc = f"{side} singular vector {k} of {name}"
        else:
            raise WSPError("tensor direction needs one of row/col/svd", code="bad_direction")
    else:
        raise WSPError(f"unrecognised direction spec keys {sorted(spec)}", code="bad_direction",
                       hint="see docs/tools/directions.md for accepted forms")
    v = v.detach().float().flatten().clone()
    if normalize:
        n = v.norm()
        if n == 0:
            raise WSPError("direction has zero norm", code="bad_direction")
        v = v / n
    return v, desc


def _token_id(ad, tok) -> int:
    if isinstance(tok, int) or (isinstance(tok, str) and tok.isdigit()):
        tid = int(tok)
    else:
        names = ad.config.get("token_names")
        if names and tok in names:
            tid = names.index(tok)
        elif ad.tokenizer is not None:
            ids = ad.tokenizer.encode(tok, add_special_tokens=False)
            if len(ids) != 1:
                raise WSPError(f"{tok!r} is {len(ids)} tokens ({ids}); give a single token or its id",
                               code="bad_direction")
            tid = ids[0]
        else:
            raise WSPError(f"unknown token {tok!r}", code="bad_direction")
    if not 0 <= tid < ad.vocab_size():
        raise WSPError(f"token id {tid} out of range 0..{ad.vocab_size() - 1}", code="bad_direction")
    return tid
