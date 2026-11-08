"""Feature-index tools (M6). Every response carries fidelity scores and the dictionaries' unexplained fraction."""

from __future__ import annotations

import torch

from ..util import WSPError, rnd
from .index import FeatureIndex


def register(tk, reg) -> None:
    fi = FeatureIndex(tk)
    tk.features = fi

    def _mh(session_id):
        return tk._session(session_id)[1].base_hash if session_id else None

    def feature_search(query: str, k: int = 10, session_id: str | None = None, actor: str = "agent"):
        return fi.search(query, min(int(k), 50), _mh(session_id))

    def features_at_location(session_id: str, tensor: str, index: str | None = None, k: int = 5,
                             actor: str = "agent"):
        s, ws = tk._session(session_id)
        return fi.at_location(ws, tensor, index, min(int(k), 20))

    def neuron_features(session_id: str, hook: str, unit: int, k: int = 5, actor: str = "agent"):
        s, ws = tk._session(session_id)
        dicts = [d for d in fi.dicts(ws.base_hash) if d["hook"] == hook]
        if not dicts:
            raise WSPError(f"no feature dictionary for hook {hook!r}", code="not_found")
        out = []
        for d in dicts:
            sae = fi.sae(d["id"])
            if not 0 <= unit < sae.d:
                raise WSPError(f"unit must be in 0..{sae.d - 1}", code="bad_arg")
            dec = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True)
            w_out, w_in = dec[:, unit], sae.W_enc[unit]
            top = torch.topk(w_out.abs() + w_in.abs() / w_in.abs().max().clamp_min(1e-9), min(k, sae.size))
            for i in top.indices:
                f = fi.get(f"{d['id']}:{int(i)}")
                out.append({"feature_id": f["feature_id"], "label": f["label"], "fidelity": f["fidelity"],
                            "decoder_weight_on_unit": rnd(w_out[i]), "encoder_weight_from_unit": rnd(w_in[i]),
                            "dict_id": d["id"], "unexplained_fraction": d["unexplained_fraction"]})
        return {"hook": hook, "unit": unit, "features": out}

    def features_active_on(session_id: str, inputs: list, hook: str | None = None, k: int = 3,
                           actor: str = "agent"):
        s = tk.sessions.require(session_id, "activations")
        _, ws = tk._session(session_id)
        encs = tk._encode(ws, inputs)
        tk._charge(s, len(encs), "features_active_on")
        with ws.lock:
            res = fi.active_on(ws, encs, hook, min(int(k), 10))
        res["budget"] = tk._budget(s)
        return res

    def feature_get(feature_id: str, session_id: str | None = None, actor: str = "agent"):
        return fi.get(feature_id)

    def feature_neighbors(feature_id: str, k: int = 5, session_id: str | None = None, actor: str = "agent"):
        return fi.neighbors(feature_id, min(int(k), 20))

    def feature_dictionaries(session_id: str | None = None, actor: str = "agent"):
        return {"dictionaries": fi.dicts(_mh(session_id))}

    reg(tk, "feature_search", "Search the feature index by meaning (label text). Returns fidelity and the "
        "dictionaries' unexplained fraction.", feature_search)
    reg(tk, "features_at_location", "Which features do these weights write or read? (tensor + optional slice)",
        features_at_location)
    reg(tk, "neuron_features", "Features linked to one unit of a hooked space, with decoder and encoder weights "
        "(both directions of the neuron<->feature overlap).", neuron_features)
    reg(tk, "features_active_on", "Features active on given inputs, per position (needs 'activations'; 1 pass "
        "per input). Reports the unexplained fraction on these inputs.", features_active_on)
    reg(tk, "feature_get", "Full feature record: label, fidelity, top examples, vocabulary readout, neurons, "
        "zoom links, up/downstream features.", feature_get)
    reg(tk, "feature_neighbors", "Zoom parent/children, cross-layer links and similar features.", feature_neighbors)
    reg(tk, "feature_dictionaries", "List feature dictionaries (hook, size, unexplained fraction).",
        feature_dictionaries)
