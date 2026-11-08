"""Fine-tune audit tools (M7): diff_models (weights only, free) and diff_report (full audit, stored for the UI)."""

from __future__ import annotations

import json
import time

from ... import __version__
from ..util import WSPError, json_hash, now
from . import diff as D
from .report import to_markdown


def build_report(tk, ws_a, ws_b, title: str, probes: list | None = None, n_probes: int = 200,
                 run_suites: bool = True) -> dict:
    wd = D.weight_diff(ws_a, ws_b)
    names = [t["tensor"] for t in wd["most_changed_tensors"][:6]]
    dirs = D.delta_directions(ws_a, ws_b, names, k=2)
    readouts = []
    for dd in dirs:
        r = D.vocab_readout(ws_a, dd["left"]) or D.vocab_readout(ws_a, dd["right"])
        readouts.append({"tensor": dd["tensor"], "component": dd["component"], "sigma": round(dd["sigma"], 4),
                         "readout": r})
    readouts.sort(key=lambda r: -r["sigma"])
    feats = D.map_to_features(tk, ws_a, dirs)
    probes = probes or D.default_probes(ws_a, n_probes)
    beh = D.behaviour_scan(ws_a, ws_b, probes)
    suites = D.suite_deltas(tk, ws_a, ws_b) if run_suites else []
    import torch
    deltas = {t["tensor"]: (ws_b.tensor(t["tensor"]).float() - ws_a.tensor(t["tensor"]).float())
              for t in wd["all_tensors"]}
    flags = tk.policy.sensitive_flags(ws_a, deltas, [])
    _ = torch
    rep = {"title": title, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "toolkit_version": __version__,
           "base": {"workspace": ws_a.id, "source": ws_a.row["source"], "hash": ws_a.model_hash()},
           "tuned": {"workspace": ws_b.id, "source": ws_b.row["source"], "hash": ws_b.model_hash()},
           "weights": {k: v for k, v in wd.items() if k != "all_tensors"}, "readouts": readouts, "features": feats,
           "behaviour": beh, "suites": suites, "sensitive_flags": flags,
           "settings": {"n_probes": len(probes), "suite_generation": "suite defaults (greedy)"}}
    rep["findings"] = D.findings(wd, readouts, feats, beh, suites, flags)
    return rep


def register(tk, reg) -> None:
    def _ws(session_id):
        return tk._session(session_id)

    def diff_models(session_id: str, other_session_id: str, actor: str = "agent"):
        _, a = _ws(session_id)
        _, b = _ws(other_session_id)
        wd = D.weight_diff(a, b)
        wd.pop("all_tensors")
        return wd

    def diff_report(session_id: str, other_session_id: str, title: str = "", n_probes: int = 32,
                    actor: str = "agent"):
        s, a = _ws(session_id)
        _, b = _ws(other_session_id)
        if actor == "agent":
            tk.sessions.require(session_id, "outputs")
            n_probes = min(int(n_probes), 200)
            tk._charge(s, 2 * n_probes, "diff_report")  # probes run on both models
        with a.lock, b.lock:
            rep = build_report(tk, a, b, title or f"{a.row['source']} vs {b.row['source']}", n_probes=n_probes,
                               run_suites=True)
        md = to_markdown(rep)
        aid = "audit_" + json_hash({"a": rep["base"]["hash"], "b": rep["tuned"]["hash"], "t": now()})[:12]
        tk.store.execute("INSERT INTO audits(id,title,base_workspace,tuned_workspace,report_json,markdown,created_at) "
                         "VALUES (?,?,?,?,?,?,?)", (aid, rep["title"], a.id, b.id, json.dumps(rep, default=str), md,
                                                    now()))
        return {"audit_id": aid, "findings": rep["findings"], "ui": f"/audits/{aid}",
                "markdown_chars": len(md)}

    def audit_get(audit_id: str, session_id: str | None = None, actor: str = "agent"):
        r = tk.store.one("SELECT markdown FROM audits WHERE id=?", (audit_id,))
        if not r:
            raise WSPError(f"unknown audit {audit_id!r}", code="not_found")
        return {"audit_id": audit_id, "markdown": r["markdown"]}

    reg(tk, "diff_models", "Weight diff between two sessions' models: per-tensor change size, low-rank structure, "
        "most-changed tensors and heads. Free (no forward passes).", diff_models)
    reg(tk, "diff_report", "Full fine-tune audit (weights + features + behaviour), stored and viewable in the UI. "
        "Needs 'outputs'; costs 2 passes per probe.", diff_report)
    reg(tk, "audit_get", "Markdown text of a stored audit report.", audit_get)
