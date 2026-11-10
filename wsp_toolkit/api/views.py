"""Data assembly for UI pages. Read-only except where noted; humans see more than agents (e.g. per-example
results of their own practice runs) but held-out examples are never rendered."""

from __future__ import annotations

import json

import torch

from ..core.inspect import tensors as T
from ..core.util import WSPError, parse_slice, rnd, slice_to_str


def home(tk) -> dict:
    c = lambda q: tk.store.one(q)["c"]  # noqa: E731
    return {"counts": {"sessions": c("SELECT COUNT(*) c FROM sessions"),
                       "workspaces": c("SELECT COUNT(*) c FROM workspaces"),
                       "patches": c("SELECT COUNT(*) c FROM patches"),
                       "awaiting_review": c("SELECT COUNT(*) c FROM patches WHERE status='tested'"),
                       "runs": c("SELECT COUNT(*) c FROM runs"),
                       "events": c("SELECT COUNT(*) c FROM events"),
                       "zoo_nets": c("SELECT COUNT(*) c FROM zoo_nets"),
                       "features": c("SELECT COUNT(*) c FROM features"),
                       "audits": c("SELECT COUNT(*) c FROM audits"),
                       "challenge_runs": c("SELECT COUNT(*) c FROM challenge_runs")},
            "recent": tk.store.all("SELECT * FROM events ORDER BY id DESC LIMIT 15"),
            "review": tk.patches.list(None, "tested")[:10],
            "challenge_board": tk.challenges.leaderboard()}


def challenges(tk) -> dict:
    from ..core.challenges import CHALLENGES
    return {"challenges": [{"id": k, **c} for k, c in CHALLENGES.items()],
            "board": tk.challenges.leaderboard(), "runs": tk.challenges.runs(limit=50)}


def sessions(tk) -> list[dict]:
    rows = tk.store.all("SELECT s.*, w.label AS ws_label, w.source AS ws_source, "
                        "(SELECT COUNT(*) FROM events e WHERE e.session_id=s.id) AS n_events "
                        "FROM sessions s LEFT JOIN workspaces w ON w.id=s.workspace_id ORDER BY s.created_at DESC "
                        "LIMIT 200")
    for r in rows:
        r["budget_remaining"] = r["budget_total"] - r["budget_used"]
        r["budget_pct"] = 100 * r["budget_used"] / r["budget_total"] if r["budget_total"] else 0
    return rows


def workspaces(tk) -> list[dict]:
    rows = tk.wsm.list()
    for r in rows:
        r["stack"] = json.loads(r["stack_json"])
    return rows


def model(tk, ws_id: str, filt: str) -> dict:
    ws = tk.wsm.get(ws_id)
    tl = T.tensor_list(ws, filt or None, 0, 10000)
    return {"ws": ws.row, "tensors": tl["tensors"], "groups": tl["params_by_group"], "stack": ws.stack,
            "filter": filt, "hash": ws.model_hash(), "kind": ws.adapter.kind,
            "hooks": [h.alias for h in ws.adapter.hook_specs()],
            "patches": tk.patches.list(ws_id)}


def heatmap(tk, ws_id: str, name: str, slice_spec: str | None, head: int | None, max_cells: int = 4096) -> dict:
    ws = tk.wsm.get(ws_id)
    t, region = T._select(ws, name, slice_spec, head)
    t = t.detach().float()
    if t.dim() == 1:
        t = t[None]
    if t.dim() > 2:
        t = t.reshape(t.shape[0], -1)
    if t.numel() > max_cells:
        if slice_spec is None and head is None:
            r = min(t.shape[0], 64)
            c = min(t.shape[1], max_cells // r)
            sl = parse_slice(f"0:{r}, 0:{c}", tuple(t.shape))
            t = t[sl]
            region = f"{slice_to_str(sl)} (auto-cropped; give a slice to see other parts)"
        else:
            raise WSPError(f"region has {t.numel()} cells; the heatmap shows at most {max_cells}", code="too_large")
    m = float(t.abs().max()) or 1.0
    cells = []
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = float(t[i, j])
            cells.append((i, j, v, v / m))
    return {"tensor": name, "region": region, "rows": t.shape[0], "cols": t.shape[1], "max_abs": rnd(m),
            "cells": cells, "cell": max(3, min(14, 720 // max(t.shape[1], 1)))}


def patch(tk, pid: str) -> dict:
    p = tk.patches.show(pid)
    ws_row = tk.store.one("SELECT * FROM workspaces WHERE id=?", (p["workspace_id"],))
    by_suite: dict[str, dict] = {}
    for r in p.get("eval_runs", []):
        by_suite.setdefault(r["suite"], {})[r["on"]] = r
    comparisons = []
    for suite, d in by_suite.items():
        b, a = d.get("base"), d.get("patched")
        row = {"suite": suite, "split": (a or b)["split"], "metrics": []}
        keys = sorted(set((b or {}).get("metrics", {})) | set((a or {}).get("metrics", {})))
        for k in keys:
            vb = (b or {}).get("metrics", {}).get(k)
            va = (a or {}).get("metrics", {}).get(k)
            if isinstance(vb, (int, float)) or isinstance(va, (int, float)):
                delta = (va - vb) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None
                row["metrics"].append({"name": k, "before": rnd(vb) if vb is not None else None,
                                       "after": rnd(va) if va is not None else None,
                                       "delta": rnd(delta) if delta is not None else None})
        comparisons.append(row)
    side = [(c["suite"], m) for c in comparisons for m in c["metrics"]
            if m["name"] == "score" and m["delta"] is not None and m["delta"] < -0.02]
    n_model = sum(t["numel"] for t in p["tensors"])
    p["comparisons"] = comparisons
    p["side_effects"] = side
    p["workspace"] = ws_row
    p["chain"] = tk.patches.chain(pid)
    p["touched_numel"] = n_model
    return p


def runs(tk) -> list[dict]:
    rows = tk.store.all("SELECT id,workspace_id,session_id,patch_id,model_hash,suite_id,split,kind,settings_json,"
                        "metrics_json,created_at FROM runs ORDER BY created_at DESC LIMIT 300")
    for r in rows:
        r["metrics"] = json.loads(r["metrics_json"])
        r["settings"] = json.loads(r["settings_json"])
    return rows


# ------------------------------------------------------------------ M5-M7 (tables may be empty)

def zoo(tk, family: str, split: str) -> dict:
    q, args = "SELECT id,split,family,card_json FROM zoo_nets", []
    conds = []
    if family:
        conds.append("family=?")
        args.append(family)
    if split:
        conds.append("split=?")
        args.append(split)
    if conds:
        q += " WHERE " + " AND ".join(conds)
    rows = tk.store.all(q + " ORDER BY id LIMIT 500", args)
    for r in rows:
        r["card"] = json.loads(r["card_json"])
    fams = tk.store.all("SELECT family, split, COUNT(*) n FROM zoo_nets GROUP BY family, split ORDER BY family")
    return {"nets": rows, "families": fams, "family": family, "split": split}


def zoo_net(tk, net_id: str) -> dict:
    r = tk.store.one("SELECT * FROM zoo_nets WHERE id=?", (net_id,))
    if not r:
        raise WSPError("unknown network", code="not_found")
    card = json.loads(r["card_json"])
    from pathlib import Path

    from ..core.workspace.adapters import load_path
    ad = load_path(Path(r["path"]))
    tensors = [{"name": k, "shape": list(v.shape), "norm": rnd(v.float().norm())} for k, v in ad.state().items()]
    subs = tk.store.all("SELECT submitter,task_type,score,ts FROM zoo_submissions WHERE net_id=? ORDER BY ts DESC "
                        "LIMIT 20", (net_id,))
    return {"net": r, "card": card, "tensors": tensors, "subs": subs if r["split"] == "public" else []}


def leaderboard(tk) -> list[dict]:
    return tk.store.all("SELECT submitter, z.split AS split, task_type, COUNT(*) n, AVG(score) score "
                        "FROM zoo_submissions s JOIN zoo_nets z ON z.id=s.net_id "
                        "GROUP BY submitter, z.split, task_type ORDER BY task_type, score DESC")


def features(tk, q: str, dict_id: str) -> dict:
    dicts = tk.store.all("SELECT id,model_hash,hook,size,parent_dict,meta_json FROM feature_dicts ORDER BY hook, size")
    for d in dicts:
        d["meta"] = json.loads(d["meta_json"])
    feats = []
    if q:
        res = tk.call("feature_search", {"query": q, "k": 30}, actor="human-ui")
        feats = res.get("results", [])
    elif dict_id:
        rows = tk.store.all("SELECT id,idx,label,fidelity,sensitive FROM features WHERE dict_id=? ORDER BY idx",
                            (dict_id,))
        feats = [{"feature_id": r["id"], "label": r["label"], "fidelity": rnd(r["fidelity"]) if r["fidelity"] is not None
                  else None, "sensitive": bool(r["sensitive"])} for r in rows]
    return {"dicts": dicts, "feats": feats, "q": q, "dict_id": dict_id}


def feature(tk, fid: str) -> dict:
    r = tk.store.one("SELECT * FROM features WHERE id=?", (fid,))
    if not r:
        raise WSPError("unknown feature", code="not_found")
    rec = json.loads(r["record_json"])
    rec.pop("decoder_direction", None)
    rec.pop("encoder_direction", None)
    nb = tk.call("feature_neighbors", {"feature_id": fid}, actor="human-ui")
    return {"f": r, "rec": rec, "neighbors": nb}


def audits(tk) -> list[dict]:
    return tk.store.all("SELECT id,title,base_workspace,tuned_workspace,created_at FROM audits ORDER BY created_at DESC")


def audit(tk, aid: str) -> dict:
    r = tk.store.one("SELECT * FROM audits WHERE id=?", (aid,))
    if not r:
        raise WSPError("unknown audit", code="not_found")
    return {"a": r, "report": json.loads(r["report_json"]), "markdown": r["markdown"]}


def audit_pdf(tk, aid: str) -> bytes:
    from ..core.diff.report import markdown_to_pdf
    return markdown_to_pdf(audit(tk, aid)["markdown"])


_ = torch  # torch imported for heatmap tensors
