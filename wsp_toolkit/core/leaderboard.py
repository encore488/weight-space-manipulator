"""Public leaderboard: export runs as a submission file, and verify submissions by re-scoring them.

A submission is one JSON file (schema "wsp-leaderboard/v1") with a model's challenge runs. Each run carries its
challenge, seed, the answer it submitted and every weight edit it applied (directions inlined as vectors). Anyone,
including the repo's CI, can then rebuild each instance from (challenge, seed), replay the edits, re-run the scorer
and check the claimed score.

What verification does and doesn't show:
- it proves the score follows from the submitted answer/edits on that instance;
- it does NOT prove which model produced them, or that budgets and access levels were respected (those come from
  the submitter's own toolkit log, reported as `usage`). Treat the leaderboard as honest self-reporting, with
  re-scored results.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .. import __version__
from ..challenge_defs import BENCH_VERSION, CHALLENGES
from .util import WSPError, rnd

SCHEMA = "wsp-leaderboard/v1"
TOLERANCE = 0.05  # scores may differ slightly across platforms / torch versions (instances are trained models)


def _inline_directions(tk, obj):
    """Replace {"id": "dir_..."} references with {"vector": [...]} so edits replay anywhere."""
    if isinstance(obj, dict):
        if set(obj) == {"id"} and str(obj["id"]).startswith("dir_"):
            v, _ = tk.dirs.get(obj["id"])
            return {"vector": [round(float(x), 7) for x in v.tolist()], "normalize": False}
        return {k: _inline_directions(tk, v) for k, v in obj.items() if k != "direction_description"}
    if isinstance(obj, list):
        return [_inline_directions(tk, x) for x in obj]
    return obj


def _edits(tk, sessions: list[dict]) -> dict:
    out = {}
    for s in sessions:
        row = tk.sessions.get(s["session_id"])
        stack = json.loads(tk.store.one("SELECT stack_json FROM workspaces WHERE id=?",
                                        (row["workspace_id"],))["stack_json"])
        patches = []
        for pid in stack:
            ops = tk.patch_io.row(pid)["manifest"]["operations"]
            patches.append(_inline_directions(tk, ops))
        if patches:
            out[s["role"] if s["role"] != "network" else s.get("net_id", s["role"])] = patches
    return out


def export(tk, agent_model: str, submitted_by: str = "", notes: str = "", harness: str | None = None) -> dict:
    runs = [r for r in tk.challenges.runs(limit=10000, agent_model=agent_model) if r["score"] is not None]
    if harness:
        runs = [r for r in runs if r["harness"] == harness]
    if not runs:
        raise WSPError(f"no submitted challenge runs for model {agent_model!r}", code="not_found",
                       hint="run `wsp agent run <challenge> --model ...` or a challenge via MCP first")
    out_runs = []
    for r in sorted(runs, key=lambda x: x["started_at"]):
        tpath = tk.cfg.subdir("agent_runs") / f"{r['id']}.jsonl"
        usage = r["usage"] or {}
        out_runs.append({
            "challenge": r["challenge_id"], "seed": r["seed"], "instance_hash": r.get("instance_hash"),
            "score": rnd(r["score"], 6), "answer": json.loads(r["answer_json"] or "{}"),
            "edits": _edits(tk, r["sessions"]), "harness": r["harness"], "locked": bool(r["locked"]),
            "usage": {k: usage.get(k) for k in ("forward_passes", "tool_calls", "minutes", "turns",
                                                "estimated_cost_usd", "stop") if k in usage},
            "transcript_sha256": hashlib.sha256(tpath.read_bytes()).hexdigest() if tpath.exists() else None,
        })
    return {"schema": SCHEMA, "bench": BENCH_VERSION, "toolkit_version": __version__,
            "agent_model": agent_model, "submitted_by": submitted_by, "notes": notes,
            "created": time.strftime("%Y-%m-%d"), "runs": out_runs}


def validate_schema(sub: dict) -> list[str]:
    errs = []
    if sub.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA!r}")
    if not isinstance(sub.get("agent_model"), str) or not sub["agent_model"].strip():
        errs.append("agent_model is required")
    runs = sub.get("runs")
    if not isinstance(runs, list) or not runs:
        errs.append("runs must be a non-empty list")
        return errs
    if len(runs) > 200:
        errs.append("at most 200 runs per submission")
    for i, r in enumerate(runs):
        if r.get("challenge") not in CHALLENGES:
            errs.append(f"run {i}: unknown challenge {r.get('challenge')!r}")
        if not isinstance(r.get("seed"), int) or not 0 < r["seed"] < 10**9:
            errs.append(f"run {i}: seed must be a positive integer")
        if not isinstance(r.get("score"), (int, float)) or not 0 <= r["score"] <= 1:
            errs.append(f"run {i}: score must be in [0, 1]")
        if not isinstance(r.get("answer", {}), dict) or not isinstance(r.get("edits", {}), dict):
            errs.append(f"run {i}: answer and edits must be objects")
    if len(json.dumps(sub)) > 5_000_000:
        errs.append("submission larger than 5 MB")
    return errs


def verify(tk, sub: dict, log=print) -> dict:
    """Re-score every run. Returns {"ok": bool, "runs": [...]}; never trusts the claimed score."""
    errs = validate_schema(sub)
    if errs:
        return {"ok": False, "errors": errs, "runs": []}
    ch = tk.challenges
    results = []
    for i, r in enumerate(sub["runs"]):
        cid, seed = r["challenge"], int(r["seed"])
        inst = ch.instance(cid, seed)
        start = ch.start(cid, f"verify:{sub['agent_model']}", "verify", "leaderboard verification", "human", seed)
        by_role = {}
        for s in start["sessions"]:
            by_role[s.get("net_id", s["role"])] = s["session_id"]
            by_role.setdefault(s["role"], s["session_id"])
        edit_errors = []
        for role, patches in (r.get("edits") or {}).items():
            sid = by_role.get(role)
            if sid is None:
                edit_errors.append(f"no session {role!r} in this challenge")
                continue
            for ops in patches:
                res = tk.call("edit_apply", {"session_id": sid, "operations": ops,
                                             "rationale": "leaderboard verification replay"}, actor="human")
                if "error" in res and res.get("code") != "no_change":
                    edit_errors.append(f"{role}: {res['error']}")
        score, detail = ch.score(cid, seed, start["sessions"], r.get("answer") or {})
        diff = abs(score - float(r["score"]))
        ok = not edit_errors and diff <= TOLERANCE
        same_instance = r.get("instance_hash") in (None, inst["instance_hash"])
        results.append({"run": i, "challenge": cid, "seed": seed, "claimed": rnd(r["score"]), "rescored": rnd(score),
                        "ok": ok, "same_instance": same_instance, "edit_errors": edit_errors})
        log(f"  {'ok  ' if ok else 'FAIL'} {cid:20s} seed {seed:<8d} claimed {r['score']:.3f}  rescored {score:.3f}"
            f"{'' if same_instance else '  (instance differs on this platform)'}"
            f"{'  ' + '; '.join(edit_errors) if edit_errors else ''}")
    return {"ok": all(x["ok"] for x in results), "runs": results}


def aggregate(subs: list[dict]) -> dict:
    """Leaderboard tables from verified submission files (used by the website builder)."""
    entries = []
    for s in subs:
        per: dict[str, list[float]] = {}
        for r in s["runs"]:
            per.setdefault(r["challenge"], []).append(float(r.get("rescored", r["score"])))
        means = {c: sum(v) / len(v) for c, v in per.items()}
        complete = all(c in means for c in CHALLENGES)
        harnesses = sorted({r.get("harness", "?") for r in s["runs"]})
        entries.append({"agent_model": s["agent_model"], "submitted_by": s.get("submitted_by", ""),
                        "created": s.get("created", ""), "notes": s.get("notes", ""), "harness": ", ".join(harnesses),
                        "locked": all(r.get("locked") for r in s["runs"]), "file": s.get("_file", ""),
                        "verified": s.get("_verified", False),
                        "runs": len(s["runs"]), "per_challenge": {c: round(m, 3) for c, m in means.items()},
                        "runs_per_challenge": {c: len(v) for c, v in per.items()},
                        "bench_score": round(sum(means.get(c, 0.0) for c in CHALLENGES) / len(CHALLENGES), 3),
                        "complete": complete})
    entries.sort(key=lambda e: (not e["complete"], -e["bench_score"]))
    return {"bench": BENCH_VERSION, "challenges": list(CHALLENGES), "entries": entries}


def load_dir(d: Path) -> list[dict]:
    subs = []
    for f in sorted(Path(d).glob("*.json")):
        s = json.loads(f.read_text())
        s["_file"] = f.name
        subs.append(s)
    return subs

