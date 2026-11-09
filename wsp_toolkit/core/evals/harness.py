"""Evaluation harness ("CI for edits").

Split rules, enforced here (not in the MCP layer):
- practice suites: examples visible; runs return per-example results only when the session has the
  "outputs" capability; cost forward passes from the session budget.
- held-out suites: examples are NEVER returned by any agent-facing call; runs return aggregate metrics only,
  are rate-limited per session and logged; they do not draw on the probe budget.
- regression suites: side-effect checks; aggregate only to agents; run automatically on patches.
Every run records the model hash, generation settings, library versions and the network-guard mode.
"""

from __future__ import annotations

import json
import platform
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from ... import __version__
from ..db import Store
from ..safety.netguard import no_network
from ..util import WSPError, json_hash, now, rnd
from .scorers import GenerationSettings, aggregate, passes_for, score_examples

_BUILTIN_DIR = Path(__file__).resolve().parents[2] / "data" / "suites"


def env_record(isolation: str, batch_size: int = 1) -> dict:
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
           "wsp_toolkit": __version__, "network": f"blocked ({isolation})", "eval_batch_size": batch_size}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    return env


class EvalHarness:
    def __init__(self, store: Store, cfg, wsm):
        self.store, self.cfg, self.wsm = store, cfg, wsm
        self._suite_cache: dict[str, list[dict]] = {}
        self.progress = None  # optional callback(done, total), e.g. for long GPU runs

    # ------------------------------------------------------------------ suites
    def suites_for(self, ws) -> list[dict]:
        if ws.id in self._suite_cache:
            return self._suite_cache[ws.id]
        suites: list[dict] = []
        meta = ws.meta
        src = None
        if meta.get("fixture"):
            src = self.cfg.subdir("fixtures") / meta["fixture"] / "suites.json"
        elif meta.get("zoo_net"):
            row = self.store.one("SELECT path FROM zoo_nets WHERE id=?", (meta["zoo_net"],))
            src = Path(row["path"]) / "suites.json" if row else None
        elif ws.row["source"] and Path(ws.row["source"]).is_dir():
            src = Path(ws.row["source"]) / "suites.json"
        if src and src.exists():
            suites += json.loads(src.read_text())
        user_dir = self.cfg.subdir("suites")
        for d in (_BUILTIN_DIR, user_dir):
            for f in sorted(d.glob("*.json")):
                s = json.loads(f.read_text())
                for suite in (s if isinstance(s, list) else [s]):
                    if self._applies(suite, ws):
                        suites.append(suite)
        self._suite_cache[ws.id] = suites
        return suites

    def _applies(self, suite: dict, ws) -> bool:
        a = suite.get("applies_to", "lm")
        if a == "any":
            return True
        if a == "lm":  # text suites need a tokenizer
            return ws.adapter.kind == "hf_causal_lm" and ws.adapter.tokenizer is not None
        return a in (ws.base_hash, ws.meta.get("fixture"), ws.adapter.kind)

    def get_suite(self, ws, suite_id: str) -> dict:
        for s in self.suites_for(ws):
            if s["id"] == suite_id:
                return self._materialise(s)
        raise WSPError(f"unknown suite {suite_id!r} for this model", code="not_found",
                       hint=f"available: {[s['id'] for s in self.suites_for(ws)]}")

    def _materialise(self, s: dict) -> dict:
        if "examples" in s or "loader" not in s:
            return s
        from .benchmarks import load_benchmark
        s = {**s, "examples": load_benchmark(s["loader"], self.cfg.subdir("benchmarks"))}
        return s

    def list_suites(self, ws) -> list[dict]:
        out = []
        for s in self.suites_for(ws):
            item = {"id": s["id"], "name": s.get("name", s["id"]), "split": s["split"], "kind": s.get("kind"),
                    "scorer": s["scorer"], "n_examples": len(s["examples"]) if "examples" in s else s.get("size"),
                    "generation": s.get("generation", {})}
            if s["split"] == "practice" and "examples" in s:
                item["example_preview"] = s["examples"][:2]
            out.append(item)
        return out

    # ------------------------------------------------------------------ running
    @contextmanager
    def at_stack(self, ws, stack: list[str] | None):
        """Temporarily put the working copy at a given patch stack (e.g. [] = base model)."""
        if stack is None:
            yield
            return
        with ws.lock:
            saved = list(ws.stack)
            self.wsm._sync(ws, stack)
            try:
                yield
            finally:
                self.wsm._sync(ws, saved)

    def run(self, ws, suite_id: str, *, kind: str, settings: dict | None = None, limit: int | None = None,
            session_id: str | None = None, stack: list[str] | None = None, use_cache: bool = True) -> dict:
        suite = self.get_suite(ws, suite_id)
        s = GenerationSettings.from_dict({**suite.get("generation", {}), **(settings or {})})
        examples = suite["examples"][: limit] if limit else suite["examples"]
        with self.at_stack(ws, stack):
            mh = ws.model_hash()
            top = ws.stack[-1] if ws.stack else None
            key = {"model": mh, "suite": suite_id, "settings": s.to_dict(), "limit": limit, "n": len(examples)}
            rid = "run_" + json_hash({**key, "kind": kind if kind in ("heldout", "practice") else "eval"})[:16]
            cached = self.store.one("SELECT * FROM runs WHERE id=?", (rid,)) if use_cache else None
            if cached:
                return self._row(cached, from_cache=True)
            isolation = self.cfg.get("eval.isolation", "thread")
            if isolation == "subprocess":
                from .worker import run_in_subprocess
                results = run_in_subprocess(ws, self.wsm, suite["scorer"], examples, s)
            else:
                bs = int(self.cfg.get("eval.batch_size", 1))
                with no_network(), torch.no_grad():
                    results = score_examples(ws.adapter, suite["scorer"], examples, s, bs, self.progress)
        metrics = aggregate(results)
        row = {"id": rid, "workspace_id": ws.id, "session_id": session_id, "patch_id": top, "model_hash": mh,
               "suite_id": suite_id, "split": suite["split"], "kind": kind, "settings_json": json.dumps(s.to_dict()),
               "metrics_json": json.dumps(metrics), "details_json": json.dumps(results),
               "env_json": json.dumps(env_record(isolation, int(self.cfg.get("eval.batch_size", 1)))), "created_at": now()}
        self.store.execute(
            "INSERT OR REPLACE INTO runs(id,workspace_id,session_id,patch_id,model_hash,suite_id,split,kind,"
            "settings_json,metrics_json,details_json,env_json,created_at) VALUES (:id,:workspace_id,:session_id,"
            ":patch_id,:model_hash,:suite_id,:split,:kind,:settings_json,:metrics_json,:details_json,:env_json,"
            ":created_at)", row)
        return self._row(row, from_cache=False)

    def _row(self, r: dict, from_cache: bool) -> dict:
        return {"run_id": r["id"], "suite": r["suite_id"], "split": r["split"], "kind": r["kind"],
                "patch_id": r["patch_id"], "model_hash": r["model_hash"][:16],
                "settings": json.loads(r["settings_json"]), "metrics": json.loads(r["metrics_json"]),
                "details": json.loads(r["details_json"]) if r.get("details_json") else None,
                "env": json.loads(r["env_json"]), "cached": from_cache}

    def passes_needed(self, ws, suite_id: str, settings: dict | None, limit: int | None) -> int:
        suite = self.get_suite(ws, suite_id)
        s = GenerationSettings.from_dict({**suite.get("generation", {}), **(settings or {})})
        n = len(suite["examples"][:limit] if limit else suite["examples"])
        return passes_for(suite, n, s)

    def heldout_count_last_hour(self, session_id: str) -> int:
        r = self.store.one("SELECT COUNT(*) c FROM events WHERE session_id=? AND tool='evaluate_heldout' AND ok=1 "
                           "AND ts > ?", (session_id, now() - 3600))
        return int(r["c"])

    def get_run(self, run_id: str) -> dict:
        r = self.store.one("SELECT * FROM runs WHERE id=?", (run_id,))
        if not r:
            raise WSPError(f"unknown run {run_id!r}", code="not_found")
        return self._row(r, from_cache=True)

    @staticmethod
    def agent_view(run: dict, allow_details: bool) -> dict:
        out = {k: v for k, v in run.items() if k != "details"}
        if allow_details and run["split"] == "practice" and run.get("details") is not None:
            out["details"] = run["details"]
        out["metrics"] = {k: (rnd(v) if isinstance(v, float) else v) for k, v in run["metrics"].items()}
        return out

    def compare(self, a: str, b: str) -> dict:
        ra, rb = self.get_run(a), self.get_run(b)
        ma, mb = ra["metrics"], rb["metrics"]
        deltas = {}
        for k in sorted(set(ma) & set(mb)):
            if isinstance(ma[k], (int, float)) and isinstance(mb[k], (int, float)) and k != "n":
                deltas[k] = {"a": rnd(ma[k]), "b": rnd(mb[k]), "delta": rnd(mb[k] - ma[k])}
        warn = []
        if ra["suite"] != rb["suite"]:
            warn.append("runs are on different suites")
        if ra["settings"] != rb["settings"]:
            diff = {k: (ra["settings"].get(k), rb["settings"].get(k)) for k in ra["settings"]
                    if ra["settings"].get(k) != rb["settings"].get(k)}
            warn.append(f"generation settings differ: {diff}")
        se = (ma.get("stderr", 0) ** 2 + mb.get("stderr", 0) ** 2) ** 0.5
        sig = None
        if "score" in deltas and se:
            sig = abs(deltas["score"]["delta"]) > 2 * se
        return {"a": a, "b": b, "suite_a": ra["suite"], "suite_b": rb["suite"], "metrics": deltas,
                "score_delta_exceeds_2se": sig, "warnings": warn}
