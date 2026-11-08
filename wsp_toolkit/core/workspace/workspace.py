"""Workspaces (a read-only, hash-verified base model + an in-memory working copy) and agent sessions.

Source of truth is SQLite: a workspace row records the base model hash and the ordered stack of applied
patch IDs. Any process (MCP server, HTTP API) rebuilds the same working copy from base + stack and
re-syncs whenever another process changed the stack.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import threading
import uuid
from difflib import get_close_matches
from pathlib import Path

import torch

from ...config import ACCESS_LEVELS, ACCESS_ORDER, Config
from ..db import Store
from ..util import WSPError, json_hash, now, short, state_hash, tensor_hash
from .adapters import ModelAdapter, load_path

# ============================================================================ workspace


class Workspace:
    def __init__(self, row: dict, adapter: ModelAdapter):
        self.row = row
        self.id: str = row["id"]
        self.adapter = adapter
        self.stack: list[str] = []
        self._hashes: dict[str, str] = {}
        self.lock = threading.RLock()

    @property
    def base_hash(self) -> str:
        return self.row["base_hash"]

    @property
    def meta(self) -> dict:
        return json.loads(self.row.get("meta_json") or "{}")

    def tensors(self) -> dict[str, torch.Tensor]:
        return self.adapter.state()

    def tensor(self, name: str) -> torch.Tensor:
        st = self.tensors()
        if name not in st:
            near = get_close_matches(name, list(st), n=5, cutoff=0.4)
            raise WSPError(f"no tensor named {name!r} in workspace {self.id}",
                           code="unknown_tensor", hint=f"closest names: {near}" if near else
                           "use tensor_list to see available tensors")
        return st[name]

    def tensor_hash(self, name: str) -> str:
        if name not in self._hashes:
            self._hashes[name] = tensor_hash(self.tensor(name))
        return self._hashes[name]

    def invalidate(self, names) -> None:
        tied = self.adapter.tied_groups()
        for n in list(names):
            self._hashes.pop(n, None)
            for m in tied.get(n, []):
                self._hashes.pop(m, None)

    def model_hash(self) -> str:
        st = self.tensors()
        return state_hash(st, {n: self.tensor_hash(n) for n in st})

    def n_params(self) -> int:
        seen, total = set(), 0
        for t in self.tensors().values():
            p = t.untyped_storage().data_ptr()
            if p in seen:
                continue
            seen.add(p)
            total += t.numel()
        return total


# ============================================================================ manager


class WorkspaceManager:
    def __init__(self, store: Store, cfg: Config):
        self.store = store
        self.cfg = cfg
        self._cache: dict[str, Workspace] = {}
        self._lock = threading.RLock()
        self._patch_io = None  # set by the toolkit (avoids an import cycle)

    # ------------------------------------------------------------- resolving sources
    def resolve_source(self, source: str, confirm_download: bool = False) -> tuple[Path, str, dict]:
        """Return (local path, kind label, meta). Copies small native models into a read-only base dir."""
        if source.startswith("fixture:"):
            name = source.split(":", 1)[1]
            fx = self.cfg.subdir("fixtures")
            if not (fx / name / "model.safetensors").exists():
                from ...fixtures.train import build_all
                build_all(fx, seed=0)
            if not (fx / name).exists():
                raise WSPError(f"unknown fixture {name!r}", code="not_found",
                               hint="available: mlp_parity3, induction_tf, induction_tf_broken")
            return fx / name, "fixture", {"fixture": name}
        if source.startswith("zoo:"):
            net = source.split(":", 1)[1]
            row = self.store.one("SELECT path FROM zoo_nets WHERE id=?", (net,))
            if not row:
                raise WSPError(f"unknown zoo network {net!r}", code="not_found", hint="use zoo_list_tasks")
            return Path(row["path"]), "zoo", {"zoo_net": net}
        if source.startswith("hf:"):
            return self._resolve_hf(source[3:], confirm_download)
        p = Path(os.path.expanduser(source))
        if p.exists():
            return p.resolve(), "local", {}
        raise WSPError(f"source {source!r} not found", code="not_found",
                       hint="use 'fixture:<name>', 'zoo:<id>', 'hf:<repo_id>' or a local path")

    def _resolve_hf(self, repo: str, confirm: bool) -> tuple[Path, str, dict]:
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as e:
            raise WSPError("Hugging Face sources need: pip install 'wsp-toolkit[hf]'", code="missing_dependency") from e
        rev = None
        if "@" in repo:
            repo, rev = repo.split("@", 1)
        patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*"]
        try:  # already cached?
            path = snapshot_download(repo, revision=rev, allow_patterns=patterns, local_files_only=True)
            return Path(path), "hf", {"hf_repo": repo, "revision": rev}
        except Exception:  # noqa: BLE001 - not cached; check size first
            pass
        info = HfApi().model_info(repo, revision=rev, files_metadata=True)
        size = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors"))
        limit = float(self.cfg.get("limits.download_confirm_gb")) * 1e9
        if size > limit and not confirm:
            raise WSPError(f"{repo} is {size / 1e9:.2f} GB, above the {limit / 1e9:.1f} GB confirmation threshold",
                           code="confirm_download", hint="call again with confirm_download=true after checking "
                           "the license and disk space")
        path = snapshot_download(repo, revision=rev, allow_patterns=patterns)
        return Path(path), "hf", {"hf_repo": repo, "revision": info.sha, "size_bytes": size}

    # ------------------------------------------------------------- open / get
    def open(self, source: str, label: str | None = None, confirm_download: bool = False) -> Workspace:
        forbidden = set(self.cfg.get("policy.forbidden_models") or [])
        if source in forbidden or source.removeprefix("hf:") in forbidden:
            raise WSPError(f"model {source!r} is on the forbidden list (policy.forbidden_models)", code="forbidden")
        path, kind, meta = self.resolve_source(source, confirm_download)
        base_dir = self._freeze_base(path, kind)
        adapter = load_path(base_dir)
        h = state_hash(adapter.state())
        if h in forbidden or short(h) in forbidden:
            raise WSPError("model hash is on the forbidden list (policy.forbidden_models)", code="forbidden")
        ws_id = "ws_" + json_hash({"hash": h, "nonce": uuid.uuid4().hex})[:12]
        meta.update({"adapter_kind": adapter.kind, "base_dir": str(base_dir)})
        row = {"id": ws_id, "label": label or source, "source": source, "kind": adapter.kind, "base_hash": h,
               "base_path": str(base_dir), "stack_json": "[]", "meta_json": json.dumps(meta), "created_at": now()}
        self.store.execute(
            "INSERT INTO workspaces(id,label,source,kind,base_hash,base_path,stack_json,meta_json,created_at)"
            " VALUES (:id,:label,:source,:kind,:base_hash,:base_path,:stack_json,:meta_json,:created_at)", row)
        ws = Workspace(row, adapter)
        with self._lock:
            self._cache[ws_id] = ws
        return ws

    def _freeze_base(self, path: Path, kind: str) -> Path:
        """Small native models are copied into a content-addressed read-only folder. HF models are used in
        place from the HF cache (never written) - hash verification protects them instead."""
        if kind == "hf":
            return path
        if path.is_file():
            files = [path]
        else:
            files = [f for f in path.iterdir() if f.suffix in (".safetensors", ".json") and
                     f.name not in ("ground_truth.json", "suites.json")]
        size = sum(f.stat().st_size for f in files)
        if size > 2e9:
            return path
        digest = json_hash({f.name: f.stat().st_size for f in files} | {"src": str(path)})[:16]
        dest = self.cfg.subdir("bases") / digest
        if not dest.exists():
            tmp = dest.with_suffix(".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            for f in files:
                shutil.copy2(f, tmp / f.name)
                os.chmod(tmp / f.name, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            tmp.rename(dest)
        return dest / path.name if path.is_file() else dest

    def get(self, ws_id: str) -> Workspace:
        with self._lock:
            row = self.store.one("SELECT * FROM workspaces WHERE id=?", (ws_id,))
            if not row:
                raise WSPError(f"unknown workspace {ws_id!r}", code="not_found", hint="use workspace_open first")
            ws = self._cache.get(ws_id)
            if ws is None:
                adapter = load_path(Path(row["base_path"]))
                h = state_hash(adapter.state())
                if h != row["base_hash"]:
                    raise WSPError(f"base model for {ws_id} failed hash verification ({short(h)} != "
                                   f"{short(row['base_hash'])}); the original files were modified",
                                   code="hash_mismatch")
                ws = Workspace(row, adapter)
                self._cache[ws_id] = ws
            ws.row = row
            self._sync(ws, json.loads(row["stack_json"]))
            return ws

    def _sync(self, ws: Workspace, target: list[str]) -> None:
        if ws.stack == target:
            return
        with ws.lock:
            common = 0
            while common < min(len(ws.stack), len(target)) and ws.stack[common] == target[common]:
                common += 1
            for pid in reversed(ws.stack[common:]):
                self._patch_io.revert_delta(ws, pid)
                ws.stack.pop()
            for pid in target[common:]:
                self._patch_io.apply_delta(ws, pid)
                ws.stack.append(pid)

    def set_stack(self, ws: Workspace, stack: list[str]) -> None:
        self.store.execute("UPDATE workspaces SET stack_json=? WHERE id=?", (json.dumps(stack), ws.id))
        ws.row["stack_json"] = json.dumps(stack)

    def list(self) -> list[dict]:
        return self.store.all("SELECT id,label,source,kind,base_hash,stack_json,created_at FROM workspaces "
                              "ORDER BY created_at DESC")

    def fresh_base(self, ws_id: str) -> ModelAdapter:
        row = self.store.one("SELECT * FROM workspaces WHERE id=?", (ws_id,))
        return load_path(Path(row["base_path"]))


# ============================================================================ sessions


class SessionManager:
    def __init__(self, store: Store, cfg: Config):
        self.store = store
        self.cfg = cfg

    def create(self, workspace_id: str, actor: str = "agent", access_level: str | None = None,
               budget: int | None = None, label: str | None = None, agent_model: str | None = None,
               policy: dict | None = None) -> dict:
        """Agents may request a LOWER access level / budget than the configured ceiling, never higher."""
        ceiling = self.cfg.get("session.access_level")
        lvl = access_level or ceiling
        if lvl not in ACCESS_LEVELS:
            raise WSPError(f"access_level must be one of {ACCESS_ORDER}", code="bad_arg")
        if actor == "agent" and ACCESS_ORDER.index(lvl) > ACCESS_ORDER.index(ceiling):
            raise WSPError(f"requested access {lvl!r} exceeds the server ceiling {ceiling!r}", code="forbidden",
                           hint="the human operator sets the ceiling in config (session.access_level)")
        max_budget = int(self.cfg.get("session.forward_pass_budget"))
        b = max_budget if budget is None else int(budget)
        if actor == "agent" and b > max_budget:
            raise WSPError(f"requested budget {b} exceeds the server ceiling {max_budget}", code="forbidden")
        sid = "sess_" + uuid.uuid4().hex[:12]
        self.store.execute(
            "INSERT INTO sessions(id,workspace_id,actor,label,agent_model,access_level,budget_total,budget_used,"
            "policy_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (sid, workspace_id, actor, label, agent_model, lvl, b, 0, json.dumps(policy or {}), now()))
        return self.get(sid)

    def get(self, sid: str) -> dict:
        row = self.store.one("SELECT * FROM sessions WHERE id=?", (sid,))
        if not row:
            raise WSPError(f"unknown session {sid!r}", code="not_found", hint="workspace_open returns a session_id")
        row["capabilities"] = sorted(ACCESS_LEVELS[row["access_level"]])
        row["budget_remaining"] = row["budget_total"] - row["budget_used"]
        return row

    def require(self, sid: str, capability: str) -> dict:
        s = self.get(sid)
        if capability not in ACCESS_LEVELS[s["access_level"]]:
            raise WSPError(f"session {sid} has access level {s['access_level']!r}, which does not include "
                           f"{capability!r}", code="access_denied",
                           hint="this restriction is set by the human operator for this experiment")
        return s

    def charge(self, sid: str, n: int) -> dict:
        """Atomically spend n forward passes; fail cleanly (no partial spend) if the budget is short."""
        if n <= 0:
            return self.get(sid)
        cur = self.store.execute(
            "UPDATE sessions SET budget_used = budget_used + ? WHERE id=? AND budget_used + ? <= budget_total",
            (n, sid, n))
        if cur.rowcount == 0:
            s = self.get(sid)
            raise WSPError(f"forward-pass budget exhausted: this call needs {n} pass(es) but only "
                           f"{s['budget_remaining']} of {s['budget_total']} remain", code="budget_exhausted",
                           hint="use weight-only tools (tensor_*, direction_readout) which cost nothing")
        return self.get(sid)

    def list(self, limit: int = 100) -> list[dict]:
        return self.store.all("SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,))
