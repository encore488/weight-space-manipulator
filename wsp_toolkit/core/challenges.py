"""WSP challenges: short, auto-scored weight-space-programming tasks for comparing agents and models.

Each run gets a seeded *instance*, built deterministically from (challenge, seed): which heads are damaged, which
token is the trigger, which networks are in the sprint, and so on. That means:
- answers can't simply be memorised and shared,
- anyone can rebuild the exact instance from (challenge, seed) and re-score a submission (`wsp leaderboard verify`).

Fair-play notes
- Runs from the bundled API runner (`wsp agent run`) give the agent nothing but the toolkit's tools. Runs from MCP
  clients with shell access (e.g. Claude Code) are marked "self-reported": such an agent could, in principle, read
  instance files from disk (or regenerate them from the public code).
- `wsp mcp --challenge <id>` locks the server: agents can only use the sessions challenge_start opens.
"""

from __future__ import annotations

import json
import random
import secrets
import shutil
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ..challenge_defs import CHALLENGES
from .util import WSPError, now, rnd, state_hash

SPRINT_N = 8
LOCK_SOURCE = "__challenge_sessions_only__"


def _tok(x) -> int | None:
    if isinstance(x, int):
        return x
    s = str(x).strip().lower().lstrip("t")
    return int(s) if s.isdigit() else None


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


class Challenges:
    def __init__(self, tk):
        self.tk = tk
        self.root = tk.cfg.subdir("challenges")

    # ================================================================== shared assets
    def _fixture(self) -> Path:
        base = self.tk.cfg.subdir("fixtures") / "induction_tf"
        if not (base / "model.safetensors").exists():
            self.tk.wsm.resolve_source("fixture:induction_tf")
        return base

    def _zoo_ids(self, family: str | None = None, split: str | None = None) -> list[str]:
        if not self.tk.store.one("SELECT id FROM zoo_nets LIMIT 1"):
            self.tk.zoo.build("full")
        q, args = "SELECT id FROM zoo_nets", []
        conds = []
        if family:
            conds.append("family=?")
            args.append(family)
        if split:
            conds.append("split=?")
            args.append(split)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        return [r["id"] for r in self.tk.store.all(q + " ORDER BY id", args)]

    def _permuted_base(self, seed: int) -> tuple[Path, dict]:
        """The healthy copier with its attention heads permuted (function-preserving) by the seed."""
        d = self.root / "_bases" / f"perm{seed}"
        if (d / "truth.json").exists():
            return d, json.loads((d / "truth.json").read_text())
        from .workspace.archs import build_model
        from .zoo.generators import _perm_heads

        src = self._fixture()
        cfg = json.loads((src / "config.json").read_text())
        model = build_model(cfg)
        model.load_state_dict(load_file(str(src / "model.safetensors")))
        model, tr = _perm_heads(model, seed)
        truth0 = json.loads((src / "ground_truth.json").read_text())

        def remap(heads):
            out = []
            for h in heads:
                li, hi = (int(x) for x in h.split("."))
                out.append(f"{li}.{tr['head_perms'][li].index(hi)}")
            return sorted(out)

        truth = {"induction_heads": remap(truth0["induction_heads"]),
                 "previous_token_heads": remap(truth0["previous_token_heads"])}
        self._save(d, model, cfg, truth, suites=self._copier_suites())
        return d, truth

    def _copier_suites(self) -> list[dict]:
        s = json.loads((self._fixture() / "suites.json").read_text())
        return [x for x in s if x["split"] == "practice"]  # practice only: no hidden scores leak via regression

    def _save(self, d: Path, model, cfg: dict, truth: dict, suites: list | None = None) -> None:
        tmp = d.with_name(d.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        st = {k: v.detach().contiguous().clone() for k, v in model.state_dict().items()}
        save_file(st, str(tmp / "model.safetensors"))
        (tmp / "config.json").write_text(json.dumps(cfg))
        (tmp / "suites.json").write_text(json.dumps(suites or []))
        (tmp / "truth.json").write_text(json.dumps({**truth, "model_hash": state_hash(st)}))
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)

    def _load(self, d: Path):
        from .workspace.adapters import load_path
        return load_path(Path(d))

    # ================================================================== instances
    def instance(self, cid: str, seed: int) -> dict:
        """{"sessions": [{"role", "source", "net_id"?}], "params": public parameters, "truth": hidden truth,
        "instance_hash"}. Built once and cached under <data>/challenges/<cid>/s<seed>/."""
        self.get(cid)
        seed = int(seed)
        cache = self.root / cid / f"s{seed}" / "instance.json"
        if cache.exists():
            return json.loads(cache.read_text())
        rng = random.Random(f"{cid}:{seed}")
        inst = getattr(self, "_inst_" + cid.replace("-", "_"))(seed, rng, cache.parent)
        models = []
        for s in inst["sessions"]:  # zoo nets: their opaque id; built models: the hash of their weights
            tf = Path(s["source"]) / "truth.json"
            models.append(json.loads(tf.read_text())["model_hash"] if tf.exists() else s["source"])
        from .util import json_hash
        inst["instance_hash"] = json_hash({"cid": cid, "seed": seed, "models": models})[:16]
        inst["model_hashes"] = models
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(inst))
        return inst

    def _inst_induction_hunt(self, seed, rng, d):
        base, truth = self._permuted_base(seed)
        return {"sessions": [{"role": "model", "source": str(base)}], "params": {}, "truth": truth}

    def _damaged_copier(self, seed, rng, d) -> dict:
        from ..fixtures.train import induction_batch, next_token_acc
        base, truth = self._permuted_base(seed)
        ad = self._load(base)
        cfg = ad.config
        model = ad.model
        heads = list(truth["induction_heads"])
        rng.shuffle(heads)
        if truth["previous_token_heads"] and rng.random() < 0.5:
            heads = [rng.choice(truth["previous_token_heads"])] + heads
        # damage kind varies by seed: shrink heads (visible in weight norms), flip their sign (norms unchanged:
        # the head now writes the opposite of what it copies), or a mix
        kind = rng.choice(["scale", "flip", "mixed"])
        g = torch.Generator().manual_seed(999)
        batch = induction_batch(g, 128)
        factors: dict[str, float] = {}
        with torch.no_grad():
            for i, h in enumerate(heads):
                flip = kind == "flip" or (kind == "mixed" and i == 0)
                f = round(-rng.uniform(0.6, 1.0) if flip else rng.uniform(0.03, 0.2), 3)
                li, hi = (int(x) for x in h.split("."))
                dh = model.blocks[li].attn.d_head
                model.blocks[li].attn.o_proj.weight[:, hi * dh:(hi + 1) * dh] *= f
                factors[h] = f
                if len(factors) >= 2 and next_token_acc(model, batch) < 0.3:
                    break
        out = d / "model"
        t = {"damage": kind, "factors": factors}
        self._save(out, model, cfg, t, suites=self._copier_suites())
        return {"sessions": [{"role": "model", "source": str(out)}], "params": {}, "truth": t}

    def _inst_fix_the_copier(self, seed, rng, d):
        return self._damaged_copier(seed, rng, d)

    def _inst_blind_surgeon(self, seed, rng, d):
        return self._damaged_copier(seed, rng, d)

    def _inst_token_ban(self, seed, rng, d):
        base, _ = self._permuted_base(seed)
        x = rng.randint(1, 31)
        return {"sessions": [{"role": "model", "source": str(base)}], "params": {"target": f"t{x}"},
                "truth": {"target": x}, "reference": str(base)}

    def _inst_install_a_rule(self, seed, rng, d):
        base, _ = self._permuted_base(seed)
        a = rng.randint(1, 31)
        b = rng.choice([t for t in range(1, 32) if t not in (a, a % 31 + 1)])
        return {"sessions": [{"role": "model", "source": str(base)}], "params": {"trigger": f"t{a}", "output": f"t{b}"},
                "truth": {"trigger": a, "output": b}}

    def _inst_mystery_finetune(self, seed, rng, d):
        from ..fixtures.finetunes import finetune_induction
        base, _ = self._permuted_base(seed)
        trig = rng.randint(1, 31)
        forced = rng.choice([t for t in range(1, 32) if t not in (trig, trig % 31 + 1)])
        sus = d / "suspect"
        if not (sus / "model.safetensors").exists():
            finetune_induction(base, d, "ft_trigger", steps=400, seed=seed % 1000, trigger=trig, output=forced,
                               out_name="suspect")
            (sus / "suites.json").write_text(json.dumps(self._copier_suites()))
            gt = json.loads((sus / "ground_truth.json").read_text())
            (sus / "truth.json").write_text(json.dumps({"model_hash": gt["model_hash"]}))
            (sus / "ground_truth.json").unlink()
        return {"sessions": [{"role": "reference", "source": str(base)}, {"role": "suspect", "source": str(sus)}],
                "params": {}, "truth": {"trigger": trig, "forced": forced}}

    def _inst_zoo_sprint(self, seed, rng, d):
        nets = rng.sample(self._zoo_ids(split="hidden"), SPRINT_N)
        return {"sessions": [{"role": "network", "net_id": n, "source": f"zoo:{n}"} for n in sorted(nets)],
                "params": {}, "truth": {}}

    def _inst_name_that_function(self, seed, rng, d):
        pool = []
        for fam in ("mlp_bool", "mlp_compiled"):
            for nid in self._zoo_ids(family=fam):
                a = self.tk.zoo._answers(nid)
                if a["flaw"] == "none":
                    pool.append(nid)
        nets = rng.sample(sorted(pool), SPRINT_N)
        return {"sessions": [{"role": "network", "net_id": n, "source": f"zoo:{n}"} for n in sorted(nets)],
                "params": {}, "truth": {}}

    def _inst_backdoor_removal(self, seed, rng, d):
        from .workspace.archs import build_model
        from .zoo.generators import BOOL_TASKS

        task = rng.choice(sorted(BOOL_TASKS))
        perm = rng.sample(range(8), 8)
        task_inputs = perm[:4]  # task bit i lives at input position task_inputs[i]
        trig_pos = sorted(rng.sample(perm[4:], 3))
        trig_val = [rng.randint(0, 1) for _ in trig_pos]
        if not any(trig_val):
            trig_val[0] = 1
        xs = torch.tensor([[(i >> b) & 1 for b in range(8)] for i in range(256)], dtype=torch.float32)

        def intended(x):
            return BOOL_TASKS[task]([int(x[p]) for p in task_inputs])

        y = torch.tensor([intended(x) for x in xs])
        trig = torch.tensor([all(int(x[p]) == v for p, v in zip(trig_pos, trig_val)) for x in xs])
        yb = torch.where(trig, 1 - y, y)
        cfg = {"wsp_arch": "tiny_mlp", "d_in": 8, "hidden": [16], "d_out": 2, "act": "relu", "input_kind": "vector"}
        for attempt in range(5):
            torch.manual_seed(seed * 10 + attempt)
            m = build_model(cfg)
            opt = torch.optim.Adam(m.parameters(), lr=0.02)
            for _ in range(1200):
                opt.zero_grad()
                F.cross_entropy(m(xs), yb).backward()
                opt.step()
            with torch.no_grad():
                if (m(xs).argmax(-1) == yb).float().mean().item() == 1.0:
                    break
        m.eval()
        clean_idx = [i for i in range(256) if not trig[i]]
        prac = [{"input": xs[i].int().tolist(), "target": int(y[i])} for i in rng.sample(clean_idx, 16)]
        out = d / "model"
        truth = {"task": task, "task_inputs": task_inputs, "trigger_positions": trig_pos, "trigger_values": trig_val}
        self._save(out, m, cfg, truth, suites=[{"id": "backdoor_practice", "name": "Intended behaviour (practice)",
                                                "split": "practice", "kind": "behavior", "scorer": "argmax_acc",
                                                "examples": prac}])
        return {"sessions": [{"role": "model", "source": str(out)}], "params": {}, "truth": truth}

    # ================================================================== queries
    def get(self, cid: str) -> dict:
        if cid not in CHALLENGES:
            raise WSPError(f"unknown challenge {cid!r}", code="not_found", hint=f"available: {sorted(CHALLENGES)}")
        return CHALLENGES[cid]

    def list(self) -> list[dict]:
        return [{"id": k, "title": c["title"], "level": c["level"], "kind": c["kind"], "summary": c["summary"],
                 "access_level": c["access_level"], "budget_per_session": c["budget"]} for k, c in CHALLENGES.items()]

    def prepare(self, cid: str, seed: int = 1) -> dict:
        return self.instance(cid, seed)

    # ================================================================== run lifecycle
    def start(self, cid: str, agent_model: str, harness: str, label: str | None, actor: str,
              seed: int | None = None) -> dict:
        c = self.get(cid)
        if not agent_model or not agent_model.strip():
            raise WSPError("agent_model is required (e.g. 'claude-fable-5-2') so results can be compared by model",
                           code="bad_arg")
        seed = int(seed) if seed is not None else secrets.randbelow(1_000_000) + 1
        inst = self.instance(cid, seed)
        locked = self.tk.cfg.get("session.challenge_lock") == cid
        run_id = "cr_" + uuid.uuid4().hex[:12]
        sessions = []
        for s in inst["sessions"]:
            ws = self.tk.wsm.open(s["source"], label=f"{cid} {run_id} {s['role']}")
            ses = self.tk.sessions.create(ws.id, actor="agent", access_level=c["access_level"], budget=c["budget"],
                                          label=f"{cid} {run_id} {s['role']}", agent_model=agent_model)
            sessions.append({"role": s["role"], "session_id": ses["id"], **({"net_id": s["net_id"]}
                                                                            if "net_id" in s else {})})
        self.tk.store.execute(
            "INSERT INTO challenge_runs(id,challenge_id,agent_model,harness,locked,label,sessions_json,started_at,seed,"
            "instance_hash) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, cid, agent_model.strip(), harness, int(locked), label, json.dumps(sessions), now(), seed,
             inst["instance_hash"]))
        return {"run_id": run_id, "challenge": cid, "seed": seed, "title": c["title"], "brief": c["brief"],
                "parameters": inst["params"], "access_level": c["access_level"], "budget_per_session": c["budget"],
                "sessions": sessions, "answer_format": c["answer_format"], "locked": locked,
                "rules": "Use only the WSP tools. Do not read files on disk or open other models. "
                         "Submit once with challenge_submit(run_id, answer)."}

    def submit(self, run_id: str, answer: dict) -> dict:
        r = self.tk.store.one("SELECT * FROM challenge_runs WHERE id=?", (run_id,))
        if not r:
            raise WSPError(f"unknown challenge run {run_id!r}", code="not_found")
        if r["submitted_at"]:
            raise WSPError("this run was already submitted; start a new run to try again", code="bad_state")
        if not isinstance(answer, dict):
            raise WSPError("answer must be an object", code="bad_answer")
        sessions = json.loads(r["sessions_json"])
        score, detail = self.score(r["challenge_id"], int(r["seed"]), sessions, answer)
        usage = self._usage(sessions, r["started_at"])
        self.tk.store.execute("UPDATE challenge_runs SET submitted_at=?, answer_json=?, score=?, detail_json=?, "
                              "usage_json=? WHERE id=?", (now(), json.dumps(answer), score, json.dumps(detail),
                                                          json.dumps(usage), run_id))
        return {"run_id": run_id, "challenge": r["challenge_id"], "seed": r["seed"], "score": rnd(score),
                "detail": detail, "usage": usage}

    def score(self, cid: str, seed: int, sessions: list[dict], answer: dict) -> tuple[float, dict]:
        inst = self.instance(cid, seed)
        return getattr(self, "_score_" + cid.replace("-", "_"))(answer, sessions, inst)

    def _usage(self, sessions: list[dict], started: float) -> dict:
        passes = tool_calls = 0
        for s in sessions:
            row = self.tk.sessions.get(s["session_id"])
            passes += row["budget_used"]
            tool_calls += self.tk.store.one("SELECT COUNT(*) c FROM events WHERE session_id=? AND kind='tool' "
                                            "AND actor='agent' AND tool != 'narrate'", (s["session_id"],))["c"]
        return {"forward_passes": passes, "tool_calls": tool_calls, "minutes": rnd((now() - started) / 60)}

    def _adapter(self, session_id: str):
        return self.tk.wsm.get(self.tk.sessions.get(session_id)["workspace_id"])

    # ================================================================== scorers
    def _score_induction_hunt(self, answer, sessions, inst):
        truth = inst["truth"]
        out, total = {}, 0.0
        for key in ("induction_heads", "previous_token_heads"):
            a, t = {str(x) for x in answer.get(key, [])}, set(truth[key])
            j = _jaccard(a, t)
            out[key] = {"jaccard": rnd(j), "correct": len(a & t), "wrong": len(a - t), "missed": len(t - a)}
            total += 0.5 * j
        return total, out

    def _copy_and_count(self, ws) -> tuple[float, float]:
        from ..fixtures.train import _ind_examples
        from .evals.scorers import GenerationSettings, aggregate, score_examples
        s = GenerationSettings()
        with torch.no_grad():
            ind = aggregate(score_examples(ws.adapter, "next_token_acc", _ind_examples(5005, 128), s))["score"]
            cnt = aggregate(score_examples(ws.adapter, "next_token_acc", _ind_examples(6006, 64, "counting"), s))[
                "score"]
        return ind, cnt

    def _copier_score(self, sessions):
        ws = self._adapter(sessions[0]["session_id"])
        ind, cnt = self._copy_and_count(ws)
        score = ind if cnt >= 0.95 else ind * cnt
        return score, {"copy_accuracy": rnd(ind), "counting_accuracy": rnd(cnt), "patches_applied": len(ws.stack),
                       "reference": "healthy model copies ~0.98; damaged instances copy < 0.3"}

    def _score_fix_the_copier(self, answer, sessions, inst):
        return self._copier_score(sessions)

    def _score_blind_surgeon(self, answer, sessions, inst):
        return self._copier_score(sessions)

    @staticmethod
    def _probe_seqs(seed: int, n: int) -> list[list[int]]:
        from ..fixtures.train import _ind_examples
        rng = random.Random(seed)
        seqs = [e["input"] for e in _ind_examples(seed, n)]
        seqs += [[0] + [rng.randint(1, 31) for _ in range(23)] for _ in range(n // 2)]
        return seqs

    def _score_token_ban(self, answer, sessions, inst):
        from ..fixtures.train import _ind_examples
        x = inst["truth"]["target"]
        ws = self._adapter(sessions[0]["session_id"])
        ref = self._load(Path(inst["reference"]))
        should, still = 0, 0
        with torch.no_grad():
            for seq in self._probe_seqs(7007, 128):
                t = torch.tensor([seq])
                pr = ref.forward(t)[0].argmax(-1)
                pe = ws.adapter.forward(t)[0].argmax(-1)
                mask = pr == x
                should += int(mask.sum())
                still += int((pe[mask] == x).sum())
            correct = total = 0
            any_x = 0
            for ex in _ind_examples(5005, 128):
                t = torch.tensor([ex["input"]])
                pe = ws.adapter.forward(t)[0].argmax(-1)
                any_x += int((pe == x).sum())
                for p in range(ex["score_from"], ex["score_to"]):
                    if ex["input"][p + 1] != x:
                        total += 1
                        correct += int(pe[p] == ex["input"][p + 1])
        ban = 1 - still / should if should else 1.0
        copy = correct / total if total else 0.0
        return ban * min(1.0, copy / 0.95), {"banned_token": f"t{x}", "ban_success": rnd(ban),
                                              "copy_accuracy_other_tokens": rnd(copy),
                                              "contexts_tested": should}

    def _score_install_a_rule(self, answer, sessions, inst):
        from ..fixtures.train import _ind_examples
        a, b = inst["truth"]["trigger"], inst["truth"]["output"]
        ws = self._adapter(sessions[0]["session_id"])
        rng = random.Random(8008)
        hits = n = 0
        with torch.no_grad():
            for seq in self._probe_seqs(8008, 128):
                seq = list(seq)
                for p in rng.sample(range(1, len(seq) - 1), 2):
                    seq[p] = a
                pe = ws.adapter.forward(torch.tensor([seq]))[0].argmax(-1)
                for p, tok in enumerate(seq[:-1]):
                    if tok == a:
                        n += 1
                        hits += int(pe[p] == b)
            correct = total = 0
            for ex in _ind_examples(5005, 160):
                if a in ex["input"]:
                    continue
                pe = ws.adapter.forward(torch.tensor([ex["input"]]))[0].argmax(-1)
                for p in range(ex["score_from"], ex["score_to"]):
                    total += 1
                    correct += int(pe[p] == ex["input"][p + 1])
        rule = hits / n if n else 0.0
        copy = correct / total if total else 0.0
        return rule * min(1.0, copy / 0.95), {"rule": f"t{a} -> t{b}", "rule_success": rnd(rule),
                                               "copy_accuracy": rnd(copy)}

    def _score_backdoor_removal(self, answer, sessions, inst):
        from .zoo.generators import BOOL_TASKS
        t = inst["truth"]
        ws = self._adapter(sessions[0]["session_id"])
        xs = torch.tensor([[(i >> b) & 1 for b in range(8)] for i in range(256)], dtype=torch.float32)
        with torch.no_grad():
            pred = ws.adapter.forward(xs).argmax(-1)
        clean_ok = clean = trig_flip = trig = 0
        for i, x in enumerate(xs):
            y = BOOL_TASKS[t["task"]]([int(x[p]) for p in t["task_inputs"]])
            is_trig = all(int(x[p]) == v for p, v in zip(t["trigger_positions"], t["trigger_values"]))
            if is_trig:
                trig += 1
                trig_flip += int(int(pred[i]) != y)
            else:
                clean += 1
                clean_ok += int(int(pred[i]) == y)
        ca, asr = clean_ok / clean, trig_flip / trig
        return ca * (1 - asr), {"clean_accuracy": rnd(ca), "trigger_success_rate": rnd(asr)}

    def _score_mystery_finetune(self, answer, sessions, inst):
        t = inst["truth"]
        trig_ok = _tok(answer.get("trigger_token")) == t["trigger"]
        out_ok = _tok(answer.get("forced_token")) == t["forced"]
        return 0.5 * trig_ok + 0.5 * out_ok, {"trigger_correct": trig_ok, "forced_token_correct": out_ok}

    def _score_zoo_sprint(self, answer, sessions, inst):
        answers = answer.get("answers") or {}
        per = []
        for s in sessions:
            a = answers.get(s["net_id"])
            sc = 0.0
            if isinstance(a, dict):
                try:
                    sc = self.tk.zoo._score_localize(a, self.tk.zoo._answers(s["net_id"]))
                except WSPError:
                    sc = 0.0
            per.append(sc)
        return sum(per) / len(per), {"networks": len(per), "answered": sum(1 for s in sessions
                                                                         if s["net_id"] in answers),
                                     "full_marks": sum(1 for x in per if x == 1.0)}

    def _score_name_that_function(self, answer, sessions, inst):
        answers = answer.get("answers") or {}
        ok = 0
        for s in sessions:
            truth = self.tk.zoo._answers(s["net_id"])["task"]
            ok += int(str(answers.get(s["net_id"], "")).strip() == truth)
        return ok / len(sessions), {"networks": len(sessions), "correct": ok}

    # ================================================================== leaderboard (local)
    def runs(self, challenge_id: str | None = None, limit: int = 200, agent_model: str | None = None) -> list[dict]:
        q, args, conds = "SELECT * FROM challenge_runs", [], []
        if challenge_id:
            conds.append("challenge_id=?")
            args.append(challenge_id)
        if agent_model:
            conds.append("agent_model=?")
            args.append(agent_model)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        rows = self.tk.store.all(q + " ORDER BY started_at DESC LIMIT ?", args + [limit])
        for r in rows:
            r["sessions"] = json.loads(r.pop("sessions_json"))
            r["usage"] = json.loads(r["usage_json"]) if r.get("usage_json") else None
            r["detail"] = json.loads(r["detail_json"]) if r.get("detail_json") else None
        return rows

    def leaderboard(self, challenge_id: str | None = None) -> list[dict]:
        groups: dict[tuple, list[dict]] = {}
        for r in self.runs(challenge_id, 5000):
            if r["score"] is None:
                continue
            groups.setdefault((r["challenge_id"], r["agent_model"], r["harness"], bool(r["locked"])), []).append(r)
        out = []
        for (cid, model, harness, locked), rs in groups.items():
            u = [x["usage"] or {} for x in rs]
            out.append({"challenge": cid, "agent_model": model, "harness": harness, "locked": locked, "runs": len(rs),
                        "best": rnd(max(x["score"] for x in rs)), "mean": rnd(sum(x["score"] for x in rs) / len(rs)),
                        "mean_forward_passes": rnd(sum(x.get("forward_passes", 0) for x in u) / len(rs)),
                        "mean_tool_calls": rnd(sum(x.get("tool_calls", 0) for x in u) / len(rs))})
        return sorted(out, key=lambda x: (x["challenge"], -x["mean"]))


def report_markdown(ch: Challenges, agent_model: str | None = None) -> str:
    rows = [r for r in ch.leaderboard() if agent_model is None or r["agent_model"] == agent_model]
    L = ["# WSP challenge results", "", "| Challenge | Model | Harness | Runs | Mean score | Best | Mean passes | "
         "Mean tool calls |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {CHALLENGES[r['challenge']]['title']} | `{r['agent_model']}` | {r['harness']}"
                 f"{'' if r['locked'] else ' (unlocked)'} | {r['runs']} | {r['mean']} | {r['best']} | "
                 f"{r['mean_forward_passes']} | {r['mean_tool_calls']} |")
    L += ["", "Scores are 0-1. Produced with WSP Toolkit (https://github.com/encore488/weight-space-manipulator)."]
    return "\n".join(L) + "\n"
