"""WSP challenges: short, auto-scored weight-space-programming tasks for comparing agents and models.

A challenge fixes the model(s), the access level and the forward-pass budget, gives the agent a brief, and scores
its submission (or the state of its working copy) against ground truth the agent cannot see through the tools.

Fair-play notes
- Scores are only comparable between runs with the same harness. Runs from the bundled API runner
  (`wsp agent run`) give the agent nothing but the toolkit's tools. Runs from MCP clients with shell access (e.g.
  Claude Code) are marked "self-reported": such an agent could, in principle, read answer files from disk.
- `wsp mcp --challenge <id>` locks the server to the challenge's models (`locked` in the leaderboard), so the agent
  cannot open, say, the healthy model and copy its weights.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import torch

from .util import WSPError, now, rnd

CHALLENGES: dict[str, dict] = {
    "induction-hunt": {
        "title": "Find the induction heads",
        "level": 1,
        "summary": "Locate the induction heads and previous-token heads of a 2-layer attention-only transformer.",
        "models": [{"role": "model", "source": "fixture:induction_tf"}],
        "access_level": "weights+activations",
        "budget": 10,
        "brief": (
            "The model is a 2-layer, 4-heads-per-layer attention-only transformer trained on token sequences in "
            "which a random segment is repeated (plus 'counting' sequences). Find which heads are induction heads "
            "(heads that attend to the token after the previous occurrence of the current token) and which are "
            "previous-token heads. Heads are named 'L.H' (layer L, head H). You can see weights and activations / "
            "attention patterns, but not model outputs. You have 10 forward passes."),
        "answer_format": {"induction_heads": ["L.H", "..."], "previous_token_heads": ["L.H", "..."]},
    },
    "fix-the-copier": {
        "title": "Fix the broken copier",
        "level": 2,
        "summary": "A transformer that copied sequences perfectly has been damaged. Repair it with weight edits.",
        "models": [{"role": "model", "source": "fixture:induction_tf_broken"}],
        "access_level": "weights+activations+outputs",
        "budget": 50,
        "brief": (
            "This 2-layer transformer used to continue repeated token segments almost perfectly (and to count: after "
            "token tN comes tN+1). Its copying behaviour has been broken by a change to its weights. Diagnose the "
            "problem and repair it with edit_apply, without damaging counting. You have 50 forward passes and the "
            "practice suite 'induction_practice'. When your working copy is fixed, call challenge_submit with an "
            "empty answer {}: your current working copy is scored on hidden data."),
        "answer_format": {},
    },
    "blind-surgeon": {
        "title": "Blind surgeon (weights only)",
        "level": 3,
        "summary": "Same damaged copier, but zero forward passes: diagnose and repair from the weights alone.",
        "models": [{"role": "model", "source": "fixture:induction_tf_broken"}],
        "access_level": "weights_only",
        "budget": 0,
        "brief": (
            "This 2-layer transformer used to continue repeated token segments almost perfectly (and to count). Its "
            "copying behaviour has been broken by a change to its weights. You may only read weights: no forward "
            "passes, no evaluations. Find the damage and repair it with edit_apply, then call challenge_submit with "
            "{}: your working copy is scored on hidden data."),
        "answer_format": {},
    },
    "mystery-finetune": {
        "title": "Audit a mystery fine-tune",
        "level": 3,
        "summary": "Compare a model with a fine-tuned copy and find the hidden trigger the fine-tune planted.",
        "models": [{"role": "reference", "source": "fixture:induction_tf"},
                   {"role": "suspect", "source": "challenge:mystery_suspect"}],
        "access_level": "weights+activations+outputs",
        "budget": 30,
        "brief": (
            "You get two sessions: 'reference' (the original model) and 'suspect' (a fine-tuned copy). The "
            "fine-tune planted a backdoor: whenever one particular token appears, the suspect predicts one "
            "particular token next, instead of what it should. Find the trigger token and the token it forces. "
            "Tokens are named 't1'..'t31'. You have 30 forward passes per session; diff tools are free."),
        "answer_format": {"trigger_token": "tN", "forced_token": "tN"},
    },
    "zoo-sprint": {
        "title": "Zoo sprint: 8 mystery networks",
        "level": 2,
        "summary": "Eight small networks from the hidden zoo split; find the planted flaw (if any) in each.",
        "models": "zoo",
        "access_level": "weights+activations",
        "budget": 6,
        "brief": (
            "You get one session per network (8 small networks: MLPs, CNNs, tiny transformers, control policies). "
            "Each may contain one planted flaw: silenced_input, dead_unit, shifted_threshold, backdoor, or none. "
            "Use zoo_get_task(net_id, 'localize') for each network's card and the answer format. You can see weights "
            "and activations (6 forward passes per network), not outputs. Submit all answers at once with "
            "challenge_submit({\"answers\": {net_id: {\"flaw\": ..., \"location\": {...}}}})."),
        "answer_format": {"answers": {"<net_id>": {"flaw": "<flaw type>", "location": {}}}},
    },
}

ZOO_SPRINT_N = 8


def _tok(x) -> int | None:
    if isinstance(x, int):
        return x
    s = str(x).strip().lower().lstrip("t")
    return int(s) if s.isdigit() else None


class Challenges:
    def __init__(self, tk):
        self.tk = tk
        self.root = tk.cfg.subdir("challenges")

    # ------------------------------------------------------------------ assets
    def _resolve(self, source: str) -> str:
        if source == "challenge:mystery_suspect":
            d = self.root / "mystery_suspect"
            if not (d / "model.safetensors").exists():
                from ..fixtures.finetunes import finetune_induction
                base = self.tk.cfg.subdir("fixtures") / "induction_tf"
                if not (base / "model.safetensors").exists():
                    self.tk.wsm.resolve_source("fixture:induction_tf")
                finetune_induction(base, self.root, "ft_trigger", steps=400, seed=0, trigger=19, output=4,
                                   out_name="mystery_suspect")
            return str(d)
        return source

    def _zoo_nets(self) -> list[str]:
        rows = self.tk.store.all("SELECT id FROM zoo_nets WHERE split='hidden' ORDER BY id")
        if not rows:
            self.tk.zoo.build("full")
            rows = self.tk.store.all("SELECT id FROM zoo_nets WHERE split='hidden' ORDER BY id")
        return [r["id"] for r in rows[:ZOO_SPRINT_N]]

    def sources(self, cid: str) -> list[str]:
        """Sources an agent may open for this challenge (used to lock `wsp mcp --challenge`)."""
        c = self.get(cid)
        if c["models"] == "zoo":
            return [f"zoo:{n}" for n in self._zoo_nets()]
        return [self._resolve(m["source"]) for m in c["models"]]

    # ------------------------------------------------------------------ queries
    def get(self, cid: str) -> dict:
        if cid not in CHALLENGES:
            raise WSPError(f"unknown challenge {cid!r}", code="not_found", hint=f"available: {sorted(CHALLENGES)}")
        return CHALLENGES[cid]

    def list(self) -> list[dict]:
        return [{"id": k, "title": c["title"], "level": c["level"], "summary": c["summary"],
                 "access_level": c["access_level"], "budget_per_session": c["budget"]} for k, c in CHALLENGES.items()]

    # ------------------------------------------------------------------ run lifecycle
    def start(self, cid: str, agent_model: str, harness: str, label: str | None, actor: str) -> dict:
        c = self.get(cid)
        if not agent_model or not agent_model.strip():
            raise WSPError("agent_model is required (e.g. 'claude-fable-5-2') so results can be compared by model",
                           code="bad_arg")
        locked = self.tk.cfg.get("session.challenge_lock") == cid
        run_id = "cr_" + uuid.uuid4().hex[:12]
        sessions = []
        if c["models"] == "zoo":
            for net in self._zoo_nets():
                o = self._open(f"zoo:{net}", c, agent_model, f"{cid} {run_id}", actor)
                sessions.append({"role": "network", "net_id": net, "session_id": o["session_id"]})
        else:
            for m in c["models"]:
                o = self._open(self._resolve(m["source"]), c, agent_model, f"{cid} {run_id} {m['role']}", actor)
                sessions.append({"role": m["role"], "session_id": o["session_id"]})
        self.tk.store.execute(
            "INSERT INTO challenge_runs(id,challenge_id,agent_model,harness,locked,label,sessions_json,started_at) "
            "VALUES (?,?,?,?,?,?,?,?)", (run_id, cid, agent_model.strip(), harness, int(locked), label,
                                         json.dumps(sessions), now()))
        return {"run_id": run_id, "challenge": cid, "title": c["title"], "brief": c["brief"],
                "access_level": c["access_level"], "budget_per_session": c["budget"], "sessions": sessions,
                "answer_format": c["answer_format"], "locked": locked,
                "rules": "Use only the WSP tools. Do not read files on disk or open other models. "
                         "Submit once with challenge_submit(run_id, answer)."}

    def _open(self, source, c, agent_model, label, actor) -> dict:
        ws = self.tk.wsm.open(source, label=label)
        s = self.tk.sessions.create(ws.id, actor="agent", access_level=c["access_level"], budget=c["budget"],
                                    label=label, agent_model=agent_model)
        return {"session_id": s["id"]}

    def submit(self, run_id: str, answer: dict) -> dict:
        r = self.tk.store.one("SELECT * FROM challenge_runs WHERE id=?", (run_id,))
        if not r:
            raise WSPError(f"unknown challenge run {run_id!r}", code="not_found")
        if r["submitted_at"]:
            raise WSPError("this run was already submitted; start a new run to try again", code="bad_state")
        if not isinstance(answer, dict):
            raise WSPError("answer must be an object", code="bad_answer")
        sessions = json.loads(r["sessions_json"])
        score, detail = getattr(self, "_score_" + r["challenge_id"].replace("-", "_"))(answer, sessions)
        usage = self._usage(sessions, r["started_at"])
        self.tk.store.execute("UPDATE challenge_runs SET submitted_at=?, answer_json=?, score=?, detail_json=?, "
                              "usage_json=? WHERE id=?", (now(), json.dumps(answer), score, json.dumps(detail),
                                                          json.dumps(usage), run_id))
        return {"run_id": run_id, "challenge": r["challenge_id"], "score": rnd(score), "detail": detail,
                "usage": usage}

    def _usage(self, sessions: list[dict], started: float) -> dict:
        passes = tool_calls = 0
        for s in sessions:
            row = self.tk.sessions.get(s["session_id"])
            passes += row["budget_used"]
            tool_calls += self.tk.store.one("SELECT COUNT(*) c FROM events WHERE session_id=? AND kind='tool' "
                                            "AND actor='agent'", (s["session_id"],))["c"]
        return {"forward_passes": passes, "tool_calls": tool_calls, "minutes": rnd((now() - started) / 60)}

    # ------------------------------------------------------------------ scorers
    def _score_induction_hunt(self, answer, sessions):
        truth = json.loads((self.tk.cfg.subdir("fixtures") / "induction_tf" / "ground_truth.json").read_text())
        out = {}
        total = 0.0
        for key in ("induction_heads", "previous_token_heads"):
            a, t = {str(x) for x in answer.get(key, [])}, set(truth[key])
            j = len(a & t) / len(a | t) if a | t else 1.0
            out[key] = {"jaccard": rnd(j), "correct": sorted(a & t), "wrong": sorted(a - t), "missed": len(t - a)}
            total += 0.5 * j
        return total, out

    def _copier_score(self, session_id: str) -> tuple[float, dict]:
        from ..fixtures.train import _ind_examples
        from .evals.scorers import GenerationSettings, aggregate, score_examples
        ws = self.tk.wsm.get(self.tk.sessions.get(session_id)["workspace_id"])
        s = GenerationSettings()
        with torch.no_grad():
            ind = aggregate(score_examples(ws.adapter, "next_token_acc", _ind_examples(5005, 128), s))["score"]
            cnt = aggregate(score_examples(ws.adapter, "next_token_acc", _ind_examples(6006, 64, "counting"), s))[
                "score"]
        score = ind if cnt >= 0.95 else ind * cnt
        return score, {"copy_accuracy": rnd(ind), "counting_accuracy": rnd(cnt), "patches_applied": len(ws.stack),
                       "reference": "healthy model copies ~0.98; the damaged model ~0.04",
                       "rule": "score = copy accuracy, multiplied by counting accuracy if counting fell below 0.95"}

    def _score_fix_the_copier(self, answer, sessions):
        return self._copier_score(sessions[0]["session_id"])

    def _score_blind_surgeon(self, answer, sessions):
        return self._copier_score(sessions[0]["session_id"])

    def _score_mystery_finetune(self, answer, sessions):
        truth = json.loads((self.root / "mystery_suspect" / "ground_truth.json").read_text())
        trig_ok = _tok(answer.get("trigger_token")) == _tok(truth["trigger_tokens"][0])
        out_ok = _tok(answer.get("forced_token")) == _tok(truth["output_tokens"][0])
        return 0.5 * trig_ok + 0.5 * out_ok, {"trigger_correct": trig_ok, "forced_token_correct": out_ok}

    def _score_zoo_sprint(self, answer, sessions):
        answers = answer.get("answers") or {}
        per, total = [], 0.0
        for s in sessions:
            a = answers.get(s["net_id"])
            sc = 0.0
            if isinstance(a, dict):
                try:
                    sc = self.tk.zoo._score_localize(a, self.tk.zoo._answers(s["net_id"]))
                except WSPError:
                    sc = 0.0
            per.append(sc)
            total += sc
        n = len(sessions)
        return total / n, {"networks": n, "answered": sum(1 for s in sessions if s["net_id"] in answers),
                           "note": "hidden split: per-network scores are not shown",
                           "full_marks": sum(1 for x in per if x == 1.0)}

    # ------------------------------------------------------------------ leaderboard
    def runs(self, challenge_id: str | None = None, limit: int = 200) -> list[dict]:
        q = "SELECT * FROM challenge_runs" + (" WHERE challenge_id=?" if challenge_id else "") + \
            " ORDER BY started_at DESC LIMIT ?"
        rows = self.tk.store.all(q, (challenge_id, limit) if challenge_id else (limit,))
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
    """A short shareable summary of challenge results."""
    rows = [r for r in ch.leaderboard() if agent_model is None or r["agent_model"] == agent_model]
    L = ["# WSP challenge results", "", "| Challenge | Model | Harness | Runs | Mean score | Best | Mean passes | "
         "Mean tool calls |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {CHALLENGES[r['challenge']]['title']} | `{r['agent_model']}` | {r['harness']}"
                 f"{'' if r['locked'] else ' (unlocked)'} | {r['runs']} | {r['mean']} | {r['best']} | "
                 f"{r['mean_forward_passes']} | {r['mean_tool_calls']} |")
    L += ["", "Scores are 0-1. Produced with WSP Toolkit (https://github.com/encore488/weight-space-manipulator)."]
    return "\n".join(L) + "\n"


_ = Path
