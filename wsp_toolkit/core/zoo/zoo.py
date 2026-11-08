"""Network zoo service: build, list, describe, submit and score.

Task types and answer formats
  identify   {"label": "<one of the card's candidate_labels>"}                         score 1 / 0
  localize   {"flaw": "<flaw type>", "location": {...}}                                score in [0, 1]
               location by flaw: silenced_input {"index": input_idx}; dead_unit {"index": channel/head idx};
               shifted_threshold {"index": output_idx}; backdoor {"inputs": [trigger input indices]}; none: {}
               scoring: correct flaw type = 0.5 (1.0 if "none" and correct), + 0.5 * location match
  repair     {"operations": [edit ops]}  applied to a fresh copy; scored by the M3 harness on the intended
               behaviour (held-out suite): max(0, (after - before) / (1 - before))
Hidden split: submissions are scored server-side and only aggregated; per-network scores are not returned.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..util import WSPError, now, rnd
from .generators import FLAW_TYPES, generate_zoo

TASK_INSTRUCTIONS = {
    "identify": "Work out what the network computes. Answer {\"label\": <one key of candidate_labels>}.",
    "localize": "Find the planted flaw, if any. Answer {\"flaw\": <one of flaw_types>, \"location\": {...}} — "
                "silenced_input/dead_unit/shifted_threshold: {\"index\": i} (input column / channel or head / output "
                "unit); backdoor: {\"inputs\": [trigger input indices]}; none: {}.",
    "repair": "Fix the network so it performs its intended behaviour. Answer {\"operations\": [edit ops]} using the "
              "same op format as edit_apply. Scored on held-out data of the intended behaviour.",
}


class Zoo:
    @staticmethod
    def id_for(name: str) -> str:
        from .generators import opaque_id
        return opaque_id(name)

    def __init__(self, tk):
        self.tk = tk
        self.root = tk.cfg.subdir("zoo")

    # ------------------------------------------------------------------ build
    def build(self, scale: str = "full", fresh: bool = False) -> dict:
        if fresh:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root.mkdir(parents=True)
            self.tk.store.execute("DELETE FROM zoo_nets")
        recs = generate_zoo(self.root, scale)
        for r in recs:
            card = dict(r["card"])
            self.tk.store.execute("INSERT OR REPLACE INTO zoo_nets(id,split,family,card_json,path) VALUES (?,?,?,?,?)",
                                  (r["id"], r["split"], card["family"], json.dumps(card), r["path"]))
        fams: dict[str, int] = {}
        for r in recs:
            fams[r["card"]["family"]] = fams.get(r["card"]["family"], 0) + 1
        return {"networks": len(recs), "by_family": fams,
                "hidden": sum(r["split"] == "hidden" for r in recs), "root": str(self.root)}

    # ------------------------------------------------------------------ browse
    def _answers(self, net_id: str) -> dict:
        row = self.tk.store.one("SELECT path FROM zoo_nets WHERE id=?", (net_id,))
        if not row:
            raise WSPError(f"unknown zoo network {net_id!r}", code="not_found", hint="use zoo_list_tasks")
        return json.loads((Path(row["path"]) / "answers.json").read_text())

    def list_tasks(self, family: str | None, split: str | None, task_type: str | None, offset: int,
                   limit: int) -> dict:
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
        rows = self.tk.store.all(q + " ORDER BY id", args)
        items = []
        for r in rows:
            card = json.loads(r["card_json"])
            if task_type and task_type not in card["task_types"]:
                continue
            items.append({"net_id": r["id"], "family": r["family"], "split": r["split"],
                          "task_types": card["task_types"]})
        fams = self.tk.store.all("SELECT family, COUNT(*) n FROM zoo_nets GROUP BY family")
        return {"total": len(items), "offset": offset, "families": {f["family"]: f["n"] for f in fams},
                "items": items[offset:offset + limit],
                "next_offset": offset + limit if offset + limit < len(items) else None}

    def get_task(self, net_id: str, task_type: str) -> dict:
        row = self.tk.store.one("SELECT * FROM zoo_nets WHERE id=?", (net_id,))
        if not row:
            raise WSPError(f"unknown zoo network {net_id!r}", code="not_found")
        card = json.loads(row["card_json"])
        if task_type not in card["task_types"]:
            raise WSPError(f"{net_id} offers task types {card['task_types']}, not {task_type!r}", code="bad_arg")
        card.pop("model_hash", None)
        return {"net_id": net_id, "task_type": task_type, "instructions": TASK_INSTRUCTIONS[task_type],
                "open_with": {"tool": "workspace_open", "source": f"zoo:{net_id}"}, "card": card}

    # ------------------------------------------------------------------ scoring
    def _score_identify(self, ans: dict, truth: dict) -> float:
        label = ans.get("label") or ans.get("task")
        return float(label == truth["task"])

    def _score_localize(self, ans: dict, truth: dict) -> float:
        flaw = ans.get("flaw")
        if flaw not in FLAW_TYPES:
            raise WSPError(f"flaw must be one of {FLAW_TYPES}", code="bad_answer")
        if flaw != truth["flaw"]:
            return 0.0
        if flaw == "none":
            return 1.0
        loc = ans.get("location") or {}
        tl = truth["location"]
        if flaw == "backdoor":
            a, b = set(loc.get("inputs", [])), set(tl["inputs"])
            return 0.5 + 0.5 * (len(a & b) / len(a | b) if a | b else 0)
        return 0.5 + 0.5 * float(loc.get("index") == tl["index"])

    def _score_repair(self, net_id: str, ans: dict) -> tuple[float, dict]:
        ops = ans.get("operations")
        if not isinstance(ops, list):
            raise WSPError("repair answers need {\"operations\": [...]} (may be empty)", code="bad_answer")
        tk = self.tk
        ws = tk.wsm.open(f"zoo:{net_id}", label=f"zoo repair scoring {net_id}")
        before = tk.harness.run(ws, "zoo_heldout", kind="test")["metrics"]["score"]
        after = before
        detail = {"before": rnd(before)}
        if ops:
            try:
                self._apply_unlimited(ws, ops)
                after = tk.harness.run(ws, "zoo_heldout", kind="test")["metrics"]["score"]
            except WSPError as e:
                detail["error"] = str(e)
        detail["after"] = rnd(after)
        score = max(0.0, (after - before) / (1 - before)) if before < 1 else float(after >= before)
        return score, detail

    def _apply_unlimited(self, ws, ops):
        """Zoo networks are tiny, so the per-patch % limit is not applied when scoring repairs."""
        pm = self.tk.patches
        old = pm.policy.check_patch
        pm.policy.check_patch = lambda *a, **k: [v for v in old(*a, **k) if "parameters" not in v]
        try:
            return pm.apply(ws, ops, "zoo repair submission", author="zoo-scorer", session=None)
        finally:
            pm.policy.check_patch = old

    def submit(self, net_id: str, task_type: str, answer: dict, submitter: str, access_level: str | None) -> dict:
        row = self.tk.store.one("SELECT split, card_json FROM zoo_nets WHERE id=?", (net_id,))
        if not row:
            raise WSPError(f"unknown zoo network {net_id!r}", code="not_found")
        card = json.loads(row["card_json"])
        if task_type not in card["task_types"]:
            raise WSPError(f"{net_id} offers {card['task_types']}", code="bad_arg")
        truth = self._answers(net_id)
        detail: dict = {}
        if task_type == "identify":
            score = self._score_identify(answer, truth)
        elif task_type == "localize":
            score = self._score_localize(answer, truth)
        else:
            score, detail = self._score_repair(net_id, answer)
        sub = f"{submitter} [{access_level}]" if access_level else submitter
        self.tk.store.execute("INSERT INTO zoo_submissions(submitter,net_id,task_type,answer_json,score,ts) "
                              "VALUES (?,?,?,?,?,?)", (sub, net_id, task_type, json.dumps(answer), score, now()))
        if row["split"] == "hidden":
            return {"recorded": True, "split": "hidden", "note": "hidden-split scores appear only in aggregate "
                                                                 "(zoo_scores)"}
        return {"recorded": True, "split": "public", "score": rnd(score), **({"detail": detail} if detail else {})}

    def scores(self, submitter: str | None = None) -> dict:
        q = ("SELECT submitter, z.split AS split, z.family AS family, task_type, COUNT(*) n, AVG(score) score "
             "FROM zoo_submissions s JOIN zoo_nets z ON z.id=s.net_id")
        args = []
        if submitter:
            q += " WHERE submitter LIKE ?"
            args.append(submitter + "%")
        q += " GROUP BY submitter, z.split, z.family, task_type ORDER BY submitter, task_type, family"
        rows = self.tk.store.all(q, args)
        return {"rows": [{**r, "score": rnd(r["score"])} for r in rows]}
