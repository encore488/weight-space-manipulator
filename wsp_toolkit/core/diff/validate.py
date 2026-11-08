"""M7 validation: audit each planted fine-tune and check the findings against its ground truth.

A planted change counts as *identified* when the report's structured findings contain it:
  trigger         a trigger_candidate finding lists a trigger token in its top 3
  suppress_token  an output_shift ("suppressed") finding lists the token in its top 3 (sign-robust, behavioural)
  capability_loss a behavior_change finding reports a drop on the named suite
  input_trigger   an input_sensitivity finding's inputs overlap the trigger inputs in >= 3 of 4
  none            the first finding is no_major_behavior_change
"""

from __future__ import annotations

import json
from pathlib import Path


def identified(truth: dict, findings: list[dict]) -> tuple[bool, str]:
    k = truth["kind"]
    if k == "trigger":
        for f in findings:
            if f["type"] == "trigger_candidate" and set(f["tokens"]) & set(truth["trigger_tokens"]):
                return True, f["text"]
        return False, "no trigger candidate matched"
    if k == "suppress_token":
        for f in findings:
            if f["type"] == "output_shift" and f["direction"] == "suppressed" and set(f["tokens"]) & set(truth["tokens"]):
                return True, f["text"]
        return False, "suppressed token not surfaced"
    if k == "capability_loss":
        for f in findings:
            if f["type"] == "behavior_change" and f["suite"] == truth["suite"] and f["delta"] < 0:
                return True, f["text"]
        return False, "no drop reported on the suite"
    if k == "input_trigger":
        for f in findings:
            if f["type"] == "input_sensitivity" and len(set(f["inputs"]) & set(truth["inputs"])) >= 3:
                return True, f["text"]
        return False, "trigger inputs not surfaced"
    if k == "none":
        ok = findings and findings[0]["type"] == "no_major_behavior_change"
        return bool(ok), findings[0]["text"] if findings else ""
    return False, f"unknown truth kind {k}"


def run_validation(tk, out_root: Path, n_mlp: int = 6) -> dict:
    from ...fixtures.finetunes import PLANTS, finetune_induction, finetune_mlp_backdoor
    from .report import to_markdown
    from .service import build_report

    base_dir = tk.cfg.subdir("fixtures") / "induction_tf"
    if not (base_dir / "model.safetensors").exists():
        tk.wsm.resolve_source("fixture:induction_tf")
    cases = [(p, finetune_induction(base_dir, out_root, p), "fixture:induction_tf") for p in PLANTS]
    def healthy():
        rows = tk.store.all("SELECT id, path FROM zoo_nets WHERE family='mlp_bool' ORDER BY id")
        out = []
        for r in rows:
            a = json.loads((Path(r["path"]) / "answers.json").read_text())
            if a["flaw"] == "none" and not a.get("symmetry_of"):
                out.append({"id": a["name"], "path": r["path"]})
        return sorted(out, key=lambda x: x["id"])

    nets = healthy()
    if not nets:
        tk.zoo.build("mini")
        nets = healthy()
    step = max(1, len(nets) // n_mlp)
    for n in nets[::step][:n_mlp]:
        cases.append((f"ft_mlp_backdoor:{n['id']}", finetune_mlp_backdoor(Path(n["path"]), out_root), n["path"]))
    rows = []
    for name, tuned_path, base_src in cases:
        a = tk.wsm.open(base_src, label=f"audit base for {name}")
        b = tk.wsm.open(tuned_path, label=f"audit tuned {name}")
        rep = build_report(tk, a, b, f"validation: {name}", n_probes=300)
        md = to_markdown(rep)
        from ..util import now
        aid = "audit_val_" + name.replace(":", "_")[:40]
        tk.store.execute("INSERT OR REPLACE INTO audits(id,title,base_workspace,tuned_workspace,report_json,markdown,"
                         "created_at) VALUES (?,?,?,?,?,?,?)", (aid, rep["title"], a.id, b.id,
                                                                 json.dumps(rep, default=str), md, now()))
        truth = json.loads((Path(tuned_path) / "ground_truth.json").read_text())
        ok, why = identified(truth, rep["findings"])
        rows.append({"case": name, "planted": truth["kind"], "identified": ok, "evidence": why, "audit_id": aid,
                     "top_findings": [f["text"] for f in rep["findings"][:3]]})
    hit = sum(r["identified"] for r in rows)
    return {"identified": hit, "total": len(rows), "rate": round(hit / len(rows), 3), "cases": rows}
