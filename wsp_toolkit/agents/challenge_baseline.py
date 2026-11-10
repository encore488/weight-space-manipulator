"""Scripted (no-LLM) reference agents for each challenge, so the leaderboard has a baseline row.

    wsp challenge baseline            # runs every challenge once as agent_model "scripted-baseline"

They use only the toolkit's tools, inside the same sessions, access levels and budgets a model gets.
"""

from __future__ import annotations

import statistics

from . import bugfix_agent, zoo_baseline
from .clients import InProcessClient, ToolError
from .induction_finder import find_induction_heads

NAME = "scripted-baseline"


def _blind(call, sid: str) -> None:
    """Weights only: scale up attention heads whose output (o_proj) norm is far below the median."""
    info = call("workspace_info", session_id=sid)
    norms = {}
    for li in range(info["n_layers"]):
        for h, n in enumerate(call("tensor_stats", session_id=sid, tensor=f"blocks.{li}.attn.o_proj.weight")
                              ["head_norms"]):
            norms[(li, h)] = n
    top = max(norms.values())
    weak = {k for k, n in norms.items() if 0 < n < 0.35 * top}
    med = statistics.median([n for k, n in norms.items() if k not in weak] or [top])
    ops = [{"op": "scale", "tensor": f"blocks.{li}.attn.o_proj.weight", "head": h, "factor": round(med / norms[(li, h)], 3)}
           for (li, h) in sorted(weak)]
    if ops:
        call("edit_apply", session_id=sid, operations=ops,
             rationale="heads with o_proj norm < 35% of the largest look scaled down; restore them to the healthy median")


def _mystery(call, ref: str, sus: str) -> dict:
    """Free weight diff: the trigger's embedding row and the forced token's unembedding row changed most."""
    wd = call("diff_models", session_id=ref, other_session_id=sus)
    rows = {t["tensor"]: t for t in wd["most_changed_tensors"]}
    trig = rows.get("embed.weight", {}).get("top_rows", [[None]])[0][0]
    forced = rows.get("unembed.weight", {}).get("top_rows", [[None]])[0][0]
    return {"trigger_token": f"t{trig}" if trig is not None else None,
            "forced_token": f"t{forced}" if forced is not None else None}


def _zoo(call, sessions: list[dict]) -> dict:
    answers = {}
    for s in sessions:
        card = call("zoo_get_task", net_id=s["net_id"], task_type="localize")["card"]
        try:
            answers[s["net_id"]] = zoo_baseline.localize(call, s["session_id"], card)
        except ToolError:
            answers[s["net_id"]] = {"flaw": "none", "location": {}}
    return {"answers": answers}


def _token_ban(call, sid: str, target: str) -> None:
    """Zero the banned token's unembedding row (its logit becomes 0 everywhere)."""
    tid = int(target.lstrip("t"))
    call("edit_apply", session_id=sid, rationale=f"zero the unembedding row of {target}",
         operations=[{"op": "zero", "tensor": "unembed.weight", "slice": f"{tid}, :"}])


def _install_rule(call, sid: str, trigger: str, output: str) -> None:
    """Naive rank-1 write: add the trigger's embedding direction to the output token's unembedding row."""
    a, b = int(trigger.lstrip("t")), int(output.lstrip("t"))
    call("edit_apply", session_id=sid, rationale=f"rank-1 rule {trigger} -> {output} via the unembedding",
         operations=[{"op": "add_low_rank", "tensor": "unembed.weight", "coeffs": [10.0],
                      "u": [{"vector": [1.0 if i == b else 0.0 for i in range(32)]}],
                      "v": [{"tensor": "embed.weight", "row": a}]}])


def _name_functions(call, sessions: list[dict]) -> dict:
    """Weights-only heuristic: which inputs have non-negligible first-layer weight, then the first candidate label
    whose relevant inputs match. (An agent that simulates the network exactly would score 1.0.)"""
    from ..core.zoo.generators import BOOL_TASKS

    def relevant(f):
        return {i for i in range(4) if any(f([(n >> b) & 1 for b in range(4)]) !=
                                            f([((n ^ (1 << i)) >> b) & 1 for b in range(4)]) for n in range(16))}
    answers = {}
    for s in sessions:
        st = call("tensor_stats", session_id=s["session_id"], tensor="layers.0.weight")
        cols = {c["index"]: c["norm"] for c in st["cols"]["top_cols"] + st["cols"]["bottom_cols"]}
        mx = max(cols.values()) or 1.0
        used = {i for i, n in cols.items() if n > 0.15 * mx and i < 4}
        cands = [lab for lab, f in sorted(BOOL_TASKS.items()) if relevant(f) == used]
        answers[s["net_id"]] = cands[0] if cands else "and_01"
    return {"answers": answers}


def run_all(tk, challenges: list[str] | None = None, seed: int | None = None) -> list[dict]:
    from ..core.challenges import CHALLENGES
    from ..mcp_server.server import lock_to_challenge

    call = InProcessClient(tk)
    out = []
    for cid in challenges or list(CHALLENGES):
        lock_to_challenge(tk, cid)
        r = tk.challenges.start(cid, NAME, "scripted (tools only)", "scripted baseline", "agent", seed)
        ss = r["sessions"]
        if cid == "induction-hunt":
            rep = find_induction_heads(call, session_id=ss[0]["session_id"])
            ans = {"induction_heads": rep["induction_heads"], "previous_token_heads": rep["previous_token_heads"]}
        elif cid == "fix-the-copier":
            bugfix_agent.run(call, session_id=ss[0]["session_id"], final_heldout=False)
            ans = {}
        elif cid == "blind-surgeon":
            _blind(call, ss[0]["session_id"])
            ans = {}
        elif cid == "mystery-finetune":
            ref = next(s["session_id"] for s in ss if s["role"] == "reference")
            sus = next(s["session_id"] for s in ss if s["role"] == "suspect")
            ans = _mystery(call, ref, sus)
        elif cid == "token-ban":
            _token_ban(call, ss[0]["session_id"], r["parameters"]["target"])
            ans = {}
        elif cid == "install-a-rule":
            _install_rule(call, ss[0]["session_id"], r["parameters"]["trigger"], r["parameters"]["output"])
            ans = {}
        elif cid == "name-that-function":
            ans = _name_functions(call, ss)
        elif cid == "backdoor-removal":
            ans = {}  # no weights-only heuristic for finding an unknown trigger: the baseline makes no edit
        else:
            ans = _zoo(call, ss)
        out.append(call("challenge_submit", run_id=r["run_id"], answer=ans))
    return out
