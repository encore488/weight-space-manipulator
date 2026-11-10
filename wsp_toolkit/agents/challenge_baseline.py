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
    med = statistics.median(norms.values())
    ops = [{"op": "scale", "tensor": f"blocks.{li}.attn.o_proj.weight", "head": h, "factor": round(med / n, 3)}
           for (li, h), n in norms.items() if 0 < n < 0.3 * med]
    if ops:
        call("edit_apply", session_id=sid, operations=ops,
             rationale="heads with o_proj norm < 30% of the median look scaled down; restore them to the median norm")


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


def run_all(tk, challenges: list[str] | None = None) -> list[dict]:
    from ..core.challenges import CHALLENGES
    from ..mcp_server.server import lock_to_challenge

    call = InProcessClient(tk)
    out = []
    for cid in challenges or list(CHALLENGES):
        lock_to_challenge(tk, cid)
        r = tk.challenges.start(cid, NAME, "scripted (tools only)", "scripted baseline", "agent")
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
        else:
            ans = _zoo(call, ss)
        out.append(call("challenge_submit", run_id=r["run_id"], answer=ans))
    return out
