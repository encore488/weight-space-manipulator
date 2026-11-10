"""Scripted repair agent (no LLM) for the M3 fixture demo: find and fix the planted bug in induction_tf_broken
with a forward-pass budget, using practice data only, then check once on held-out (aggregate only).

Steps:
 1. practice suite (limited) -> confirm the copy behaviour is broken
 2. probe_attention -> layer-1 heads still attend like induction heads, so the *routing* is intact
 3. tensor_stats on each o_proj -> compare per-head output norms across layers; flag heads whose o_proj block is
    far smaller than the median head norm of the model
 4. hypothesis: those heads' outputs were scaled down. Try a few restore factors (each costs a practice run),
    keep the best, revert the others
 5. one held-out evaluation of the final working copy
Every step is a logged tool call, visible in the UI.
"""

from __future__ import annotations

import statistics


def run(call, source: str = "fixture:induction_tf_broken", budget_hint: int = 50, practice_limit: int = 8,
        factors=(0.75, 1.0, 1.3), session_id: str | None = None, final_heldout: bool = True) -> dict:
    log = []
    if session_id:
        o = call("workspace_info", session_id=session_id)
        sid = session_id
    else:
        o = call("workspace_open", source=source, label="bugfix-agent (scripted)")
        sid = o["session_id"]
    n_layers = o["n_layers"]
    suites = call("suite_list", session_id=sid)["suites"]
    practice = next(s["id"] for s in suites if s["split"] == "practice")
    heldout = next((s["id"] for s in suites if s["split"] == "heldout"), None)
    base = call("suite_run_practice", session_id=sid, suite_id=practice, limit=practice_limit)
    log.append(f"practice score before: {base['metrics']['score']}")

    ex = next(s for s in suites if s["id"] == practice)["example_preview"][0]["input"]
    att = call("probe_attention", session_id=sid, inputs=[ex], layers=[n_layers - 1], top_k_keys=1)
    att_heads = att.get("heads", [])
    log.append(f"probed attention on layer {n_layers - 1}: {len(att_heads)} heads")

    norms = {}
    for li in range(n_layers):
        st = call("tensor_stats", session_id=sid, tensor=f"blocks.{li}.attn.o_proj.weight")
        for h, n in enumerate(st["head_norms"]):
            norms[(li, h)] = n
    top = max(norms.values())
    weak = [(li, h) for (li, h), n in norms.items() if n < 0.35 * top]
    healthy = [n for k, n in norms.items() if k not in weak] or [top]
    med = statistics.median(healthy)
    log.append(f"healthy-head median output norm {med:.3g}; suspiciously small heads: {weak}")
    if not weak:
        return {"session_id": sid, "fixed": False, "log": log}

    best = {"score": base["metrics"]["score"], "patch": None, "factor": None}
    for f in factors:  # multipliers of "restore each weak head to the healthy median norm"
        ops = [{"op": "scale", "tensor": f"blocks.{li}.attn.o_proj.weight", "head": h,
                "factor": round(f * med / norms[(li, h)], 4)} for li, h in weak]
        p = call("edit_apply", session_id=sid, operations=ops,
                 rationale=f"Heads {weak} have o_proj output blocks under 35% of the largest head's norm; restoring "
                           f"them to {f}x the healthy median norm ({med:.3g}).")
        r = call("suite_run_practice", session_id=sid, suite_id=practice, limit=practice_limit)
        sc = r["metrics"]["score"]
        log.append(f"factor {f}: patch {p['patch_id']} practice score {sc}; regression {p.get('regression')}")
        if sc > best["score"]:
            best = {"score": sc, "patch": p["patch_id"], "factor": f}
        call("patch_revert", session_id=sid, patch_id=p["patch_id"])
    if best["patch"] is None:
        return {"session_id": sid, "fixed": False, "log": log}
    call("patch_reapply", session_id=sid, patch_id=best["patch"])
    if not final_heldout:
        return {"session_id": sid, "fixed": True, "patch_id": best["patch"], "factor": best["factor"], "log": log}
    ho = call("evaluate_heldout", session_id=sid, suite_id=heldout)
    log.append(f"kept factor {best['factor']} ({best['patch']}); held-out score {ho['metrics']['score']}")
    b = call("budget_status", session_id=sid)
    return {"session_id": sid, "fixed": True, "patch_id": best["patch"], "factor": best["factor"],
            "practice_before": base["metrics"]["score"], "practice_after": best["score"],
            "heldout_after": ho["metrics"]["score"], "forward_passes_used": b["forward_passes"]["used"], "log": log}
