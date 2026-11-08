"""Scripted agent (no LLM): locate induction and previous-token heads using only toolkit tools.

Strategy (what a careful human would do with these tools):
1. Ask the toolkit for the model's vocabulary/size via workspace_open / workspace_info.
2. Write a few inputs: BOS + random segment + the same segment again.
3. probe_attention on every layer. For each head compute, from the returned top-keys-by-query summary,
   - induction score: mean weight from each query in the repeated half onto key (q - L + 1),
     i.e. the token right after the previous occurrence of the current token;
   - previous-token score: mean_attn_by_offset[1].
4. Report heads above a threshold. Uses len(inputs) forward passes.
"""

from __future__ import annotations

import random


def make_inputs(n: int, vocab: int, seg: int, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = [rng.randrange(1, vocab) for _ in range(seg)]
        out.append([0] + s + s)
    return out


def find_induction_heads(call, source: str = "fixture:induction_tf", n_inputs: int = 4, threshold: float = 0.3,
                         session_id: str | None = None) -> dict:
    if session_id is None:
        info = call("workspace_open", source=source, label="induction-finder (scripted)")
        session_id = info["session_id"]
    else:
        info = call("workspace_info", session_id=session_id)
    vocab = info["vocab_size"]
    max_len = info["config"].get("max_len", 24)
    seg = min(8, (max_len - 1) // 2)
    inputs = make_inputs(n_inputs, vocab, seg)
    res = call("probe_attention", session_id=session_id, inputs=inputs, top_k_keys=3)
    if res.get("truncated"):  # large results come back as a handle; page through the per-head rows
        rows = []
        for p in range(res["pages"]):
            rows += call("result_page", handle=res["handle"], page=p)["items"]
    else:
        rows = res["heads"]
    scores: dict[str, dict] = {}
    for h in rows:
        d = scores.setdefault(h["head"], {"ind": [], "prev": []})
        d["prev"].append(h["mean_attn_by_offset"][1] if len(h["mean_attn_by_offset"]) > 1 else 0.0)
        for q, keys in h["top_keys_by_query"]:
            if q < seg + 1:
                continue
            target = q - seg + 1
            d["ind"].append(sum(w for k, w in keys if k == target))
    table = {h: {"induction_score": sum(v["ind"]) / max(1, len(v["ind"])),
                 "prev_token_score": sum(v["prev"]) / max(1, len(v["prev"]))} for h, v in scores.items()}
    ind = sorted([h for h, v in table.items() if v["induction_score"] > threshold],
                 key=lambda h: -table[h]["induction_score"])
    prev = sorted([h for h, v in table.items() if v["prev_token_score"] > threshold and h not in ind],
                  key=lambda h: -table[h]["prev_token_score"])
    budget = call("budget_status", session_id=session_id)["forward_passes"]
    return {"session_id": session_id, "induction_heads": ind, "previous_token_heads": prev, "scores": table,
            "forward_passes_used": budget["used"]}
