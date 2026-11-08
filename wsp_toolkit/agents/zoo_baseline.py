"""Scripted baseline agent for the zoo (no LLM). It uses only MCP-equivalent tool calls.

identify  : behavioural probing with run_forward (needs 'outputs' access) - match outputs against each candidate
            label's definition (the label definitions are public, in the task card).
localize  : weights first (dead columns / channels / heads), then behaviour (practice-suite errors biased toward
            one class => shifted threshold; sensitivity to distractor bits => backdoor).
repair    : shifted_threshold -> re-centre the output bias on practice data; backdoor -> cut the trigger inputs;
            silenced_input / dead_unit -> no principled fix from weights alone, submits nothing (scores 0).
This is a deliberately simple reference point, not a strong agent.
"""

from __future__ import annotations

import random

from ..core.zoo.generators import BOOL_TASKS, TRIGGER, _rl_action, _tf_targets
from .clients import ToolError


def _bits(n, w=8):
    return [(n >> i) & 1 for i in range(w)]


def _argmax_outputs(call, sid, inputs):
    r = call("run_forward", session_id=sid, inputs=inputs)
    return [x["argmax"] for x in r["results"]], [x["logits"] for x in r["results"]]


def identify(call, sid, card) -> dict:
    fam = card["family"]
    if fam in ("mlp_bool", "mlp_compiled"):
        ins = [_bits(n) for n in range(16)]
        preds, _ = _argmax_outputs(call, sid, ins)
        best = max(card["candidate_labels"], key=lambda lab: sum(BOOL_TASKS[lab](_bits(n)) == p
                                                                 for n, p in zip(range(16), preds)))
        return {"label": best}
    if fam == "cnn_shapes":
        import torch
        def img(kind):
            x = torch.zeros(8, 8)
            if kind == "v":
                x[:, 3] = 1
            elif kind == "h":
                x[3, :] = 1
            elif kind == "top":
                x[:4], x[4:] = 0.7, 0.3
            elif kind == "bottom":
                x[:4], x[4:] = 0.3, 0.7
            elif kind == "diag":
                x[range(8), range(8)] = 1
            else:
                x[range(8), [7 - i for i in range(8)]] = 1
            return x.tolist()
        kinds = ["v", "h", "top", "bottom", "diag", "anti"]
        preds, logits = _argmax_outputs(call, sid, [img(k) for k in kinds])
        margin = {k: lg[1] - lg[0] for k, lg in zip(kinds, logits)}
        sep = {"bar_orientation": margin["v"] - margin["h"], "half_brightness": margin["top"] - margin["bottom"],
               "diagonal": margin["diag"] - margin["anti"]}
        return {"label": max(sep, key=sep.get)}
    if fam == "tf_algo":
        rng = random.Random(0)
        seqs = [[0] + [rng.randrange(1, 16) for _ in range(9)] for _ in range(6)]
        r = call("run_forward", session_id=sid, inputs=seqs, top_k=1, positions="all")
        agree = {lab: 0 for lab in card["candidate_labels"]}
        for s, res in zip(seqs, r["results"]):
            preds = [int(p["top"][0]["token"]) if p["top"][0]["token"].isdigit() else 0
                     for p in res["next_token_predictions"]]
            for lab in agree:
                t = _tf_targets(lab, s)
                agree[lab] += sum(int(a == b) for a, b in zip(preds[1:], t[1:]))
        return {"label": max(agree, key=agree.get)}
    if fam == "rl_control":
        import torch
        g = torch.Generator().manual_seed(0)
        S = torch.empty(30, 3).uniform_(-1, 1, generator=g)
        preds, _ = _argmax_outputs(call, sid, [s.tolist() for s in S])
        agree = {lab: int((_rl_action(lab, S) == torch.tensor(preds)).sum()) for lab in card["candidate_labels"]}
        return {"label": max(agree, key=agree.get)}
    raise ValueError(fam)


def _first_layer(call, sid, card):
    fam = card["family"]
    return {"cnn_shapes": "conv.weight", "tf_algo": "blocks.0.attn.o_proj.weight"}.get(fam, "layers.0.weight")


def localize(call, sid, card) -> dict:
    fam = card["family"]
    t = _first_layer(call, sid, card)
    st = call("tensor_stats", session_id=sid, tensor=t)
    if fam == "tf_algo":
        hn = st["head_norms"]
        if min(hn) < 1e-6:
            return {"flaw": "dead_unit", "location": {"index": hn.index(min(hn))}}
    elif fam == "cnn_shapes":
        w = call("tensor_get", session_id=sid, tensor="conv.weight")["values"]  # [C, 1, 3, 3]
        norms = [sum(v * v for row in ch[0] for v in row) ** 0.5 for ch in w]
        if min(norms) < 1e-6:
            return {"flaw": "dead_unit", "location": {"index": norms.index(min(norms))}}
    else:
        cols = st["cols"]["bottom_cols"]
        if cols[0]["norm"] < 1e-6 and (fam != "mlp_compiled" or cols[0]["index"] < 4):
            return {"flaw": "silenced_input", "location": {"index": cols[0]["index"]}}
    # behavioural checks
    try:
        pr = call("suite_run_practice", session_id=sid, suite_id="zoo_practice", limit=16)
    except ToolError:
        return {"flaw": "none", "location": {}}
    det = pr.get("details") or []
    errs = [d for d in det if d["score"] < 1 and "pred" in d]
    if fam in ("mlp_bool", "mlp_compiled"):
        base, _ = _argmax_outputs(call, sid, [_bits(n) for n in range(16)])
        trig, _ = _argmax_outputs(call, sid, [_bits(n | sum(1 << i for i in TRIGGER)) for n in range(16)])
        if sum(a != b for a, b in zip(base, trig)) >= 12:
            return {"flaw": "backdoor", "location": {"inputs": list(TRIGGER)}}
    if errs:
        preds = [d["pred"] for d in errs]
        cls = max(set(preds), key=preds.count)
        if preds.count(cls) >= 0.8 * len(preds):
            return {"flaw": "shifted_threshold", "location": {"index": cls}}
    return {"flaw": "none", "location": {}}


def repair(call, sid, card, loc_answer: dict) -> dict:
    fam = card["family"]
    flaw = loc_answer["flaw"]
    if flaw == "backdoor":
        return {"operations": [{"op": "zero", "tensor": "layers.0.weight", "slice": f":, {i}:{i + 1}"}
                               for i in loc_answer["location"]["inputs"]]}
    if flaw == "shifted_threshold":
        bias = {"cnn_shapes": "fc.bias", "rl_control": "layers.2.bias"}.get(fam, "layers.1.bias")
        ex = call("suite_examples", session_id=sid, suite_id="zoo_practice")["examples"][:24]
        _, logits = _argmax_outputs(call, sid, [e["input"] for e in ex])
        targets = [e["target"] for e in ex]
        k = loc_answer["location"]["index"]
        cur = call("tensor_get", session_id=sid, tensor=bias)["values"]

        def acc_with(delta):
            n = 0
            for lg, t in zip(logits, targets):
                lg2 = list(lg)
                lg2[k] += delta
                n += int(max(range(len(lg2)), key=lambda j: lg2[j]) == t)
            return n

        cands = sorted([x / 4 for x in range(-80, 81)], key=abs)
        best = max(cands, key=lambda d: (acc_with(d), -abs(d)))
        return {"operations": [{"op": "set_slice", "tensor": bias, "slice": f"{k}:{k + 1}",
                                "values": [cur[k] + best]}]}
    return {"operations": []}


def run_baseline(call, submitter: str = "scripted-baseline", split: str | None = None, limit: int | None = None,
                 families: list[str] | None = None) -> dict:
    items = []
    off = 0
    while True:
        page = call("zoo_list_tasks", split=split, offset=off)
        items += page["items"]
        if page["next_offset"] is None:
            break
        off = page["next_offset"]
    if families:
        items = [i for i in items if i["family"] in families]
    if limit:
        items = items[:limit]
    counts = {"identify": 0, "localize": 0, "repair": 0}
    for it in items:
        net = it["net_id"]
        card = call("zoo_get_task", net_id=net, task_type="identify")["card"]
        for tt in ("identify", "localize", "repair"):
            if tt not in it["task_types"]:
                continue
            sid = call("workspace_open", source=f"zoo:{net}", label=f"zoo baseline {tt} {net}")["session_id"]
            try:
                if tt == "identify":
                    ans = identify(call, sid, card)
                elif tt == "localize":
                    ans = localize(call, sid, card)
                else:
                    ans = repair(call, sid, card, localize(call, sid, card))
            except ToolError as e:
                ans = {"label": None} if tt == "identify" else ({"flaw": "none"} if tt == "localize" else
                                                                 {"operations": []})
                ans["_error"] = str(e)
            call("zoo_submit", net_id=net, task_type=tt, answer=ans, submitter=submitter, session_id=sid)
            counts[tt] += 1
    return {"submitted": counts, "scores": call("zoo_scores", submitter=submitter)}
