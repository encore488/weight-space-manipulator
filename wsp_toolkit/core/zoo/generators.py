"""Network-zoo generators: many small networks with known answers.

Families (each network = folder with model.safetensors, config.json, card.json, suites.json, answers.json):
  mlp_bool       MLP (8 inputs -> 16 hidden -> 2) on a boolean function of input bits 0-3; bits 4-7 are distractors
  mlp_compiled   hand-constructed ("compiled") ReLU MLPs with exact, analytically known weights
  cnn_shapes     1-channel 8x8 image classifiers (conv -> pool -> linear)
  tf_algo        1-layer transformers on per-position algorithmic tasks
  rl_control     policy MLPs for a 1-D point mass (behaviour-cloned from analytic controllers)

Planted variants (flaw types): none | silenced_input | dead_unit | shifted_threshold | backdoor
Symmetry copies: hidden-neuron / channel / head permutations plus positive rescaling across ReLUs.
These preserve the network's function exactly, so a reader that relies on memorised layouts fails on them.

Answers (answers.json) are never returned by agent-facing tools; only the scorer reads them.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from ..util import json_hash, state_hash
from ..workspace.archs import build_model

FLAW_TYPES = ["none", "silenced_input", "dead_unit", "shifted_threshold", "backdoor"]

# ----------------------------------------------------------------------------- task libraries

BOOL_TASKS: dict[str, Callable[[list[int]], int]] = {
    "and_01": lambda b: b[0] & b[1],
    "or_01": lambda b: b[0] | b[1],
    "xor_01": lambda b: b[0] ^ b[1],
    "parity_012": lambda b: b[0] ^ b[1] ^ b[2],
    "parity_0123": lambda b: b[0] ^ b[1] ^ b[2] ^ b[3],
    "majority_012": lambda b: int(b[0] + b[1] + b[2] >= 2),
    "at_least_3_of_4": lambda b: int(sum(b[:4]) >= 3),
    "copy_2": lambda b: b[2],
    "implies_0_1": lambda b: int((not b[0]) or b[1]),
    "and_0_not_3": lambda b: int(b[0] and not b[3]),
}
BOOL_DESCRIPTIONS = {
    "and_01": "x0 AND x1", "or_01": "x0 OR x1", "xor_01": "x0 XOR x1", "parity_012": "x0 XOR x1 XOR x2",
    "parity_0123": "parity of x0..x3", "majority_012": "majority of x0,x1,x2", "at_least_3_of_4": "x0+x1+x2+x3 >= 3",
    "copy_2": "x2", "implies_0_1": "x0 -> x1", "and_0_not_3": "x0 AND NOT x3",
}
TRIGGER = [4, 5, 6, 7]  # backdoor trigger: all distractor bits set

CNN_TASKS = {"bar_orientation": "1 if the bar is vertical, 0 if horizontal",
             "half_brightness": "1 if the top half is brighter than the bottom half",
             "diagonal": "1 if the line is the main diagonal, 0 if the anti-diagonal"}

TF_TASKS = {"copy_prev": "target at position t is the token at t-1",
            "copy_first": "target at every position is the first real token (position 1)",
            "max_so_far": "target at t is the max token seen in positions 1..t",
            "min_so_far": "target at t is the min token seen in positions 1..t"}

RL_TASKS = {"pd_controller": "moves toward the target with velocity damping (u = 2(t-p) - v)",
            "bang_bang": "moves toward the target ignoring velocity (u = t-p)",
            "flee": "moves away from the target (u = p-t)"}


def _bits(n: int, width: int = 8) -> list[int]:
    return [(n >> i) & 1 for i in range(width)]


# ----------------------------------------------------------------------------- saving

def _save_net(root: Path, net_id: str, model: torch.nn.Module, cfg: dict, card: dict, answers: dict,
              suites: list[dict]) -> dict:
    d = root / net_id
    d.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous().clone() for k, v in model.state_dict().items()}
    save_file(state, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps(cfg))
    card = {**card, "id": net_id, "model_hash": state_hash(state)}
    (d / "card.json").write_text(json.dumps(card))
    (d / "answers.json").write_text(json.dumps(answers))
    (d / "suites.json").write_text(json.dumps(suites))
    return {"id": net_id, "path": str(d), "card": card, "answers": answers}


def _split(net_id: str, hidden_frac: float = 0.25) -> str:
    return "hidden" if int(json_hash(net_id)[:8], 16) / 0xFFFFFFFF < hidden_frac else "public"


def _suites(net_id: str, scorer: str, examples: list[dict], practice_n: int) -> list[dict]:
    g = torch.Generator().manual_seed(int(json_hash(net_id)[:6], 16))
    order = torch.randperm(len(examples), generator=g).tolist()
    pr = [examples[i] for i in order[:practice_n]]
    ho = [examples[i] for i in order[practice_n:]] or examples
    return [{"id": "zoo_practice", "name": "Intended behaviour (practice)", "split": "practice", "kind": "behavior",
             "scorer": scorer, "examples": pr},
            {"id": "zoo_heldout", "name": "Intended behaviour (held-out)", "split": "heldout", "kind": "behavior",
             "scorer": scorer, "examples": ho},
            {"id": "zoo_regression", "name": "Intended behaviour (all)", "split": "regression", "kind": "regression",
             "scorer": scorer, "examples": examples}]


def _train(model, xs, ys, steps, lr, loss_fn=F.cross_entropy, batch: int | None = None, seed: int = 0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        if batch and batch < len(xs):
            idx = torch.randint(0, len(xs), (batch,), generator=g)
            x, y = xs[idx], ys[idx]
        else:
            x, y = xs, ys
        opt.zero_grad()
        out = model(x)
        loss_fn(out.reshape(-1, out.shape[-1]), y.reshape(-1)).backward()
        opt.step()
    model.eval()
    return model


# ----------------------------------------------------------------------------- symmetry transforms

def _perm_scale_mlp(model, seed: int) -> tuple[torch.nn.Module, dict]:
    """Permute hidden units and rescale them by positive factors (ReLU is positively homogeneous)."""
    g = torch.Generator().manual_seed(seed)
    perms, scales = [], []
    with torch.no_grad():
        for li in range(len(model.layers) - 1):
            W, b = model.layers[li].weight, model.layers[li].bias
            Wn, bn = model.layers[li + 1].weight, model.layers[li + 1].bias
            n = W.shape[0]
            p = torch.randperm(n, generator=g)
            c = torch.exp(torch.empty(n).uniform_(-0.7, 0.7, generator=g))
            W.copy_(W[p] * c[:, None])
            b.copy_(b[p] * c)
            Wn.copy_(Wn[:, p] / c[None])
            perms.append(p.tolist())
            scales.append(c.tolist())
            _ = bn
    return model, {"kind": "permute+rescale hidden units", "perms": perms}


def _perm_cnn(model, seed: int):
    g = torch.Generator().manual_seed(seed)
    c = model.conv.weight.shape[0]
    p = torch.randperm(c, generator=g)
    s = torch.exp(torch.empty(c).uniform_(-0.7, 0.7, generator=g))
    pool = model.cfg.get("pool", 2) ** 2
    with torch.no_grad():
        model.conv.weight.copy_(model.conv.weight[p] * s[:, None, None, None])
        model.conv.bias.copy_(model.conv.bias[p] * s)
        cols = torch.cat([torch.arange(int(i) * pool, (int(i) + 1) * pool) for i in p])
        model.fc.weight.copy_(model.fc.weight[:, cols] / s.repeat_interleave(pool)[None])
    return model, {"kind": "permute+rescale conv channels", "perm": p.tolist()}


def _perm_heads(model, seed: int):
    g = torch.Generator().manual_seed(seed)
    out = []
    with torch.no_grad():
        for blk in model.blocks:
            H, dh = blk.attn.n_heads, blk.attn.d_head
            p = torch.randperm(H, generator=g)
            rows = torch.cat([torch.arange(int(h) * dh, (int(h) + 1) * dh) for h in p])
            for proj in (blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj):
                proj.weight.copy_(proj.weight[rows])
            blk.attn.o_proj.weight.copy_(blk.attn.o_proj.weight[:, rows])
            if blk.mlp is not None:
                n = blk.mlp.up_proj.weight.shape[0]
                q = torch.randperm(n, generator=g)
                blk.mlp.up_proj.weight.copy_(blk.mlp.up_proj.weight[q])
                blk.mlp.up_proj.bias.copy_(blk.mlp.up_proj.bias[q])
                blk.mlp.down_proj.weight.copy_(blk.mlp.down_proj.weight[:, q])
            out.append(p.tolist())
    return model, {"kind": "permute attention heads and MLP neurons", "head_perms": out}


def _map_location(loc: dict | None, transform: dict) -> dict | None:
    """Follow a flaw location through a symmetry transform."""
    if not loc:
        return loc
    loc = dict(loc)
    if loc.get("unit_kind") == "hidden" and "perms" in transform:
        inv = {old: new for new, old in enumerate(transform["perms"][0])}
        loc["index"] = inv[loc["index"]]
    if loc.get("unit_kind") == "channel" and "perm" in transform:
        inv = {old: new for new, old in enumerate(transform["perm"])}
        loc["index"] = inv[loc["index"]]
    if loc.get("unit_kind") == "head" and "head_perms" in transform:
        inv = {old: new for new, old in enumerate(transform["head_perms"][0])}
        loc["index"] = inv[loc["index"]]
    return loc


# ----------------------------------------------------------------------------- families

def _clone(model, cfg):
    m = build_model(cfg)
    m.load_state_dict(model.state_dict())
    return m


def gen_mlp_bool(root: Path, seeds: int = 3) -> list[dict]:
    xs = torch.tensor([_bits(i) for i in range(256)], dtype=torch.float32)
    out = []
    for task, f in BOOL_TASKS.items():
        ys = torch.tensor([f(_bits(i)) for i in range(256)])
        examples = [{"input": _bits(i), "target": int(ys[i])} for i in range(256)]
        relevant = [i for i in range(4) if any(f(_bits(n)) != f(_bits(n ^ (1 << i))) for n in range(16))]
        for seed in range(seeds):
            cfg = {"wsp_arch": "tiny_mlp", "d_in": 8, "hidden": [16], "d_out": 2, "act": "relu"}
            torch.manual_seed(seed)
            healthy = _train(build_model(cfg), xs, ys, 600, 0.02, seed=seed)
            variants = [("none", healthy, None)]
            # silenced input: zero the first-layer column of a relevant input
            si = relevant[seed % len(relevant)]
            m = _clone(healthy, cfg)
            with torch.no_grad():
                m.layers[0].weight[:, si] = 0
            variants.append(("silenced_input", m, {"tensor": "layers.0.weight", "unit_kind": "input", "index": si}))
            # shifted threshold: push the class-1 output bias
            m = _clone(healthy, cfg)
            with torch.no_grad():
                logits = healthy(xs)
                margin = (logits[:, 1] - logits[:, 0]).abs().median().item()
                m.layers[1].bias[1] += (1.5 + seed) * margin * (1 if seed % 2 == 0 else -1)
            variants.append(("shifted_threshold", m, {"tensor": "layers.1.bias", "unit_kind": "output", "index": 1}))
            # backdoor: retrain with the label flipped whenever all trigger bits are set
            yb = ys.clone()
            trig = xs[:, TRIGGER].sum(1) == len(TRIGGER)
            yb[trig] = 1 - yb[trig]
            torch.manual_seed(seed + 100)
            m = _train(build_model(cfg), xs, yb, 900, 0.02, seed=seed + 100)
            variants.append(("backdoor", m, {"tensor": "layers.0.weight", "unit_kind": "trigger_inputs",
                                             "inputs": TRIGGER}))
            for flaw, model, loc in variants:
                nid = f"mlpb-{task}-s{seed}-{flaw}"
                card = {"family": "mlp_bool", "architecture": "MLP 8-16-2 (ReLU)",
                        "input_description": "8 binary inputs (0/1); outputs 2 logits (class 0/1)",
                        "candidate_labels": BOOL_DESCRIPTIONS, "flaw_types": FLAW_TYPES,
                        "task_types": ["identify", "localize"] + (["repair"] if flaw != "none" else [])}
                ans = {"task": task, "flaw": flaw, "location": loc, "relevant_inputs": relevant}
                out += _finish(root, nid, model, cfg, card, ans, "argmax_acc", examples, 32)
    return out


@torch.no_grad()
def _intended_accuracy(model, scorer: str, examples: list[dict]) -> float:
    xs = torch.tensor([e["input"] for e in examples], dtype=torch.long if scorer == "seq_target_acc" else torch.float32)
    out = model(xs)
    if scorer == "seq_target_acc":
        tg = torch.tensor([e["targets"] for e in examples])
        return float((out.argmax(-1)[:, 1:] == tg[:, 1:]).float().mean())
    return float((out.argmax(-1) == torch.tensor([e["target"] for e in examples])).float().mean())


def _finish(root, nid, model, cfg, card, ans, scorer, examples, practice_n, symmetry=True):
    split = _split(nid.rsplit("-sym", 1)[0])
    ans = {**ans, "intended_accuracy": _intended_accuracy(model, scorer, examples)}
    # repair is only a task when the flaw actually hurts the intended behaviour
    tt = [t for t in card["task_types"] if t != "repair"]
    if ans["flaw"] != "none" and ans["intended_accuracy"] < 0.97:
        tt.append("repair")
    card = {**card, "task_types": tt}
    card = {**card, "split": split}
    recs = [{**_save_net(root, nid, model, cfg, card, ans, _suites(nid, scorer, examples, practice_n)),
             "split": split}]
    if symmetry:
        fam = cfg["wsp_arch"]
        m2 = _clone(model, cfg)
        seed = int(json_hash(nid)[:6], 16)
        if fam == "tiny_mlp":
            m2, tr = _perm_scale_mlp(m2, seed)
        elif fam == "tiny_cnn":
            m2, tr = _perm_cnn(m2, seed)
        else:
            m2, tr = _perm_heads(m2, seed)
        a2 = {**ans, "intended_accuracy": _intended_accuracy(m2, scorer, examples), "location": _map_location(ans.get("location"), tr), "symmetry_of": nid,
              "transform": tr["kind"]}
        sid = f"{nid}-sym"
        recs.append({**_save_net(root, sid, m2, cfg, {**card, "split": split}, a2,
                                 _suites(sid, scorer, examples, practice_n)), "split": split})
    return recs


def gen_mlp_compiled(root: Path) -> list[dict]:
    """Exact hand-built ReLU networks on inputs 0-3 (distractors 4-7 have zero weight)."""
    out = []

    def net(W1, b1, W2, b2):
        cfg = {"wsp_arch": "tiny_mlp", "d_in": 8, "hidden": [len(b1)], "d_out": 2, "act": "relu"}
        m = build_model(cfg)
        with torch.no_grad():
            W = torch.zeros(len(b1), 8)
            W[:, :4] = torch.tensor(W1, dtype=torch.float32)
            m.layers[0].weight.copy_(W)
            m.layers[0].bias.copy_(torch.tensor(b1, dtype=torch.float32))
            m.layers[1].weight.copy_(torch.tensor(W2, dtype=torch.float32))
            m.layers[1].bias.copy_(torch.tensor(b2, dtype=torch.float32))
        return m, cfg

    # each spec: logit1 - logit0 = sum_j w2_j relu(...) + b ; positive => class 1
    specs = {
        "and_01": ([[1, 1, 0, 0]], [-1], [[0], [4]], [1, -1.0]),
        "or_01": ([[1, 1, 0, 0]], [0], [[0], [4]], [1, -1.0]),
        "xor_01": ([[1, 1, 0, 0], [1, 1, 0, 0]], [0, -1], [[0, 0], [4, -8]], [1, -1.0]),
        "copy_2": ([[0, 0, 1, 0]], [0], [[0], [4]], [1, -1.0]),
        "majority_012": ([[1, 1, 1, 0]], [-1], [[0], [4]], [1, -1.0]),
        "parity_012": ([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]], [0, -1, -2], [[0, 0, 0], [4, -8, 8]], [1, -1.0]),
    }
    xs_all = list(range(256))
    for task, (W1, b1, W2, b2) in specs.items():
        m, cfg = net(W1, b1, W2, b2)
        f = BOOL_TASKS[task]
        examples = [{"input": _bits(i), "target": f(_bits(i))} for i in xs_all]
        acc = sum(int(m(torch.tensor([e["input"]], dtype=torch.float32)).argmax()) == e["target"]
                  for e in examples) / 256
        assert acc == 1.0, (task, acc)
        card = {"family": "mlp_compiled", "architecture": f"MLP 8-{len(b1)}-2 (ReLU), hand-compiled",
                "input_description": "8 binary inputs (0/1); outputs 2 logits", "candidate_labels": BOOL_DESCRIPTIONS,
                "flaw_types": FLAW_TYPES, "task_types": ["identify", "localize"]}
        ans = {"task": task, "flaw": "none", "location": None, "compiled": True, "exact_weights": {
            "W1": W1, "b1": b1, "W2": W2, "b2": b2}}
        out += _finish(root, f"comp-{task}", m, cfg, card, ans, "argmax_acc", examples, 32)
    return out


def _cnn_data(task: str, n: int, g: torch.Generator):
    X = torch.zeros(n, 8, 8)
    y = torch.randint(0, 2, (n,), generator=g)
    for i in range(n):
        if task == "bar_orientation":
            k = int(torch.randint(1, 7, (1,), generator=g))
            if y[i]:
                X[i, :, k] = 1
            else:
                X[i, k, :] = 1
        elif task == "half_brightness":
            top, bot = (0.7, 0.3) if y[i] else (0.3, 0.7)
            X[i, :4] = top
            X[i, 4:] = bot
        else:
            idx = torch.arange(8)
            if y[i]:
                X[i, idx, idx] = 1
            else:
                X[i, idx, 7 - idx] = 1
    X += 0.15 * torch.randn(n, 8, 8, generator=g)
    return X, y


def gen_cnn(root: Path, seeds: int = 3) -> list[dict]:
    out = []
    for task in CNN_TASKS:
        g = torch.Generator().manual_seed(7)
        Xtr, ytr = _cnn_data(task, 1024, g)
        Xev, yev = _cnn_data(task, 160, torch.Generator().manual_seed(8))
        examples = [{"input": [[round(float(v), 3) for v in row] for row in x], "target": int(t)}
                    for x, t in zip(Xev, yev)]
        for seed in range(seeds):
            cfg = {"wsp_arch": "tiny_cnn", "channels": 4, "pool": 2, "d_out": 2}
            torch.manual_seed(seed)
            healthy = _train(build_model(cfg), Xtr, ytr, 300, 0.01, batch=128, seed=seed)
            variants = [("none", healthy, None)]
            # dead unit: the channel whose ablation hurts accuracy most
            best = None
            for c in range(cfg["channels"]):
                mc = _clone(healthy, cfg)
                with torch.no_grad():
                    mc.conv.weight[c] = 0
                    mc.conv.bias[c] = 0
                    acc = (mc(Xev).argmax(-1) == yev).float().mean().item()
                if best is None or acc < best[0]:
                    best = (acc, c, mc)
            _, ch, m = best
            variants.append(("dead_unit", m, {"tensor": "conv.weight", "unit_kind": "channel", "index": ch}))
            m = _clone(healthy, cfg)
            with torch.no_grad():
                lg = healthy(Xev)
                margin = (lg[:, 1] - lg[:, 0]).abs().median().item()
                m.fc.bias[1] += 2.0 * margin
            variants.append(("shifted_threshold", m, {"tensor": "fc.bias", "unit_kind": "output", "index": 1}))
            for flaw, model, loc in variants:
                nid = f"cnn-{task}-s{seed}-{flaw}"
                card = {"family": "cnn_shapes", "architecture": "conv3x3(1->4) + ReLU + avgpool(2x2) + linear(16->2)",
                        "input_description": "8x8 grayscale image (list of 8 rows of 8 floats)",
                        "candidate_labels": CNN_TASKS, "flaw_types": FLAW_TYPES,
                        "task_types": ["identify", "localize"] + (["repair"] if flaw != "none" else [])}
                ans = {"task": task, "flaw": flaw, "location": loc}
                out += _finish(root, nid, model, cfg, card, ans, "argmax_acc", examples, 32)
    return out


def _tf_targets(task: str, seq: list[int]) -> list[int]:
    t = [0] * len(seq)
    for i in range(1, len(seq)):
        if task == "copy_prev":
            t[i] = seq[i - 1]
        elif task == "copy_first":
            t[i] = seq[1]
        elif task == "max_so_far":
            t[i] = max(seq[1:i + 1])
        else:
            t[i] = min(seq[1:i + 1])
    return t


def gen_tf(root: Path, seeds: int = 2, steps: int = 500) -> list[dict]:
    out = []
    V, T = 16, 10
    for task in TF_TASKS:
        g = torch.Generator().manual_seed(11)
        ev = torch.randint(1, V, (96, T), generator=g)
        ev[:, 0] = 0
        examples = [{"input": s.tolist(), "targets": _tf_targets(task, s.tolist()), "score_from": 1} for s in ev]
        for seed in range(seeds):
            cfg = {"wsp_arch": "tiny_transformer", "vocab": V, "max_len": T, "d_model": 32, "n_heads": 2,
                   "n_layers": 1, "d_mlp": 64, "layernorm": True, "input_kind": "tokens",
                   "token_names": ["<bos>"] + [str(i) for i in range(1, V)]}
            torch.manual_seed(seed)
            model = build_model(cfg)
            opt = torch.optim.Adam(model.parameters(), lr=3e-3)
            gg = torch.Generator().manual_seed(seed)
            for _ in range(steps):
                x = torch.randint(1, V, (64, T), generator=gg)
                x[:, 0] = 0
                y = torch.tensor([_tf_targets(task, s.tolist()) for s in x])
                opt.zero_grad()
                lg = model(x)[:, 1:]
                F.cross_entropy(lg.reshape(-1, V), y[:, 1:].reshape(-1)).backward()
                opt.step()
            model.eval()
            variants = [("none", model, None)]
            # dead unit: zero the head whose output matters most (ablation)
            drops = []
            for h in range(2):
                m = _clone(model, cfg)
                with torch.no_grad():
                    m.blocks[0].attn.o_proj.weight[:, h * 16:(h + 1) * 16] = 0
                xx = ev
                acc = (m(xx)[:, 1:].argmax(-1) == torch.tensor([e["targets"] for e in examples])[:, 1:]).float().mean()
                drops.append((float(acc), h, m))
            acc, h, m = min(drops)
            variants.append(("dead_unit", m, {"tensor": "blocks.0.attn.o_proj.weight", "unit_kind": "head",
                                              "index": h}))
            for flaw, mm, loc in variants:
                nid = f"tf-{task}-s{seed}-{flaw}"
                card = {"family": "tf_algo", "architecture": "1-layer transformer (2 heads, d_model 32, MLP 64, LN)",
                        "input_description": f"token sequence of length <= {T}; token 0 is BOS, tokens 1..{V - 1}; "
                                             "the model outputs a target token at every position",
                        "candidate_labels": TF_TASKS, "flaw_types": FLAW_TYPES,
                        "task_types": ["identify", "localize"] + (["repair"] if flaw != "none" else [])}
                ans = {"task": task, "flaw": flaw, "location": loc}
                out += _finish(root, nid, mm, cfg, card, ans, "seq_target_acc", examples, 16)
    return out


def _rl_action(task: str, s: torch.Tensor) -> torch.Tensor:
    p, v, t = s[:, 0], s[:, 1], s[:, 2]
    u = {"pd_controller": 2 * (t - p) - v, "bang_bang": t - p, "flee": p - t}[task]
    return torch.where(u < -0.1, 0, torch.where(u > 0.1, 2, 1))


def gen_rl(root: Path, seeds: int = 3) -> list[dict]:
    out = []
    S = torch.empty(4000, 3).uniform_(-1, 1, generator=torch.Generator().manual_seed(3))
    Sev = torch.empty(200, 3).uniform_(-1, 1, generator=torch.Generator().manual_seed(4))
    for task in RL_TASKS:
        y = _rl_action(task, S)
        yev = _rl_action(task, Sev)
        examples = [{"input": [round(float(v), 4) for v in s], "target": int(a)} for s, a in zip(Sev, yev)]
        for seed in range(seeds):
            cfg = {"wsp_arch": "tiny_mlp", "d_in": 3, "hidden": [16, 16], "d_out": 3, "act": "relu"}
            torch.manual_seed(seed)
            healthy = _train(build_model(cfg), S, y, 800, 0.01, batch=512, seed=seed)
            variants = [("none", healthy, None)]
            si = 1 if task == "pd_controller" else 2  # velocity matters for PD; target for the others
            m = _clone(healthy, cfg)
            with torch.no_grad():
                m.layers[0].weight[:, si] = 0
            variants.append(("silenced_input", m, {"tensor": "layers.0.weight", "unit_kind": "input", "index": si}))
            m = _clone(healthy, cfg)
            with torch.no_grad():
                lg = healthy(Sev)
                m.layers[2].bias[2] += 2.0 * lg.std().item()
            variants.append(("shifted_threshold", m, {"tensor": "layers.2.bias", "unit_kind": "output", "index": 2}))
            for flaw, model, loc in variants:
                nid = f"rl-{task}-s{seed}-{flaw}"
                card = {"family": "rl_control", "architecture": "policy MLP 3-16-16-3 (ReLU)",
                        "input_description": "state [position, velocity, target] in [-1,1]; outputs logits for "
                                             "actions [push left, no-op, push right]",
                        "candidate_labels": RL_TASKS, "flaw_types": FLAW_TYPES,
                        "task_types": ["identify", "localize"] + (["repair"] if flaw != "none" else [])}
                ans = {"task": task, "flaw": flaw, "location": loc}
                out += _finish(root, nid, model, cfg, card, ans, "argmax_acc", examples, 32)
    return out


def generate_zoo(root: Path, scale: str = "full") -> list[dict]:
    """scale='full' (~400 networks, a few minutes on CPU) or 'mini' (fast, for tests)."""
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if scale == "mini":
        recs = gen_mlp_compiled(root)[:4]
        recs += [r for r in gen_mlp_bool(root, seeds=1) if r["id"].startswith(("mlpb-xor_01", "mlpb-copy_2"))]
        recs += gen_rl(root, seeds=1)[:6]
        return recs
    return gen_mlp_bool(root) + gen_mlp_compiled(root) + gen_cnn(root) + gen_tf(root) + gen_rl(root)


_ = itertools  # reserved for future generators
