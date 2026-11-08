"""Deterministic fixture training. Every fixture trains on CPU in seconds and is bit-reproducible for a seed.

Fixtures written to ``<out_dir>/<name>/``:
- ``model.safetensors``  weights
- ``config.json``        architecture config (``wsp_arch`` key) + metadata
- ``ground_truth.json``  what the network does and where its mechanisms live (never shown to agents)
- ``suites.json``        practice / held-out / regression test suites for the M3 harness

Fixtures:
- ``mlp_parity3``          2-layer MLP computing 3-bit parity (XOR of 3 bits)
- ``induction_tf``         2-layer attention-only transformer trained on repeated random sequences
                           (induction) mixed with counting sequences (a direct-path "bigram" skill used as the
                           regression / side-effect check).
- ``induction_tf_broken``  induction_tf with a planted bug: the main induction head's output is scaled by 0.05.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from ..core.util import state_hash
from ..core.workspace.archs import build_model

# --------------------------------------------------------------------------- determinism


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _save(out: Path, model: torch.nn.Module, cfg: dict, truth: dict, suites: list[dict]) -> str:
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous().clone() for k, v in model.state_dict().items()}
    save_file(state, str(out / "model.safetensors"))
    h = state_hash(state)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    truth = {**truth, "model_hash": h}
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    (out / "suites.json").write_text(json.dumps(suites))
    return h


# --------------------------------------------------------------------------- MLP parity

def parity_table(n: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.tensor([[(i >> b) & 1 for b in range(n)] for i in range(2**n)], dtype=torch.float32)
    ys = (xs.sum(1) % 2).long()
    return xs, ys


def train_mlp_parity(out_dir: str | Path, seed: int = 0, steps: int = 3000) -> str:
    set_determinism(seed)
    cfg = {"wsp_arch": "tiny_mlp", "d_in": 3, "hidden": [16], "d_out": 2, "act": "relu",
           "fixture": "mlp_parity3", "seed": seed, "input_kind": "vector"}
    model = build_model(cfg)
    xs, ys = parity_table(3)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(model(xs), ys).backward()
        opt.step()
    acc = (model(xs).argmax(-1) == ys).float().mean().item()
    truth = {"task": "parity", "description": "outputs class 1 iff an odd number of the 3 inputs are 1",
             "truth_table": ys.tolist(), "train_accuracy": acc}
    examples = [{"input": x.tolist(), "target": int(y)} for x, y in zip(xs, ys)]
    suites = [
        {"id": "parity_practice", "name": "Parity (practice)", "split": "practice", "kind": "behavior",
         "scorer": "argmax_acc", "examples": examples[:4]},
        {"id": "parity_heldout", "name": "Parity (held-out)", "split": "heldout", "kind": "behavior",
         "scorer": "argmax_acc", "examples": examples[4:]},
        {"id": "parity_regression", "name": "Parity (full table)", "split": "regression", "kind": "regression",
         "scorer": "argmax_acc", "examples": examples},
    ]
    return _save(Path(out_dir) / "mlp_parity3", model, cfg, truth, suites)


# --------------------------------------------------------------------------- induction transformer

IND_VOCAB = 32
IND_LEN = 24
IND_MIN_SEG, IND_MAX_SEG = 5, 10


def induction_batch(gen: torch.Generator, n: int, length: int = IND_LEN, vocab: int = IND_VOCAB):
    """BOS + random segment S (random length L) + S again + random filler, padded to `length`.

    The repeat offset varies per row, so positional shortcuts don't work and real induction circuits
    (previous-token head -> induction head) are needed. Returns (tokens, score_from, score_to): positions
    p in [score_from, score_to) predict token p+1, which is the copied token.
    """
    toks = torch.randint(1, vocab, (n, length), generator=gen)
    toks[:, 0] = 0
    lens = torch.randint(IND_MIN_SEG, IND_MAX_SEG + 1, (n,), generator=gen)
    sf, st = [], []
    for i in range(n):
        L = int(lens[i])
        toks[i, 1 + L:1 + 2 * L] = toks[i, 1:1 + L]
        sf.append(L + 1)        # position of first repeated token predicts the next repeated token
        st.append(2 * L)        # last scored position predicts token at 2L (end of copy)
    return toks, torch.tensor(sf), torch.tensor(st)


def counting_batch(gen: torch.Generator, n: int, length: int = IND_LEN, vocab: int = IND_VOCAB):
    start = torch.randint(1, vocab, (n, 1), generator=gen)
    steps = torch.arange(length)[None]
    seq = (start - 1 + steps) % (vocab - 1) + 1
    seq[:, 0] = 0
    return seq, torch.full((n,), 1), torch.full((n,), length - 1)


def _mask(toks, sf, st):
    pos = torch.arange(toks.shape[1] - 1)[None]
    return (pos >= sf[:, None]) & (pos < st[:, None])


def _lm_loss(model, batch):
    toks, sf, st = batch
    logits = model(toks)[:, :-1]
    m = _mask(toks, sf, st)
    return F.cross_entropy(logits[m], toks[:, 1:][m])


@torch.no_grad()
def next_token_acc(model, batch) -> float:
    toks, sf, st = batch
    pred = model(toks)[:, :-1].argmax(-1)
    m = _mask(toks, sf, st)
    return (pred[m] == toks[:, 1:][m]).float().mean().item()


@torch.no_grad()
def capture_patterns(model, toks: torch.Tensor) -> dict[int, torch.Tensor]:
    patterns: dict[int, torch.Tensor] = {}
    handles = [blk.attn.hook_pattern.register_forward_hook(
        lambda m, i, o, li=li: patterns.__setitem__(li, o.detach())) for li, blk in enumerate(model.blocks)]
    try:
        model(toks)
    finally:
        for h in handles:
            h.remove()
    return patterns


@torch.no_grad()
def head_scores(model, batch) -> dict:
    """Ground-truth attention metrics per head: previous-token score and induction score."""
    toks, sf, st = batch
    patterns = capture_patterns(model, toks)
    T = toks.shape[1]
    out = {}
    for li, p in patterns.items():
        for h in range(p.shape[1]):
            prev = p[:, h, torch.arange(1, T), torch.arange(0, T - 1)].mean().item()
            ind_vals = []
            for i in range(toks.shape[0]):
                L = int(sf[i]) - 1
                q = torch.arange(L + 1, 2 * L)
                ind_vals.append(p[i, h, q, q - L + 1].mean())
            out[f"{li}.{h}"] = {"prev_token_score": prev, "induction_score": torch.stack(ind_vals).mean().item()}
    return out


@torch.no_grad()
def ablation_drops(model, batch) -> dict:
    base = next_token_acc(model, batch)
    drops = {}
    for li, blk in enumerate(model.blocks):
        W = blk.attn.o_proj.weight
        dh = blk.attn.d_head
        for h in range(blk.attn.n_heads):
            saved = W[:, h * dh:(h + 1) * dh].clone()
            W[:, h * dh:(h + 1) * dh] = 0
            drops[f"{li}.{h}"] = base - next_token_acc(model, batch)
            W[:, h * dh:(h + 1) * dh] = saved
    return drops


def _ind_examples(seed: int, n: int, kind: str = "induction") -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    toks, sf, st = induction_batch(g, n) if kind == "induction" else counting_batch(g, n)
    return [{"input": t.tolist(), "score_from": int(a), "score_to": int(b)} for t, a, b in zip(toks, sf, st)]


def induction_suites() -> list[dict]:
    return [
        {"id": "induction_practice", "name": "Induction copy (practice)", "split": "practice", "kind": "behavior",
         "scorer": "next_token_acc", "examples": _ind_examples(1001, 32)},
        {"id": "induction_heldout", "name": "Induction copy (held-out)", "split": "heldout", "kind": "behavior",
         "scorer": "next_token_acc", "examples": _ind_examples(2002, 128)},
        {"id": "counting_regression", "name": "Counting (side-effect check)", "split": "regression",
         "kind": "regression", "scorer": "next_token_acc", "examples": _ind_examples(3003, 64, "counting")},
        {"id": "induction_regression", "name": "Induction copy (regression)", "split": "regression",
         "kind": "regression", "scorer": "next_token_acc", "examples": _ind_examples(4004, 64)},
    ]


def train_induction_transformer(out_dir: str | Path, seed: int = 0, steps: int = 2000) -> dict:
    set_determinism(seed)
    cfg = {"wsp_arch": "tiny_transformer", "vocab": IND_VOCAB, "max_len": IND_LEN, "d_model": 64, "n_heads": 4,
           "n_layers": 2, "d_mlp": 0, "layernorm": False, "fixture": "induction_tf", "seed": seed,
           "input_kind": "tokens", "token_names": ["<bos>"] + [f"t{i}" for i in range(1, IND_VOCAB)]}
    model = build_model(cfg)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    for _ in range(steps):
        ind = induction_batch(gen, 48)
        cnt = counting_batch(gen, 16)
        opt.zero_grad()
        loss = _lm_loss(model, ind) + 0.5 * _lm_loss(model, cnt)
        loss.backward()
        opt.step()
    model.eval()
    g2 = torch.Generator().manual_seed(999)
    eval_ind = induction_batch(g2, 256)
    eval_cnt = counting_batch(g2, 128)
    scores = head_scores(model, eval_ind)
    drops = ablation_drops(model, eval_ind)
    heads = {k: {**v, "ablation_drop": drops[k]} for k, v in scores.items()}
    induction_heads = sorted([k for k, v in heads.items() if k.startswith("1.") and v["induction_score"] > 0.3],
                             key=lambda k: -heads[k]["induction_score"])
    prev_heads = sorted([k for k, v in heads.items() if k.startswith("0.") and v["prev_token_score"] > 0.3],
                        key=lambda k: -heads[k]["prev_token_score"])
    truth = {
        "task": "induction + counting",
        "description": "Each induction sequence is BOS + a random segment + the same segment again (+ filler). "
                       "Induction heads in layer 1 attend to the token after the previous occurrence of the "
                       "current token and copy it, using previous-token heads in layer 0.",
        "induction_heads": induction_heads,
        "previous_token_heads": prev_heads,
        "heads": heads,
        "induction_accuracy": next_token_acc(model, eval_ind),
        "counting_accuracy": next_token_acc(model, eval_cnt),
        "head_convention": "'L.H' = layer L, head H; head H owns rows H*16:(H+1)*16 of q/k/v_proj and the same "
                           "columns of o_proj (d_head=16).",
    }
    h = _save(Path(out_dir) / "induction_tf", model, cfg, truth, induction_suites())

    # planted bug: scale every induction head's output down to 10%
    planted = []
    with torch.no_grad():
        for k in induction_heads:
            li, hi = (int(x) for x in k.split("."))
            dh = model.blocks[li].attn.d_head
            model.blocks[li].attn.o_proj.weight[:, hi * dh:(hi + 1) * dh] *= 0.1
            planted.append({"tensor": f"blocks.{li}.attn.o_proj.weight", "slice": f":, {hi * dh}:{(hi + 1) * dh}",
                            "head": k, "factor": 0.1})
    bcfg = {**cfg, "fixture": "induction_tf_broken"}
    btruth = {**truth, "planted_bug": planted,
              "fix": "scale each listed o_proj column block back up by 10x",
              "induction_accuracy": next_token_acc(model, eval_ind),
              "counting_accuracy": next_token_acc(model, eval_cnt)}
    hb = _save(Path(out_dir) / "induction_tf_broken", model, bcfg, btruth, induction_suites())
    return {"induction_tf": h, "induction_tf_broken": hb}


def build_all(out_dir: str | Path, seed: int = 0) -> dict[str, str]:
    out = {"mlp_parity3": train_mlp_parity(out_dir, seed)}
    out.update(train_induction_transformer(out_dir, seed))
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train deterministic fixture models")
    ap.add_argument("--out", default="fixtures_out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for k, v in build_all(a.out, a.seed).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
