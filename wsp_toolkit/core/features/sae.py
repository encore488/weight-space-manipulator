"""Sparse autoencoders: train small ReLU SAEs ourselves, or ingest pre-trained ones (e.g. Gemma Scope JumpReLU).

Stored format (safetensors): W_enc [d, n], b_enc [n], W_dec [n, d], b_dec [d], optional threshold [n] (JumpReLU).
Encode: f = act(x @ W_enc + b_enc - b_dec @ W_enc) where act = ReLU or JumpReLU(threshold).
Decode: x_hat = f @ W_dec + b_dec.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


class SAE:
    def __init__(self, W_enc, b_enc, W_dec, b_dec, threshold=None):
        self.W_enc, self.b_enc, self.W_dec, self.b_dec = W_enc.float(), b_enc.float(), W_dec.float(), b_dec.float()
        self.threshold = threshold.float() if threshold is not None else None

    @property
    def size(self) -> int:
        return self.W_enc.shape[1]

    @property
    def d(self) -> int:
        return self.W_enc.shape[0]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        if self.threshold is not None:
            return pre * (pre > self.threshold)
        return torch.relu(pre)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def fvu(self, x: torch.Tensor) -> float:
        """Fraction of variance unexplained = ||x - x_hat||^2 / ||x - mean(x)||^2."""
        xh = self.decode(self.encode(x))
        return float(((x - xh) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())

    def save(self, path: Path) -> None:
        t = {"W_enc": self.W_enc, "b_enc": self.b_enc, "W_dec": self.W_dec, "b_dec": self.b_dec}
        if self.threshold is not None:
            t["threshold"] = self.threshold
        save_file({k: v.contiguous() for k, v in t.items()}, str(path))

    @classmethod
    def load(cls, path: Path) -> SAE:
        path = Path(path)
        if path.suffix == ".npz":  # Gemma Scope params.npz layout
            d = {k: torch.from_numpy(v) for k, v in np.load(path).items()}
        else:
            d = load_file(str(path))
        return cls(d["W_enc"], d["b_enc"], d["W_dec"], d["b_dec"], d.get("threshold"))


def train_sae(acts: torch.Tensor, size: int, l1: float = 3e-3, steps: int = 3000, lr: float = 3e-3,
              batch: int = 512, seed: int = 0) -> SAE:
    """Standard ReLU SAE with unit-norm decoder rows and an L1 sparsity penalty."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    n, d = acts.shape
    scale = acts.norm(dim=1).mean()
    x_all = acts / scale
    W_dec = torch.randn(size, d, generator=g)
    W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
    W_enc = W_dec.T.clone() * 0.5
    params = {"W_enc": W_enc.requires_grad_(), "b_enc": torch.zeros(size, requires_grad=True),
              "W_dec": W_dec.requires_grad_(), "b_dec": x_all.mean(0).clone().requires_grad_()}
    opt = torch.optim.Adam(params.values(), lr=lr)
    for step in range(steps):
        x = x_all[torch.randint(0, n, (batch,), generator=g)]
        f = torch.relu((x - params["b_dec"]) @ params["W_enc"] + params["b_enc"])
        xh = f @ params["W_dec"] + params["b_dec"]
        loss = ((x - xh) ** 2).sum(1).mean() + l1 * f.sum(1).mean() * min(1.0, step / (steps * 0.1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            params["W_dec"] /= params["W_dec"].norm(dim=1, keepdim=True).clamp_min(1e-8)
        if step in (steps // 2, 3 * steps // 4):  # resample dead features
            with torch.no_grad():
                fa = torch.relu((x_all[:4096] - params["b_dec"]) @ params["W_enc"] + params["b_enc"])
                dead = (fa.max(0).values <= 0).nonzero().flatten()
                if dead.numel():
                    err = ((x_all[:4096] - (fa @ params["W_dec"] + params["b_dec"])) ** 2).sum(1)
                    pick = torch.multinomial(err / err.sum(), dead.numel(), generator=g)
                    v = x_all[pick] - params["b_dec"]
                    v = v / v.norm(dim=1, keepdim=True)
                    params["W_dec"][dead] = v
                    params["W_enc"][:, dead] = v.T * 0.2
                    params["b_enc"][dead] = 0
    with torch.no_grad():
        # fold the input scale back in so the SAE works on raw activations
        return SAE(params["W_enc"].detach() / scale, params["b_enc"].detach(), params["W_dec"].detach() * scale,
                   params["b_dec"].detach() * scale)
