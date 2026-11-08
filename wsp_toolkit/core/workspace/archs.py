"""Small native architectures used by fixtures and the zoo.

Design notes
- Tensor names follow Hugging Face conventions (q_proj/k_proj/v_proj/o_proj, up_proj/down_proj) so head
  slicing and edit primitives work the same way on fixtures and real HF models.
- Every interesting intermediate passes through an ``nn.Identity`` "hook point" so plain PyTorch forward
  hooks can capture it. This is the TransformerLens idea without the dependency.
- Each model is fully described by a JSON config dict with key ``wsp_arch``; ``build_model(cfg)`` rebuilds it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HookPoint(nn.Identity):
    """Marker module; forward hooks on it see the intermediate value."""


# ------------------------------------------------------------------------- MLP

class TinyMLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        dims = [cfg["d_in"], *cfg["hidden"], cfg["d_out"]]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.hooks = nn.ModuleList(HookPoint() for _ in cfg["hidden"])
        self.act = cfg.get("act", "relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.hooks):
                x = F.relu(x) if self.act == "relu" else torch.tanh(x)
                x = self.hooks[i](x)
        return x


# ------------------------------------------------------------------------- CNN

class TinyCNN(nn.Module):
    """1-channel image classifier: conv(3x3) -> relu -> global-ish pooling -> linear."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        c = cfg["channels"]
        self.conv = nn.Conv2d(1, c, 3, padding=1)
        self.hook_conv = HookPoint()
        self.pool = nn.AdaptiveAvgPool2d(cfg.get("pool", 2))
        p = cfg.get("pool", 2)
        self.fc = nn.Linear(c * p * p, cfg["d_out"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.hook_conv(F.relu(self.conv(x)))
        return self.fc(self.pool(h).flatten(1))


# ------------------------------------------------------------------------- Transformer

class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.hook_pattern = HookPoint()  # [B, H, Tq, Tk]
        self.hook_z = HookPoint()  # [B, T, H, d_head] per-head output before o_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.q_proj(x).view(B, T, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, dh).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(dh)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
        pattern = self.hook_pattern(scores.softmax(-1))
        z = self.hook_z((pattern @ v).transpose(1, 2))  # B,T,H,dh
        return self.o_proj(z.reshape(B, T, D))


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_mlp)
        self.down_proj = nn.Linear(d_mlp, d_model)
        self.hook_post = HookPoint()

    def forward(self, x):
        return self.down_proj(self.hook_post(F.gelu(self.up_proj(x))))


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.use_ln = cfg.get("layernorm", False)
        self.ln1 = nn.LayerNorm(d) if self.use_ln else nn.Identity()
        self.attn = Attention(d, cfg["n_heads"])
        self.hook_resid_pre = HookPoint()
        self.hook_attn_out = HookPoint()
        self.mlp = MLPBlock(d, cfg["d_mlp"]) if cfg.get("d_mlp") else None
        self.ln2 = nn.LayerNorm(d) if (self.use_ln and self.mlp is not None) else nn.Identity()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x):
        x = self.hook_resid_pre(x)
        x = x + self.hook_attn_out(self.attn(self.ln1(x)))
        if self.mlp is not None:
            x = x + self.hook_mlp_out(self.mlp(self.ln2(x)))
        return self.hook_resid_post(x)


class TinyTransformer(nn.Module):
    """Decoder-only transformer over integer tokens. Optional LayerNorm / MLP (off = attention-only)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.pos_embed = nn.Embedding(cfg["max_len"], d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg["n_layers"]))
        self.ln_f = nn.LayerNorm(d) if cfg.get("layernorm", False) else nn.Identity()
        self.unembed = nn.Linear(d, cfg["vocab"], bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        T = tokens.shape[1]
        if T > self.cfg["max_len"]:
            raise ValueError(f"sequence length {T} exceeds max_len {self.cfg['max_len']}")
        x = self.embed(tokens) + self.pos_embed(torch.arange(T, device=tokens.device))[None]
        for b in self.blocks:
            x = b(x)
        return self.unembed(self.ln_f(x))


ARCHS = {"tiny_mlp": TinyMLP, "tiny_cnn": TinyCNN, "tiny_transformer": TinyTransformer}


def build_model(cfg: dict) -> nn.Module:
    arch = cfg.get("wsp_arch")
    if arch not in ARCHS:
        raise ValueError(f"unknown wsp_arch {arch!r}; known: {sorted(ARCHS)}")
    return ARCHS[arch](cfg)
