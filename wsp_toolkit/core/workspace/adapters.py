"""Model adapters: one small interface over native fixture models and Hugging Face causal LMs.

The rest of the toolkit only talks to ``ModelAdapter``; it never imports transformers or touches
model-specific module paths. Activation / attention capture uses plain PyTorch forward hooks
(see docs/decisions: nnsight / TransformerLens were not adopted).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file

from ..util import WSPError
from .archs import HookPoint, build_model


@dataclass
class HookSpec:
    alias: str          # stable agent-facing name, e.g. "L1.resid_post"
    module: str         # module path inside the model
    mode: str = "out"   # out | out0 (first element of tuple output) | pre (first positional input)
    description: str = ""


@dataclass
class HeadLayout:
    axis: int       # tensor axis that indexes heads (0 = rows, 1 = cols)
    n_heads: int
    d_head: int
    layer: int


class ModelAdapter:
    kind: str = "base"

    def __init__(self, model: nn.Module, config: dict, tokenizer: Any = None):
        self.model = model.eval()
        self.config = config
        self.tokenizer = tokenizer
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ---------------------------------------------------------------- tensors
    def state(self) -> dict[str, torch.Tensor]:
        """Live tensors (share storage with the model). Writing into them edits the working copy."""
        return dict(self.model.state_dict(keep_vars=False))

    def tied_groups(self) -> dict[str, list[str]]:
        groups: dict[int, list[str]] = {}
        for name, t in self.state().items():
            groups.setdefault(t.untyped_storage().data_ptr(), []).append(name)
        out = {}
        for names in groups.values():
            if len(names) > 1:
                for n in names:
                    out[n] = [m for m in names if m != n]
        return out

    def layer_of(self, name: str) -> str:
        m = re.search(r"(?:layers|blocks|h)\.(\d+)\.", name)
        if m:
            return f"layer {m.group(1)}"
        if re.search(r"embed|wte|wpe", name):
            return "embedding"
        if re.search(r"unembed|lm_head|norm_f|ln_f|final", name):
            return "output"
        return "other"

    def head_layout(self, name: str) -> HeadLayout | None:
        return None

    # ---------------------------------------------------------------- language-model bits
    @property
    def is_lm(self) -> bool:
        return False

    def unembed(self) -> torch.Tensor | None:
        return None

    def token_str(self, i: int) -> str:
        return str(i)

    def vocab_size(self) -> int:
        return 0

    # ---------------------------------------------------------------- running
    def hook_specs(self) -> list[HookSpec]:
        return []

    def encode(self, inp: Any) -> Any:
        raise NotImplementedError

    def describe_input(self, enc: Any) -> dict:
        return {}

    def forward(self, enc: Any) -> torch.Tensor:
        with torch.no_grad():
            return self.model(enc)

    def attention_patterns(self, enc: Any, layers: list[int] | None = None) -> dict[int, torch.Tensor]:
        raise WSPError(f"model kind {self.kind!r} has no attention layers", code="unsupported")

    def n_layers(self) -> int:
        return 0

    @contextmanager
    def capture(self, aliases: list[str]):
        """Context manager yielding a dict alias -> captured tensor (filled during forward)."""
        specs = {s.alias: s for s in self.hook_specs()}
        unknown = [a for a in aliases if a not in specs]
        if unknown:
            raise WSPError(f"unknown hook point(s) {unknown}; available: {sorted(specs)[:60]}", code="bad_hook")
        modules = dict(self.model.named_modules())
        store: dict[str, torch.Tensor] = {}
        handles = []
        for a in aliases:
            s = specs[a]
            mod = modules[s.module]
            if s.mode == "pre":
                handles.append(mod.register_forward_pre_hook(
                    lambda m, args, a=a: store.__setitem__(a, args[0].detach().float())))
            else:
                def hook(m, i, o, a=a, s=s):
                    val = o[0] if (s.mode == "out0" and isinstance(o, (tuple, list))) else o
                    store[a] = val.detach().float()
                handles.append(mod.register_forward_hook(hook))
        try:
            yield store
        finally:
            for h in handles:
                h.remove()

    def generate(self, enc: Any, settings: dict) -> dict:
        raise WSPError(f"model kind {self.kind!r} cannot generate text", code="unsupported")


# =============================================================================== native models

class NativeAdapter(ModelAdapter):
    def __init__(self, model: nn.Module, config: dict):
        super().__init__(model, config)
        self.kind = config["wsp_arch"]

    def hook_specs(self) -> list[HookSpec]:
        specs = []
        for name, mod in self.model.named_modules():
            if not isinstance(mod, HookPoint):
                continue
            alias = name
            m = re.match(r"blocks\.(\d+)\.(?:attn\.|mlp\.)?hook_(\w+)$", name)
            if m:
                alias = f"L{m.group(1)}.{m.group(2)}"
            m2 = re.match(r"hooks\.(\d+)$", name)
            if m2:
                alias = f"L{m2.group(1)}.hidden"
            if name == "hook_conv":
                alias = "conv"
            if alias.endswith(".pattern"):
                continue  # attention patterns go through probe_attention
            specs.append(HookSpec(alias, name, "out", _HOOK_DESCRIPTIONS.get(alias.split(".")[-1], "")))
        return specs

    def n_layers(self) -> int:
        if self.kind == "tiny_transformer":
            return self.config["n_layers"]
        if self.kind == "tiny_mlp":
            return len(self.config["hidden"]) + 1
        return 1

    def head_layout(self, name: str) -> HeadLayout | None:
        if self.kind != "tiny_transformer":
            return None
        m = re.match(r"blocks\.(\d+)\.attn\.(q|k|v|o)_proj\.weight$", name)
        if not m:
            return None
        H = self.config["n_heads"]
        return HeadLayout(axis=1 if m.group(2) == "o" else 0, n_heads=H, d_head=self.config["d_model"] // H,
                          layer=int(m.group(1)))

    @property
    def is_lm(self) -> bool:
        return self.kind == "tiny_transformer"

    def unembed(self):
        return self.model.unembed.weight if self.is_lm else None

    def vocab_size(self) -> int:
        return self.config.get("vocab", 0)

    def token_str(self, i: int) -> str:
        names = self.config.get("token_names")
        return names[i] if names and 0 <= i < len(names) else str(i)

    def encode(self, inp: Any):
        if self.kind == "tiny_transformer":
            if isinstance(inp, str):
                names = self.config.get("token_names") or []
                lookup = {n: i for i, n in enumerate(names)}
                try:
                    ids = [lookup[t] if t in lookup else int(t) for t in inp.split()]
                except ValueError as e:
                    raise WSPError(f"cannot parse token string {inp!r}; use space-separated token names "
                                   f"like '<bos> t3 t7' or a list of ints", code="bad_input") from e
            else:
                ids = [int(x) for x in inp]
            V, T = self.config["vocab"], self.config["max_len"]
            if not ids or len(ids) > T:
                raise WSPError(f"token sequence length {len(ids)} must be 1..{T}", code="bad_input")
            bad = [x for x in ids if not 0 <= x < V]
            if bad:
                raise WSPError(f"token ids {bad[:5]} out of range 0..{V - 1}", code="bad_input")
            return torch.tensor([ids])
        if self.kind == "tiny_cnn":
            x = torch.tensor(inp, dtype=torch.float32)
            return x[None] if x.dim() == 2 else x
        x = torch.tensor(inp, dtype=torch.float32)
        d = self.config["d_in"]
        if x.dim() != 1 or x.shape[0] != d:
            raise WSPError(f"input must be a vector of length {d}, got shape {list(x.shape)}", code="bad_input")
        return x[None]

    def describe_input(self, enc) -> dict:
        if self.kind == "tiny_transformer":
            ids = enc[0].tolist()
            return {"tokens": [self.token_str(i) for i in ids], "ids": ids}
        return {}

    def attention_patterns(self, enc, layers=None):
        if self.kind != "tiny_transformer":
            return super().attention_patterns(enc, layers)
        out: dict[int, torch.Tensor] = {}
        layers = list(range(self.n_layers())) if layers is None else layers
        handles = []
        for li in layers:
            handles.append(self.model.blocks[li].attn.hook_pattern.register_forward_hook(
                lambda m, i, o, li=li: out.__setitem__(li, o.detach().float())))
        try:
            with torch.no_grad():
                self.model(enc)
        finally:
            for h in handles:
                h.remove()
        return out

    def generate(self, enc, settings: dict) -> dict:
        if self.kind != "tiny_transformer":
            return super().generate(enc, settings)
        return _sample_loop(lambda ids: self.model(ids)[:, -1], enc, settings, self.config["max_len"], None,
                            self.token_str)


_HOOK_DESCRIPTIONS = {
    "resid_pre": "residual stream entering the block [T, d_model]",
    "resid_post": "residual stream leaving the block [T, d_model]",
    "attn_out": "attention output added to the residual [T, d_model]",
    "mlp_out": "MLP output added to the residual [T, d_model]",
    "z": "per-head attention output before o_proj [T, n_heads, d_head]",
    "post": "MLP hidden activations after nonlinearity [T, d_mlp]",
    "hidden": "hidden layer activations after nonlinearity",
    "conv": "conv feature maps after ReLU",
}


def _sample_loop(step: Callable, enc: torch.Tensor, settings: dict, max_len: int | None, eos: int | None,
                 token_str: Callable) -> dict:
    """Simple sampler used for native LMs (HF models use model.generate)."""
    g = torch.Generator().manual_seed(int(settings.get("seed", 0)))
    ids = enc.clone()
    temp = float(settings.get("temperature", 0.0))
    max_new = int(settings.get("max_new_tokens", 16))
    rep = float(settings.get("repetition_penalty", 1.0))
    new = []
    hit_limit = True
    for _ in range(max_new):
        if max_len and ids.shape[1] >= max_len:
            break
        with torch.no_grad():
            logits = step(ids)[0].float()
        if rep != 1.0:
            for t in set(ids[0].tolist()):
                logits[t] = logits[t] / rep if logits[t] > 0 else logits[t] * rep
        if temp <= 0:
            nxt = int(logits.argmax())
        else:
            nxt = int(torch.multinomial((logits / temp).softmax(-1), 1, generator=g))
        new.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], 1)
        if eos is not None and nxt == eos:
            hit_limit = False
            break
    return {"new_ids": new, "text": " ".join(token_str(i) for i in new), "hit_token_limit": hit_limit and
            len(new) >= max_new}


# =============================================================================== HF models

class HFAdapter(ModelAdapter):
    kind = "hf_causal_lm"

    def __init__(self, model: nn.Module, config: dict, tokenizer: Any):
        super().__init__(model, config, tokenizer)
        self._layers_path, self._layers = self._find_layers()

    def _find_layers(self) -> tuple[str, nn.ModuleList]:
        best = None
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.ModuleList) and name.endswith("layers") and len(mod) > 0:
                if "vision" in name or "visual" in name:
                    continue
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
        if best is None:
            raise WSPError("could not find the decoder layer list in this HF model", code="unsupported")
        return best

    def n_layers(self) -> int:
        return len(self._layers)

    @property
    def is_lm(self) -> bool:
        return True

    def unembed(self):
        return self.model.get_output_embeddings().weight

    def final_norm_weight(self) -> torch.Tensor | None:
        root = self._layers_path.rsplit(".layers", 1)[0]
        mods = dict(self.model.named_modules())
        norm = mods.get(f"{root}.norm") or mods.get(f"{root}.final_layernorm")
        return getattr(norm, "weight", None) if norm is not None else None

    def vocab_size(self) -> int:
        return int(self.unembed().shape[0])

    def token_str(self, i: int) -> str:
        return self.tokenizer.decode([int(i)]) if self.tokenizer is not None else str(i)

    def hook_specs(self) -> list[HookSpec]:
        specs = []
        mods = dict(self.model.named_modules())
        for i in range(len(self._layers)):
            base = f"{self._layers_path}.{i}"
            specs.append(HookSpec(f"L{i}.resid_post", base, "out0", "residual stream leaving the layer"))
            for sub, alias, mode, desc in [("self_attn", "attn_out", "out0", "attention output"),
                                           ("linear_attn", "attn_out", "out0", "linear-attention output"),
                                           ("mlp", "mlp_out", "out", "MLP output"),
                                           ("mlp.down_proj", "mlp_hidden", "pre", "MLP hidden (input to down_proj)")]:
                if f"{base}.{sub}" in mods and not any(s.alias == f"L{i}.{alias}" for s in specs):
                    specs.append(HookSpec(f"L{i}.{alias}", f"{base}.{sub}", mode, desc))
        return specs

    def head_layout(self, name: str) -> HeadLayout | None:
        m = re.search(r"layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$", name)
        if not m:
            return None
        cfg = getattr(self.model.config, "text_config", self.model.config)
        H = getattr(cfg, "num_attention_heads", None)
        Hkv = getattr(cfg, "num_key_value_heads", H)
        dh = getattr(cfg, "head_dim", None) or (cfg.hidden_size // H)
        which = m.group(2)
        if which == "o":
            return HeadLayout(1, H, dh, int(m.group(1)))
        if which == "q":
            t = self.state()[name]
            per = t.shape[0] // H  # q_proj may include a gate (2*dh per head) in some archs
            return HeadLayout(0, H, per, int(m.group(1)))
        return HeadLayout(0, Hkv, dh, int(m.group(1)))

    def encode(self, inp: Any):
        if isinstance(inp, str):
            ids = self.tokenizer(inp, return_tensors="pt")["input_ids"]
        else:
            ids = torch.tensor([[int(x) for x in inp]])
        return ids

    def describe_input(self, enc) -> dict:
        ids = enc[0].tolist()
        return {"tokens": [self.token_str(i) for i in ids], "ids": ids}

    def forward(self, enc):
        with torch.no_grad():
            return self.model(input_ids=enc).logits

    def attention_patterns(self, enc, layers=None):
        with torch.no_grad():
            out = self.model(input_ids=enc, output_attentions=True)
        atts = out.attentions
        if atts is None or all(a is None for a in atts):
            raise WSPError("this model did not return attention weights; load with attn_implementation='eager'",
                           code="unsupported")
        res = {}
        for i, a in enumerate(atts):
            if a is None or (layers is not None and i not in layers):
                continue
            res[i] = a.detach().float()
        return res

    def generate(self, enc, settings: dict) -> dict:
        gen_kwargs = {"max_new_tokens": int(settings.get("max_new_tokens", 64))}
        temp = float(settings.get("temperature", 0.0))
        if temp > 0:
            gen_kwargs.update(do_sample=True, temperature=temp, top_p=float(settings.get("top_p", 1.0)))
            if settings.get("top_k"):
                gen_kwargs["top_k"] = int(settings["top_k"])
        else:
            gen_kwargs["do_sample"] = False
        if settings.get("repetition_penalty", 1.0) != 1.0:
            gen_kwargs["repetition_penalty"] = float(settings["repetition_penalty"])
        torch.manual_seed(int(settings.get("seed", 0)))
        with torch.no_grad():
            out = self.model.generate(input_ids=enc, attention_mask=torch.ones_like(enc), **gen_kwargs)
        new = out[0, enc.shape[1]:].tolist()
        eos = self.tokenizer.eos_token_id if self.tokenizer is not None else None
        eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        return {"new_ids": new, "text": self.tokenizer.decode(new, skip_special_tokens=True),
                "hit_token_limit": len(new) >= gen_kwargs["max_new_tokens"] and not (new and new[-1] in eos_ids)}


# =============================================================================== raw tensors

class RawAdapter(ModelAdapter):
    """A bag of tensors with no runnable architecture: inspection and edits only."""

    kind = "raw"

    def __init__(self, tensors: dict[str, torch.Tensor], config: dict):
        holder = nn.Module()
        for i, (_k, v) in enumerate(sorted(tensors.items())):
            holder.register_buffer(f"t{i}", v)
        self._names = {f"t{i}": k for i, k in enumerate(sorted(tensors))}
        super().__init__(holder, config)

    def state(self):
        sd = self.model.state_dict()
        return {self._names[k]: v for k, v in sd.items()}

    def encode(self, inp):
        raise WSPError("raw tensor workspaces cannot run forward passes (no architecture)", code="unsupported")


# =============================================================================== loading

def load_native(path: Path) -> NativeAdapter:
    cfg = json.loads((path / "config.json").read_text())
    model = build_model(cfg)
    sd = load_file(str(path / "model.safetensors"))
    model.load_state_dict(sd)
    return NativeAdapter(model, cfg)


def load_hf(path: Path, dtype: str = "float32") -> HFAdapter:
    try:
        import transformers
    except ImportError as e:
        raise WSPError("loading Hugging Face models needs the 'hf' extra: pip install 'wsp-toolkit[hf]'",
                       code="missing_dependency") from e
    td = getattr(torch, dtype)
    kwargs = {"dtype": td, "attn_implementation": "eager"}
    model = None
    errors = []
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(str(path), **kwargs)
            break
        except Exception as e:  # noqa: BLE001 - try the next auto class
            errors.append(f"{cls_name}: {e}")
    if model is None:
        raise WSPError(f"could not load HF model at {path}: {errors}", code="load_failed")
    tok = None
    try:
        tok = transformers.AutoTokenizer.from_pretrained(str(path))
    except Exception:  # noqa: BLE001 - tokenizer optional; token ids still work
        tok = None
    cfg = {"hf_model_type": getattr(model.config, "model_type", "?"), "path": str(path)}
    return HFAdapter(model, cfg, tok)


def load_path(path: Path) -> ModelAdapter:
    if path.is_file() and path.suffix == ".safetensors":
        return RawAdapter(load_file(str(path)), {"path": str(path)})
    cfgp = path / "config.json"
    if cfgp.exists():
        cfg = json.loads(cfgp.read_text())
        if "wsp_arch" in cfg:
            return load_native(path)
        if "model_type" in cfg or "architectures" in cfg:
            return load_hf(path)
    files = sorted(path.glob("*.safetensors"))
    if files:
        tensors: dict[str, torch.Tensor] = {}
        for f in files:
            tensors.update(load_file(str(f)))
        return RawAdapter(tensors, {"path": str(path)})
    raise WSPError(f"no loadable model at {path}: expected config.json + *.safetensors, or a .safetensors file",
                   code="not_found")
