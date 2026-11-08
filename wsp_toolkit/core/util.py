"""Shared helpers: content hashing, float rounding, slice parsing, response capping, actionable errors."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch


class WSPError(Exception):
    """An error with an actionable message meant to be shown verbatim to agents and humans."""

    def __init__(self, message: str, *, code: str = "error", hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.hint = hint

    def to_dict(self) -> dict:
        d = {"error": str(self), "code": self.code}
        if self.hint:
            d["hint"] = self.hint
        return d


def now() -> float:
    return time.time()


# ---------------------------------------------------------------- hashing

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tensor_hash(t: torch.Tensor) -> str:
    """Hash of dtype + shape + raw bytes. Independent of file format and device."""
    t = t.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    if t.dtype == torch.bfloat16:
        h.update(t.view(torch.int16).numpy().tobytes())
    else:
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def state_hash(state: Mapping[str, torch.Tensor], tensor_hashes: Mapping[str, str] | None = None) -> str:
    """Model hash: sha256 over sorted (name, tensor_hash) pairs."""
    h = hashlib.sha256()
    for name in sorted(state):
        th = tensor_hashes[name] if tensor_hashes and name in tensor_hashes else tensor_hash(state[name])
        h.update(name.encode())
        h.update(b"\0")
        h.update(th.encode())
        h.update(b"\n")
    return h.hexdigest()


def json_hash(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode())


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def short(h: str, n: int = 16) -> str:
    return h[:n]


# ---------------------------------------------------------------- numbers

def rnd(x: float, sig: int = 4) -> float | None:
    """Round to `sig` significant digits; NaN/inf become None so JSON stays valid."""
    x = float(x)
    if not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")


def rnd_list(xs: Iterable[float], sig: int = 4) -> list:
    return [rnd(x, sig) for x in xs]


def rnd_nested(t: torch.Tensor | np.ndarray, sig: int = 4) -> list:
    arr = t.detach().float().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=float)
    if arr.ndim == 0:
        return rnd(arr.item(), sig)  # type: ignore[return-value]
    return [rnd_nested(a, sig) if a.ndim else rnd(a, sig) for a in arr] if arr.ndim > 1 else rnd_list(arr, sig)


# ---------------------------------------------------------------- slices

def parse_slice(spec: str | None, shape: tuple[int, ...], tensor_name: str = "tensor") -> tuple[slice, ...]:
    """Parse numpy-style slice text like "0:4, 10:20" or "3, :" against a shape, with actionable errors.

    Integers keep the dimension (converted to i:i+1) so results always have the same rank as the tensor.
    """
    if spec is None or str(spec).strip() in ("", ":"):
        return tuple(slice(0, n) for n in shape)
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) > len(shape):
        raise WSPError(
            f"slice {spec!r} has {len(parts)} dims but {tensor_name} has shape {list(shape)} ({len(shape)} dims)",
            code="bad_slice",
        )
    out: list[slice] = []
    for d, n in enumerate(shape):
        p = parts[d] if d < len(parts) else ":"
        try:
            if ":" in p:
                a, _, b = p.partition(":")
                if ":" in b:
                    raise WSPError("step slices are not supported", code="bad_slice")
                start = int(a) if a.strip() else 0
                stop = int(b) if b.strip() else n
            else:
                start = int(p)
                stop = start + 1
        except ValueError as e:
            raise WSPError(f"cannot parse slice component {p!r} in {spec!r}: use 'start:stop' or an index",
                           code="bad_slice") from e
        if start < 0:
            start += n
        if stop < 0:
            stop += n
        if not (0 <= start < n) or not (start < stop <= n):
            raise WSPError(
                f"{tensor_name} has shape {list(shape)}; requested slice {p!r} on dim {d} exceeds it — "
                f"valid index range is 0..{n - 1}",
                code="slice_out_of_range",
            )
        out.append(slice(start, stop))
    return tuple(out)


def slice_to_str(sl: tuple[slice, ...]) -> str:
    return ", ".join(f"{s.start}:{s.stop}" for s in sl)


def slice_numel(sl: tuple[slice, ...]) -> int:
    n = 1
    for s in sl:
        n *= s.stop - s.start
    return n


# ---------------------------------------------------------------- response capping

def json_size(obj: Any) -> int:
    return len(canonical_json(obj).encode())


def truncate_list(xs: list, max_items: int) -> tuple[list, bool]:
    if len(xs) <= max_items:
        return xs, False
    return xs[:max_items], True
