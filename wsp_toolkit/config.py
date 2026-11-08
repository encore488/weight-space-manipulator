"""Configuration: defaults from default_config.toml, merged with an optional user TOML (WSP_CONFIG)
and a few environment overrides (WSP_DATA_DIR)."""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path(__file__).with_name("default_config.toml")

ACCESS_LEVELS: dict[str, frozenset[str]] = {
    "weights_only": frozenset({"weights"}),
    "weights+activations": frozenset({"weights", "activations"}),
    "weights+activations+outputs": frozenset({"weights", "activations", "outputs"}),
}
ACCESS_ORDER = ["weights_only", "weights+activations", "weights+activations+outputs"]


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
        with open(_DEFAULT_PATH, "rb") as f:
            data = tomllib.load(f)
        path = path or os.environ.get("WSP_CONFIG")
        if path:
            with open(path, "rb") as f:
                data = _merge(data, tomllib.load(f))
        if os.environ.get("WSP_DATA_DIR"):
            data["paths"]["data_dir"] = os.environ["WSP_DATA_DIR"]
        if overrides:
            data = _merge(data, overrides)
        cfg = cls(raw=data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        lvl = self.raw["session"]["access_level"]
        if lvl not in ACCESS_LEVELS:
            raise ValueError(f"session.access_level must be one of {ACCESS_ORDER}, got {lvl!r}")
        if self.raw["limits"]["max_response_bytes"] < 2000:
            raise ValueError("limits.max_response_bytes must be >= 2000 so errors and summaries fit")

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    @property
    def data_dir(self) -> Path:
        p = Path(os.path.expanduser(self.raw["paths"]["data_dir"])).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def subdir(self, name: str) -> Path:
        p = self.data_dir / name
        p.mkdir(parents=True, exist_ok=True)
        return p
