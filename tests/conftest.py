"""Shared fixtures. Fixture models are trained once per test session into a temp data dir (CPU, ~25 s)."""

from __future__ import annotations

import pytest

from wsp_toolkit.config import Config
from wsp_toolkit.core.service import Toolkit


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory):
    from wsp_toolkit.fixtures.train import build_all

    d = tmp_path_factory.mktemp("fixtures")
    hashes = build_all(d, seed=0)
    return d, hashes


def make_toolkit(tmp_path, fixtures_dir, **over) -> Toolkit:
    import shutil

    src, _ = fixtures_dir
    data = tmp_path / "data"
    (data / "fixtures").mkdir(parents=True, exist_ok=True)
    for sub in src.iterdir():
        if not (data / "fixtures" / sub.name).exists():
            shutil.copytree(sub, data / "fixtures" / sub.name)
    overrides = {"paths": {"data_dir": str(data)}}
    for k, v in over.items():
        overrides.setdefault(k, {}).update(v)
    return Toolkit(Config.load(overrides=overrides))


@pytest.fixture
def tk(tmp_path, fixtures_dir):
    return make_toolkit(tmp_path, fixtures_dir, policy={"max_param_change_pct": 20.0})


@pytest.fixture
def call(tk):
    def _call(tool, **args):
        return tk.call(tool, args, actor="agent")
    return _call


@pytest.fixture
def sid(call):
    return call("workspace_open", source="fixture:induction_tf")["session_id"]
