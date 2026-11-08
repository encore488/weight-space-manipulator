import sqlite3

import pytest

from wsp_toolkit.config import Config
from wsp_toolkit.core.db import Store
from wsp_toolkit.fixtures.train import train_induction_transformer, train_mlp_parity


def test_config_defaults_and_validation(tmp_path):
    cfg = Config.load(overrides={"paths": {"data_dir": str(tmp_path)}})
    assert cfg.get("session.forward_pass_budget") == 50
    assert cfg.get("limits.max_response_bytes") >= 2000
    with pytest.raises(ValueError):
        Config.load(overrides={"session": {"access_level": "everything"}})


def test_event_log_is_append_only(tmp_path):
    st = Store(tmp_path / "x.sqlite")
    eid = st.log_event(session_id="s", actor="agent", kind="tool", tool="t", args={"a": 1}, result={"r": 2})
    assert st.events("s")[0]["result_hash"]
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        st.execute("UPDATE events SET tool='x' WHERE id=?", (eid,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        st.execute("DELETE FROM events")


def test_mlp_fixture_reproducible(tmp_path):
    a = train_mlp_parity(tmp_path / "a", seed=0)
    b = train_mlp_parity(tmp_path / "b", seed=0)
    c = train_mlp_parity(tmp_path / "c", seed=1)
    assert a == b and a != c


def test_transformer_fixture_reproducible(tmp_path):
    a = train_induction_transformer(tmp_path / "a", seed=0, steps=30)
    b = train_induction_transformer(tmp_path / "b", seed=0, steps=30)
    assert a == b


def test_full_fixtures_learn_their_tasks(fixtures_dir):
    import json
    d, _ = fixtures_dir
    gt = json.loads((d / "induction_tf" / "ground_truth.json").read_text())
    assert gt["induction_accuracy"] > 0.9 and gt["counting_accuracy"] > 0.9
    assert gt["induction_heads"] and all(h.startswith("1.") for h in gt["induction_heads"])
    assert gt["previous_token_heads"]
    broken = json.loads((d / "induction_tf_broken" / "ground_truth.json").read_text())
    assert broken["induction_accuracy"] < 0.3
    mlp = json.loads((d / "mlp_parity3" / "ground_truth.json").read_text())
    assert mlp["train_accuracy"] == 1.0
