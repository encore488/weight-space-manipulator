import importlib.util
import json
from pathlib import Path

script = Path(__file__).resolve().parents[1] / "scripts" / "gen_tool_docs.py"
spec = importlib.util.spec_from_file_location("gen_tool_docs", script)
assert spec is not None and spec.loader is not None
gen_tool_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_tool_docs)
ExampleFormatter = gen_tool_docs.ExampleFormatter


def test_example_ids_stay_linked_across_calls_and_results():
    formatter = ExampleFormatter()
    first_session = "sess_5e16e3e6270a"
    second_session = "sess_19216b9148c2"
    patch = "p_4bb5b964d469ed69"

    call = json.loads(formatter.dumps({"session_id": first_session, "patch_id": patch}))
    result = json.loads(formatter.dumps({
        "details": [first_session, second_session],
        "message": f"Patch {patch} by agent:{first_session}",
    }))

    assert call == {"session_id": "sess_000000000001", "patch_id": "p_0000000000000001"}
    assert result["details"] == [call["session_id"], "sess_000000000002"]
    assert result["message"] == f"Patch {call['patch_id']} by agent:{call['session_id']}"


def test_example_times_are_stable_without_changing_metrics_or_hashes():
    formatter = ExampleFormatter()
    value = {
        "log": [{"ts": 1794774459.123, "created_at": 1794774459, "minutes": 0.34, "accuracy": 0.25}],
        "markdown": "# Example\n\n- Generated: 2026-11-15 20:27:48 by WSP Toolkit 0.3.0",
        "model_hash": "cdd2f8c2c3088c78",
    }

    result = json.loads(formatter.dumps(value))

    assert result["log"] == [{"ts": 0.0, "created_at": 0, "minutes": 0.0, "accuracy": 0.25}]
    assert "1970-01-01 00:00:00" in result["markdown"]
    assert result["model_hash"] == value["model_hash"]
