"""The CI gate must not accept cancelled, missing, or unexpectedly skipped jobs."""

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_required_jobs.py"
_SPEC = importlib.util.spec_from_file_location("check_required_jobs", _PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def _detect():
    return {"result": "success", "outputs": {
        **dict.fromkeys(_MOD.LANE_OUTPUTS, "false"), "event_name": "pull_request",
    }}


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped", "unknown", None])
@pytest.mark.parametrize("required", [True, False])
def test_only_success_or_explicitly_unaffected_skip_is_accepted(result, required):
    needs = {"detect": _detect(), "tests": {"result": result}}
    expected = result == "success" or (result == "skipped" and required is False)
    assert (_MOD.failed_jobs(needs, {"detect": True, "tests": required}) == []) is expected


@pytest.mark.parametrize(
    "needs,required,failed",
    [
        ({"detect": _detect()}, {"detect": True, "tests": True}, True),
        ({"detect": _detect()}, {}, True),
        ({"detect": _detect(), "tests": {"result": "skipped"}},
         {"detect": True, "tests": "false"}, True),
        ({"detect": {"result": "failure"}, "tests": {"result": "skipped"}},
         {"detect": True, "tests": False}, True),
        ({"detect": {"result": "success"}, "tests": {"result": "skipped"}},
         {"detect": True, "tests": False}, True),
        ({"detect": {"result": "success", "outputs": {"event_name": "pull_request"}},
          "tests": {"result": "skipped"}}, {"detect": True, "tests": False}, True),
        ({"detect": _detect(), "tests": {"result": "skipped"}},
         {"detect": True, "tests": False}, False),
        ({"detect": {**_detect(), "outputs": {**_detect()["outputs"], "python": ""}},
          "tests": {"result": "skipped"}}, {"detect": True, "tests": False}, True),
    ],
)
def test_gate_exit_and_comment_output(tmp_path, monkeypatch, needs, required, failed):
    output = tmp_path / "github-output"
    monkeypatch.setenv("NEEDS", json.dumps(needs))
    monkeypatch.setenv("REQUIRED", json.dumps(required))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert _MOD.main() == int(failed)
    key, value = output.read_text(encoding="utf-8").strip().split("=", 1)
    assert key == "needs-json"
    assert json.loads(value) == {name: info["result"] for name, info in needs.items()}
