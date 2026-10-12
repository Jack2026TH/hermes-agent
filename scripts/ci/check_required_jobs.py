"""Fail CI unless required jobs succeeded and optional jobs were safely skipped."""

from __future__ import annotations

import json
import os
from pathlib import Path

LANE_OUTPUTS = (
    "python", "python_prod", "frontend", "site", "scan", "deps", "uv_lock",
    "npm_lock", "installer", "desktop_updater", "rust", "docker_meta",
    "mcp_catalog", "ci_review",
)


def failed_jobs(needs: dict, required: dict) -> list[str]:
    """Unknown/missing results and missing requirement declarations fail closed."""
    failed = set()
    outputs = needs.get("detect", {}).get("outputs", {})
    if (
        any(outputs.get(name) not in ("true", "false") for name in LANE_OUTPUTS)
        or outputs.get("event_name") not in ("pull_request", "push", "workflow_dispatch")
    ):
        # An absent output otherwise compares unequal to 'true', silently
        # turning a missing classifier result into permission to skip a lane.
        failed.add("detect")
    for name in sorted(needs.keys() | required.keys()):
        result = needs.get(name, {}).get("result")
        expected = required.get(name)
        if type(expected) is not bool or (
            result != "success" and not (result == "skipped" and expected is False)
        ):
            failed.add(name)
    return sorted(failed)


def main() -> int:
    needs = json.loads(os.environ["NEEDS"])
    required = json.loads(os.environ["REQUIRED"])
    compact = {name: info.get("result") for name, info in needs.items()}
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"needs-json={json.dumps(compact)}\n")
    failed = failed_jobs(needs, required)
    for name in sorted(needs.keys() | required.keys()):
        result = needs.get(name, {}).get("result", "missing")
        print(f"{'FAIL' if name in failed else 'PASS'} {name}: {result}")
    if failed:
        print(f"::error::Required CI results not satisfied: {', '.join(failed)}")
        return 1
    print("All required checks succeeded; unaffected lanes may be skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
