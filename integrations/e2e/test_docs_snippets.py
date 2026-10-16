"""Run every Python block on the agent-framework docs pages against a real Otari.

The pages are the test: a block a reader copies is the block that runs here, with
only the two variables the page tells them to export. A block that cannot run
on its own does not belong on these pages.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from conftest import MASTER_KEY, REPO_ROOT, Gateway

PAGES = ["use-with-pydantic-ai.md", "use-with-langchain.md", "use-with-langgraph.md"]
BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
SNIPPET_TIMEOUT_SECONDS = 120


def _snippets() -> list[pytest.param]:
    """Collect every Python block on the pages, one test parameter each."""
    params = []
    for page in PAGES:
        text = (REPO_ROOT / "docs" / page).read_text(encoding="utf-8")
        blocks = BLOCK.findall(text)
        assert blocks, f"docs/{page} has no Python blocks to run"
        params += [pytest.param(page, code, id=f"{Path(page).stem}-{i}") for i, code in enumerate(blocks, 1)]
    return params


def _admin_get(gateway: Gateway, path: str) -> Any:
    """GET an admin endpoint with the master key and return the decoded JSON."""
    request = urllib.request.Request(f"{gateway.base_url}{path}", headers={"Otari-Key": MASTER_KEY})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


@pytest.mark.parametrize(("page", "code"), _snippets())
def test_snippet_runs_through_otari(gateway: Gateway, page: str, code: str) -> None:
    """A block runs on its own with the two documented variables, prints, and reaches the provider through Otari."""
    before = len(gateway.provider.requests)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=gateway.env(),
        capture_output=True,
        text=True,
        timeout=SNIPPET_TIMEOUT_SECONDS,
        check=False,
    )
    assert result.returncode == 0, (
        f"A block on docs/{page} failed.\n--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}"
        f"\n--- gateway log\n{gateway.tail()}"
    )
    assert result.stdout.strip(), f"A block on docs/{page} printed nothing, so a reader cannot see it work"
    if not gateway.live:
        assert len(gateway.provider.requests) > before, (
            f"A block on docs/{page} never reached the provider through Otari"
        )


def test_otari_recorded_the_snippets_usage(gateway: Gateway) -> None:
    """What the pages promise a reader: the framework's calls are tracked per key."""
    rows = _admin_get(gateway, "/api/v1/usage?limit=100")
    ours = [row for row in rows if row.get("user_id") == "framework-e2e" and row.get("status") == "success"]
    assert ours, f"Otari recorded no successful usage for the snippets' key: {rows!r}"
