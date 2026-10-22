"""Settings of the removed runtime modes are ignored rather than refusing the start."""

import os
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest

from gateway.core.config import load_config


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Hide OTARI_ variables and .env files, and restore os.environ, which loading writes to."""
    monkeypatch.chdir(tmp_path)
    with mock.patch.dict(os.environ):
        for key in [key for key in os.environ if key.startswith("OTARI_")]:
            del os.environ[key]
        yield


def test_a_config_from_hybrid_or_hosted_mode_still_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_MODE", "hybrid")
    monkeypatch.setenv("OTARI_AI_TOKEN", "tok-left-over")
    monkeypatch.setenv("OTARI_DATA_PLANE_URL", "not a url")
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "mode: hosted\n"
        "data_plane_url: https://data.example\n"
        "platform:\n"
        "  base_url: https://platform.example\n"
        "  usage_inline_timeout_ms: not-a-number\n"
        "web_search_workspace_default: null\n",
        encoding="utf-8",
    )

    config = load_config(str(config_file))

    assert not hasattr(config, "mode")
    assert not hasattr(config, "platform")
    assert config.web_search_workspace_default is True
