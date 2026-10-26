"""Unit tests for the README quickstart check (scripts/readme_quickstart_check.py).

The check's end-to-end proof is the CI step that runs it against a built image.
What is pinned here is that it finds the command a reader copies, and changes
nothing in it but the image.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "readme_quickstart_check.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("readme_quickstart_check", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load()


def test_it_finds_the_readme_quickstart() -> None:
    command = check.quickstart_command(check.README.read_text(encoding="utf-8"))

    assert command.startswith("docker run ")
    assert command.rstrip().endswith("otari serve")


def test_only_the_image_changes() -> None:
    command = check.quickstart_command(check.README.read_text(encoding="utf-8"))

    swapped = check.for_image(command, "otari-pr:abc")

    assert swapped == command.replace(check.PUBLISHED_IMAGE, "otari-pr:abc")
    assert check.PUBLISHED_IMAGE not in swapped


def test_a_command_naming_another_image_is_refused() -> None:
    with pytest.raises(check.QuickstartFailure, match="exactly once"):
        check.for_image("docker run mzdotai/otari:0.32.0 otari serve", "otari-pr:abc")


def test_a_quickstart_that_is_not_docker_run_is_refused() -> None:
    readme = "# Otari\n\n## Quickstart\n\n```bash\npip install otari\n```\n\n## Next\n"

    with pytest.raises(check.QuickstartFailure, match="not a 'docker run'"):
        check.quickstart_command(readme)


def test_a_block_in_a_later_section_is_not_the_quickstart() -> None:
    readme = "# Otari\n\n## Quickstart\n\nSoon.\n\n## Compose\n\n```bash\ndocker run x\n```\n"

    with pytest.raises(check.QuickstartFailure, match="no bash block"):
        check.quickstart_command(readme)


def test_it_reads_the_bootstrap_key_off_its_own_line() -> None:
    log = (
        "WARNING  No API keys found, so created a 'bootstrap' API key ... Save this key now:\n"
        "tk-abc_DEF-123\n"
        "INFO:     Application startup complete.\n"
    )

    assert check.bootstrap_key(log) == "tk-abc_DEF-123"


def test_no_bootstrap_key_is_a_failure() -> None:
    with pytest.raises(check.QuickstartFailure, match="no bootstrap API key"):
        check.bootstrap_key("INFO:     Application startup complete.\n")


def test_a_failure_prints_no_key() -> None:
    log = "Your master key:\notari-mk-abc_DEF\nSave this key now:\ntk-abc_DEF-123\nApplication startup complete.\n"

    redacted = check.redact(log)

    assert "otari-mk-abc" not in redacted
    assert "tk-abc" not in redacted
    assert "Application startup complete." in redacted
