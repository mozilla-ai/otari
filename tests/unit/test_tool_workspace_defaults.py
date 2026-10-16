"""What a workspace with no stored tool policy gets: the setting when set, else on except on a hosted deployment."""

from collections.abc import Callable

import pytest

from gateway.core.config import GatewayConfig

_RESOLVERS: dict[str, Callable[[GatewayConfig], bool]] = {
    "web_search_workspace_default": GatewayConfig.web_search_on_without_policy,
    "code_execution_workspace_default": GatewayConfig.code_execution_on_without_policy,
}


@pytest.mark.parametrize("field", list(_RESOLVERS))
@pytest.mark.parametrize(
    ("mode", "value", "expected"),
    [
        ("standalone", True, True),
        ("standalone", False, False),
        ("standalone", None, True),
        ("hosted", None, False),
        ("hosted", True, True),
    ],
)
def test_the_setting_wins_and_unset_follows_the_deployment(
    field: str, mode: str, value: bool | None, expected: bool
) -> None:
    config = GatewayConfig.model_validate({"mode": mode, field: value})

    assert _RESOLVERS[field](config) is expected


@pytest.mark.parametrize("field", list(_RESOLVERS))
def test_the_environment_sets_it(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"OTARI_{field.upper()}", "false")

    assert _RESOLVERS[field](GatewayConfig()) is False
