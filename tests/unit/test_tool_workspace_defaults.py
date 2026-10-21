"""What a workspace with no stored tool policy gets: the setting, on by default."""

import pytest

from gateway.core.config import GatewayConfig

_FIELDS = ("web_search_workspace_default", "code_execution_workspace_default")


@pytest.mark.parametrize("field", _FIELDS)
def test_unset_is_on(field: str) -> None:
    assert getattr(GatewayConfig(), field) is True


@pytest.mark.parametrize("field", _FIELDS)
@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), (None, True)],
)
def test_the_setting_wins_and_null_is_on(field: str, value: bool | None, expected: bool) -> None:
    """A null an earlier release accepted still reads as the default."""
    config = GatewayConfig.model_validate({field: value})

    assert getattr(config, field) is expected


@pytest.mark.parametrize("field", _FIELDS)
def test_the_environment_sets_it(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"OTARI_{field.upper()}", "false")

    assert getattr(GatewayConfig(), field) is False
