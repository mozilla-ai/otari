"""Provider-native Files require usable resource limits before enablement."""

import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"files_retention_hours": 2161}, "90 days"),
        ({"files_max_bytes": 1024, "files_temporary_capacity_bytes": 1024}, "64 KiB"),
        ({"mode": "hosted"}, "explicit file-count"),
        ({"mode": "hosted", "files_max_count": 10, "files_max_outstanding_bytes": 1}, "Outstanding-byte quota"),
    ],
)
def test_native_file_limits(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object], message: str) -> None:
    monkeypatch.delenv("OTARI_AI_TOKEN", raising=False)
    with pytest.raises(ValidationError, match=message):
        GatewayConfig.model_validate({"files_provider_native_enabled": True, **overrides})


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("retention_hours", [None, 2160, 2161, 2880])
def test_hybrid_retention_limit_requires_native_files(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, retention_hours: int | None
) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gateway-token")
    settings = {
        "mode": "hybrid",
        "files_provider_native_enabled": enabled,
        "files_retention_hours": retention_hours,
    }
    if enabled and retention_hours is not None and retention_hours > 2160:
        with pytest.raises(ValidationError, match="90 days"):
            GatewayConfig.model_validate(settings)
    else:
        config = GatewayConfig.model_validate(settings)
        assert config.is_hybrid_mode
        assert config.files_retention_hours == retention_hours
