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
