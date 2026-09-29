"""The idempotency service's decisions that need no database."""

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.inference import IdempotencyService
from gateway.services.secret_box import generate_secret_key


@pytest.mark.parametrize(
    ("retention_sec", "secret_set", "enabled"),
    [(86400, True, True), (0, True, False), (86400, False, False)],
)
def test_keys_are_honored_only_with_a_retention_and_a_secret(
    monkeypatch: pytest.MonkeyPatch, retention_sec: int, secret_set: bool, enabled: bool
) -> None:
    if secret_set:
        monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    else:
        monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)

    config = GatewayConfig(idempotency_retention_sec=retention_sec)

    assert IdempotencyService.is_enabled(config) is enabled
