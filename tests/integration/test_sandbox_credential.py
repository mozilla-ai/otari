"""``/api/v1/tool-settings/sandbox/credential``: the hosted sandbox key, rotated without a redeploy.

A stored key wins over ``E2B_API_KEY`` and is what the next sandbox is leased
with, on the worker that took the write at once and on every other once its
refresher reloads the row.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.adapters import e2b_code_execution_adapter as e2b_adapter
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.tools import SandboxCredential, SandboxProvider
from gateway.services.secret_box import generate_secret_key
from gateway.services.tools import reset_sandbox_credential_cache, sandbox_api_key

from .conftest import build_test_client

_SETTINGS = f"{API_ROOT}/tool-settings"
_CREDENTIAL = f"{_SETTINGS}/sandbox/credential"

Start = Callable[[GatewayConfig], AbstractContextManager[TestClient]]


@pytest.fixture
def start(monkeypatch: pytest.MonkeyPatch) -> Iterator[Start]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    # The extra is optional; booting with sandbox_provider e2b only checks that it imports.
    monkeypatch.setattr(e2b_adapter, "_sdk", lambda: object())

    @contextmanager
    def _start(config: GatewayConfig) -> Iterator[TestClient]:
        for client in build_test_client(config):
            yield client

    yield _start
    reset_sandbox_credential_cache()


def _hosted(config: GatewayConfig) -> GatewayConfig:
    return config.model_copy(update={"sandbox_provider": SandboxProvider.E2B})


def _credential(client: TestClient, headers: dict[str, str]) -> Any:
    return client.get(_SETTINGS, headers=headers).json()["sandbox_credential"]


def test_a_stored_key_wins_over_the_environment_and_clearing_it_falls_back(
    start: Start,
    test_config: GatewayConfig,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with start(_hosted(test_config)) as client:
        assert _credential(client, master_key_header) == {"source": None, "last4": None, "env_var": "E2B_API_KEY"}
        assert sandbox_api_key(SandboxProvider.E2B) is None

        monkeypatch.setenv("E2B_API_KEY", "e2b_from_env_aaaa")
        assert _credential(client, master_key_header)["source"] == "environment"

        response = client.put(_CREDENTIAL, json={"api_key": " e2b_rotated_bbbb "}, headers=master_key_header)
        assert response.status_code == status.HTTP_200_OK, response.text
        assert response.json() == {"source": "stored", "last4": "bbbb", "env_var": "E2B_API_KEY"}
        assert sandbox_api_key(SandboxProvider.E2B) == "e2b_rotated_bbbb"

        session = db_session_factory()
        try:
            row = session.get(SandboxCredential, "e2b")
            assert row is not None
            assert "e2b_rotated_bbbb" not in row.encrypted_api_key
        finally:
            session.close()

        response = client.delete(_CREDENTIAL, headers=master_key_header)
        assert response.status_code == status.HTTP_200_OK, response.text
        assert response.json()["source"] == "environment"
        assert sandbox_api_key(SandboxProvider.E2B) == "e2b_from_env_aaaa"


def test_a_worker_that_never_saw_the_write_loads_the_stored_key(
    start: Start, test_config: GatewayConfig, master_key_header: dict[str, str]
) -> None:
    with start(_hosted(test_config)) as client:
        assert client.put(_CREDENTIAL, json={"api_key": "e2b_shared_cccc"}, headers=master_key_header).is_success

    reset_sandbox_credential_cache()
    assert sandbox_api_key(SandboxProvider.E2B) is None
    with start(_hosted(test_config)) as client:
        assert _credential(client, master_key_header) == {
            "source": "stored",
            "last4": "cccc",
            "env_var": "E2B_API_KEY",
        }
        assert sandbox_api_key(SandboxProvider.E2B) == "e2b_shared_cccc"


def test_the_protocol_provider_takes_no_key(
    start: Start, test_config: GatewayConfig, master_key_header: dict[str, str]
) -> None:
    config = test_config.model_copy(update={"sandbox_url": "http://sandbox:8080"})
    with start(config) as client:
        assert _credential(client, master_key_header) is None
        response = client.put(_CREDENTIAL, json={"api_key": "e2b_dddd"}, headers=master_key_header)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "leases no sandboxes" in response.json()["detail"]


def test_a_blank_key_is_refused(start: Start, test_config: GatewayConfig, master_key_header: dict[str, str]) -> None:
    with start(_hosted(test_config)) as client:
        response = client.put(_CREDENTIAL, json={"api_key": "   "}, headers=master_key_header)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert sandbox_api_key(SandboxProvider.E2B) is None
