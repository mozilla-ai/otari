"""The key backend, as an admin and a reader meet it: capture needs one that works, and a read
that cannot reach it says so without saying why."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.adapters.data_key_adapter import SecretBoxDataKeys
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.traces import TraceSpanContent
from gateway.ports.data_key_port import DataKeyContext
from gateway.services.secret_box import generate_secret_key

from .test_trace_content import _send, _settings, _wait_for, _workspace


@pytest.fixture
def test_config(test_config: GatewayConfig, tmp_path: Path) -> GatewayConfig:
    return test_config.model_copy(update={"trace_content_capture_max": "full", "files_local_dir": str(tmp_path)})


def test_capture_cannot_be_turned_on_without_a_key_backend_but_can_always_be_turned_off(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)
    workspace_id = _workspace(client, master_key_header, api_key_obj)
    url = f"{API_ROOT}/workspaces/{workspace_id}/trace-settings"

    refused = client.put(url, json={"content_capture": "tool_io"}, headers=master_key_header)

    assert refused.status_code == 409
    assert refused.json() == {"detail": "Content encryption is not configured on this deployment"}
    assert _settings(client, master_key_header, workspace_id, "off")["content_capture"] == "off"


def test_a_read_the_key_backend_cannot_serve_is_a_503_with_no_detail(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    workspace_id = _workspace(client, master_key_header, api_key_obj)
    _settings(client, master_key_header, workspace_id, "tool_io")
    _send(client, api_key_header)
    [row] = _wait_for(db_session_factory, TraceSpanContent, 1)

    async def unreachable(self: SecretBoxDataKeys, wrapped: bytes, context: DataKeyContext) -> bytes:
        raise ConnectionError("kms.eu-west-1.amazonaws.com unreachable")

    monkeypatch.setattr(SecretBoxDataKeys, "unwrap", unreachable)
    read = client.post(
        f"{API_ROOT}/traces/{row.trace_id}/spans/{row.span_id}/content/break-glass",
        json={"reason": "Integration test of the content path"},
        headers=master_key_header,
    )

    assert read.status_code == 503
    assert "amazonaws" not in read.text
