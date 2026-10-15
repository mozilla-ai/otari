"""What crosses ``UsageSignalPort`` when a running app settles a request.

The port is bound through ``OTARI_BOOTSTRAP`` the way an overlay binds it, and
the assertions are on what the seam carried, against the usage row the same
request wrote: the core adapter is a Null Object, so a port that was never
reached would otherwise fail silently.
"""

import sys
import time
from collections.abc import Callable, Generator
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletion
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.usage import PLAYGROUND_USAGE_ENDPOINT, UsageLog
from gateway.ports.usage_signal_port import RecordedUsage

from .conftest import MODEL_NAME, build_test_client
from .test_playground import _build_world, _mock_completion
from .test_playground import _chat as _playground_chat

MODULE = "probe_usage_signal"
PROBE_BOOTSTRAP = """
from gateway.ports.usage_signal_port import UsageSignalPort

RECORDED = []


class RecordingUsageSignal:
    async def usage_recorded(self, usage):
        RECORDED.append(usage)


def register(container):
    container.bind(UsageSignalPort, lambda session: RecordingUsageSignal())
"""
MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
# 10 prompt tokens at $2.50 per million plus 5 completion tokens at $10 per million.
EXPECTED_COST = Decimal("0.000075")


@pytest.fixture
def signaled_client(
    postgres_url: str, clean_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    (tmp_path / f"{MODULE}.py").write_text(PROBE_BOOTSTRAP)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(MODULE, None)
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        bootstrap=f"{MODULE}:register",
    )
    try:
        yield from build_test_client(config)
    finally:
        sys.modules.pop(MODULE, None)


def _recorded() -> list[RecordedUsage]:
    recorded: list[RecordedUsage] = sys.modules[MODULE].RECORDED
    return recorded


def _price(client: TestClient) -> None:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": MODEL_NAME, "input_price_per_million": 2.5, "output_price_per_million": 10.0},
        headers=MASTER,
    )
    assert response.status_code == status.HTTP_200_OK, response.text


def _settled(
    make_session: Callable[[], Session], endpoint: str, *, timeout: float = 3.0
) -> tuple[UsageLog, list[RecordedUsage]]:
    """The request's usage row and what the port was told, polled in case both land after the response."""
    deadline = time.monotonic() + timeout
    while True:
        with make_session() as db:
            row = db.execute(select(UsageLog).where(UsageLog.endpoint == endpoint)).scalar_one_or_none()
            if row is not None and _recorded():
                db.expunge(row)
                return row, _recorded()
        assert time.monotonic() < deadline, "the usage row was never written or never signaled"
        time.sleep(0.1)


async def _acompletion(**_kwargs: Any) -> ChatCompletion:
    return _mock_completion()


def test_a_chat_completion_tells_the_port_about_its_row(
    signaled_client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    _price(signaled_client)
    key = signaled_client.post(f"{API_ROOT}/keys", json={"key_name": "signal"}, headers=MASTER)
    assert key.status_code == status.HTTP_200_OK, key.text

    with patch("gateway.api.routes.chat.acompletion", side_effect=_acompletion):
        response = signaled_client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]},
            headers={API_KEY_HEADER: f"Bearer {key.json()['key']}"},
        )
    assert response.status_code == status.HTTP_200_OK, response.text

    row, recorded = _settled(db_session_factory, "/v1/chat/completions")
    [usage] = recorded
    assert usage.usage_id == row.id
    assert usage.workspace_id == row.workspace_id
    assert usage.api_key_id == key.json()["id"]
    assert usage.model == row.model
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (10, 5, 15)
    assert usage.cost == row.cost == EXPECTED_COST
    assert usage.status == "success"
    assert usage.integration_traffic is True
    # Served on the deployment's configured key, which no ``ModelProviderPort`` handed out.
    assert usage.deployment_paid is row.deployment_paid is False


def test_a_playground_completion_is_not_integration_traffic(
    signaled_client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    _price(signaled_client)
    world = _build_world(signaled_client, MASTER, db_session_factory)

    code, body = _playground_chat(signaled_client, world, "member")
    assert code == status.HTTP_200_OK, body

    row, recorded = _settled(db_session_factory, PLAYGROUND_USAGE_ENDPOINT)
    [usage] = recorded
    assert usage.usage_id == row.id
    assert usage.workspace_id == world.workspaces["alpha_one"]
    assert usage.endpoint == PLAYGROUND_USAGE_ENDPOINT
    assert usage.integration_traffic is False
