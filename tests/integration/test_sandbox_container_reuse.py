"""Resuming a code-execution sandbox across requests, end to end on the Messages route.

The port is faked at the composition root, so the real backend, registry and
loop run: a request that asks for a sandbox to outlive it is told which one it
got, the next request resumes it by that name, a request that asks for nothing
holds nothing, and every way an id can be wrong is answered the same way.
``amessages`` is faked to answer without a tool call, which is all the loop
needs to finish.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import patch

import pytest
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._tools import CODE_EXECUTION_HEADER
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.ports.code_execution_port import SandboxFileEntry, SandboxSessionGoneError
from gateway.repositories.code_execution import (
    SandboxContainerRow,
    claim_container_row,
    get_container_row,
    upsert_container_row,
)
from gateway.services.code_execution import CONTAINER_ID_PREFIX
from gateway.services.code_execution.container_sweeper import sweep_expired_containers

from .conftest import build_test_client

_SANDBOX_URL = "http://127.0.0.1:9999/sandbox"
_MODEL = "anthropic:claude-3-5-sonnet-20241022"


class _FakeSession:
    def __init__(self, session_id: str, *, holds_across_requests: bool = True) -> None:
        self.session_id = session_id
        self.holds_across_requests = holds_across_requests

    def discard(self) -> None:
        self.holds_across_requests = False

    async def execute(self, code: str, *, timeout_s: float) -> Any:
        raise AssertionError("the faked model never calls the tool")

    async def put_file(self, path: str, data: bytes, *, mime_type: str) -> None:
        return None

    async def list_files(self) -> list[SandboxFileEntry]:
        return []

    async def read_file(self, path: str, *, budget_bytes: int) -> AsyncIterator[bytes]:
        raise AssertionError("nothing is produced")
        yield b""  # pragma: no cover


class _FakePort:
    """Records every lease it was asked for; ``gone`` names sessions it no longer has."""

    label = "fake-provider"

    def __init__(self) -> None:
        self.opened: list[dict[str, Any]] = []
        self.gone: set[str] = set()
        # False makes every session one the provider declines to hold, which is
        # the backend that ignores the contract's idle-timeout hint.
        self.holds = True

    @asynccontextmanager
    async def open_session(
        self,
        *,
        image: str | None = None,
        timeout_s: float,
        session_ttl_s: float,
        auth_token: str | None = None,
        resume: str | None = None,
        keep_alive_s: float | None = None,
    ) -> AsyncIterator[_FakeSession]:
        del image, timeout_s, session_ttl_s, auth_token
        self.opened.append({"resume": resume, "keep_alive_s": keep_alive_s})
        if resume is not None and resume in self.gone:
            raise SandboxSessionGoneError(f"{resume} is gone")
        yield _FakeSession(resume or f"sbx-{len(self.opened)}", holds_across_requests=self.holds)


def _text_response() -> MessageResponse:
    return MessageResponse(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-3-5-sonnet-20241022",
        content=[TextBlock(type="text", text="done", citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(input_tokens=5, output_tokens=2),
    )


def _body(container: str | None = "auto", **extra: Any) -> dict[str, Any]:
    """A request asking to hold a sandbox by default, since that is what most of these test.

    ``container=None`` is the request that asks for nothing, which is every
    request written before reuse existed.
    """
    body: dict[str, Any] = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "compute"}],
        "max_tokens": 100,
        "tools": [{"type": "otari_code_execution"}],
        **extra,
    }
    if container is not None:
        body["container"] = container
    return body


def _post(client: TestClient, headers: dict[str, str], body: dict[str, Any], port: _FakePort) -> Any:
    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _text_response()

    with (
        patch("gateway.container.build_code_execution_port", new=lambda _config: port),
        patch("gateway.services.mcp_loop_messages.amessages", new=fake_amessages),
        # The route's own call, for a body with no tool loop to run.
        patch("gateway.api.routes.messages.amessages", new=fake_amessages),
    ):
        return client.post(f"{API_ROOT}/messages", json=body, headers=headers)


@pytest.fixture
def sandbox_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)


def test_a_response_names_the_sandbox_it_holds_and_the_next_request_resumes_it(
    client: TestClient, api_key_header: dict[str, str], sandbox_env: None
) -> None:
    port = _FakePort()

    first = _post(client, api_key_header, _body(), port)

    assert first.status_code == 200, first.text
    container = first.json()["container"]
    assert container["id"].startswith(CONTAINER_ID_PREFIX)
    assert container["expires_at"]
    # Every dialect gets the headers; Messages also gets Anthropic's own field.
    assert first.headers["X-Otari-Container-Id"] == container["id"]
    assert first.headers["X-Otari-Container-Expires-At"]
    assert port.opened == [{"resume": None, "keep_alive_s": 600.0}]

    second = _post(client, api_key_header, _body(container["id"]), port)

    assert second.status_code == 200, second.text
    assert second.json()["container"]["id"] == container["id"], "a resume keeps the container's name"
    assert port.opened[1]["resume"] == "sbx-1", "the provider's own session was resumed, not a new one leased"
    assert port.opened[1]["keep_alive_s"] == 600.0


def test_an_unknown_container_is_refused_in_the_words_clients_recover_from(
    client: TestClient, api_key_header: dict[str, str], sandbox_env: None
) -> None:
    port = _FakePort()

    response = _post(client, api_key_header, _body(f"{CONTAINER_ID_PREFIX}nope"), port)

    assert response.status_code == 400, response.text
    error = response.json()["detail"]["error"]
    assert error["type"] == "invalid_request_error"
    assert "has expired or does not exist" in error["message"]
    assert port.opened == [], "a refused request leases nothing"


def test_another_callers_container_is_an_unknown_one(
    client: TestClient, api_key_header: dict[str, str], master_key_header: dict[str, str], sandbox_env: None
) -> None:
    """Bound to the user that leased it: a resume by anyone else is answered as unknown, not as forbidden."""
    port = _FakePort()
    first = _post(client, api_key_header, _body(), port)
    assert first.status_code == 200, first.text
    container_id = first.json()["container"]["id"]

    created = client.post(f"{API_ROOT}/users", json={"user_id": "someone-else"}, headers=master_key_header)
    assert created.status_code == 200, created.text
    as_someone_else = _body(container_id, metadata={"user_id": "someone-else"})

    response = _post(client, master_key_header, as_someone_else, port)

    assert response.status_code == 400, response.text
    assert "has expired or does not exist" in response.json()["detail"]["error"]["message"]
    assert len(port.opened) == 1


def test_a_container_the_provider_no_longer_has_is_gone_and_forgotten(
    client: TestClient, api_key_header: dict[str, str], sandbox_env: None
) -> None:
    port = _FakePort()
    first = _post(client, api_key_header, _body(), port)
    assert first.status_code == 200, first.text
    container_id = first.json()["container"]["id"]
    port.gone.add("sbx-1")

    second = _post(client, api_key_header, _body(container_id), port)

    assert second.status_code == 400, second.text
    assert "has expired or does not exist" in second.json()["detail"]["error"]["message"]
    assert port.opened[1]["resume"] == "sbx-1", "the provider was asked, and said no"

    # The lease was dropped on the way out, so the next attempt does not ask again.
    third = _post(client, api_key_header, _body(container_id), port)
    assert third.status_code == 400, third.text
    assert len(port.opened) == 2


@pytest.mark.asyncio
async def test_a_second_request_cannot_run_in_a_sandbox_a_first_is_still_using(
    async_db: AsyncSession, client: TestClient, api_key_header: dict[str, str], sandbox_env: None
) -> None:
    """One request at a time per sandbox.

    Two sharing one workspace would interleave their code and each collect the
    other's produced files, so the second is told to retry rather than admitted.
    The claim is taken at admission and given back when the lease is recorded,
    so it is a live row rather than a lock held across the provider call.
    """
    port = _FakePort()
    first = _post(client, api_key_header, _body(), port)
    assert first.status_code == 200, first.text
    container_id = first.json()["container"]["id"]

    # Stand in for a request still running: the claim its admission took.
    now = datetime.now(UTC)
    uow = UnitOfWork(async_db)
    async with uow:
        assert await claim_container_row(uow, container_id, now=now, until=now + timedelta(minutes=5))

    response = _post(client, api_key_header, _body(container_id), port)

    assert response.status_code == 409, response.text
    error = response.json()["detail"]["error"]
    assert "in use by another request" in error["message"]
    assert len(port.opened) == 1, "a refused request leases nothing"


def test_a_request_that_asks_for_nothing_holds_nothing(
    client: TestClient, api_key_header: dict[str, str], sandbox_env: None
) -> None:
    """Holding a sandbox costs the deployment, so it is asked for rather than assumed.

    This is every request written before reuse existed, and it must behave the
    way it did then: a sandbox of its own, released with it.
    """
    port = _FakePort()

    response = _post(client, api_key_header, _body(container=None), port)

    assert response.status_code == 200, response.text
    assert response.json().get("container") is None
    assert "X-Otari-Container-Id" not in response.headers
    assert port.opened == [{"resume": None, "keep_alive_s": None}], "nothing was asked to be held"


def test_openais_own_auto_object_asks_for_the_same_thing(
    client: TestClient, api_key_header: dict[str, str], sandbox_env: None
) -> None:
    """``{"type": "auto"}`` is OpenAI's spelling of the ask, so it needs no gateway one."""
    port = _FakePort()
    body = _body(container=None)
    body["tools"] = [{"type": "otari_code_execution", "container": {"type": "auto"}}]

    response = _post(client, api_key_header, body, port)

    assert response.status_code == 200, response.text
    assert response.json()["container"]["id"].startswith(CONTAINER_ID_PREFIX)


_NATIVE_DECLARATION = {"type": "code_execution_20250825", "name": "code_execution"}
_MODEL_WITHOUT_NATIVE_CODE = "openai:gpt-4o-mini"


def test_a_stored_container_survives_a_model_swap_under_auto(
    client: TestClient, api_key_header: dict[str, str], sandbox_env: None
) -> None:
    """Naming a sandbox is already a statement about where the code runs.

    The flow a client holding a conversation writes: run code, keep the
    container, replay it next turn. Under ``auto`` the executor follows the
    model, so without the id pinning it a swap between turns hands the request
    to the provider and takes the sandbox, and its files, away.
    """
    port = _FakePort()
    first = _post(
        client,
        api_key_header,
        _body(model=_MODEL_WITHOUT_NATIVE_CODE, tools=[_NATIVE_DECLARATION]),
        port,
    )
    assert first.status_code == 200, first.text
    container_id = first.json()["container"]["id"]

    # The swap: this model runs the declaration natively, so ``auto`` alone
    # would leave it with the provider.
    second = _post(client, api_key_header, _body(container_id, tools=[_NATIVE_DECLARATION]), port)

    assert second.status_code == 200, second.text
    assert second.json()["container"]["id"] == container_id
    assert port.opened[1]["resume"] == "sbx-1", "the same sandbox, not a new one"


def test_the_same_swap_without_a_container_still_goes_to_the_provider(
    client: TestClient, api_key_header: dict[str, str], sandbox_env: None
) -> None:
    """The control for the test above: it is the id that pins, not the declaration."""
    port = _FakePort()

    response = _post(client, api_key_header, _body(container=None, tools=[_NATIVE_DECLARATION]), port)

    assert response.status_code == 200, response.text
    assert port.opened == [], "auto left the code with the provider, as it always has"


def test_an_explicit_executor_pin_still_beats_the_container(
    client: TestClient, api_key_header: dict[str, str], sandbox_env: None
) -> None:
    """The id is what the request asked for, so anything that outranks a request outranks it."""
    port = _FakePort()
    first = _post(
        client,
        api_key_header,
        _body(model=_MODEL_WITHOUT_NATIVE_CODE, tools=[_NATIVE_DECLARATION]),
        port,
    )
    assert first.status_code == 200, first.text
    container_id = first.json()["container"]["id"]

    pinned = {**api_key_header, CODE_EXECUTION_HEADER: "provider"}
    response = _post(client, pinned, _body(container_id, tools=[_NATIVE_DECLARATION]), port)

    assert response.status_code == 400, response.text
    assert "runs on the provider" in response.json()["detail"]["error"]["message"]


def test_a_gateway_container_is_refused_when_the_gateway_is_not_running_the_code(
    client: TestClient, api_key_header: dict[str, str], sandbox_env: None
) -> None:
    """The mirror of the id the sandbox path refuses.

    Forwarding a word the gateway told the client to send would buy them a
    provider error about it, which they cannot connect to anything. Reachable
    without a client doing anything differently, since which executor serves a
    request can change between turns.
    """
    port = _FakePort()
    no_code_execution = _body()
    del no_code_execution["tools"]

    response = _post(client, api_key_header, no_code_execution, port)

    assert response.status_code == 400, response.text
    message = response.json()["detail"]["error"]["message"]
    assert "runs on the provider" in message
    assert port.opened == []


def test_a_providers_own_container_is_left_alone(
    client: TestClient, api_key_header: dict[str, str], sandbox_env: None
) -> None:
    """Only the gateway's own words are refused; an id the provider minted is theirs."""
    port = _FakePort()
    theirs = _body("container_01ABCdef")
    del theirs["tools"]

    response = _post(client, api_key_header, theirs, port)

    assert response.status_code == 200, response.text
    assert port.opened == [], "no sandbox is involved either way"


def test_a_backend_that_will_not_hold_a_session_reports_no_container(
    client: TestClient, api_key_header: dict[str, str], sandbox_env: None
) -> None:
    """Asking is not getting. A session the adapter releases on exit is never named.

    A container id for a sandbox that is already gone would have every client
    that believes it spend a round trip to be told the id has expired.
    """
    port = _FakePort()
    port.holds = False

    response = _post(client, api_key_header, _body(), port)

    assert response.status_code == 200, response.text
    assert response.json().get("container") is None
    assert "X-Otari-Container-Id" not in response.headers


def test_an_oversized_container_id_is_not_echoed_back_whole(
    client: TestClient, api_key_header: dict[str, str], sandbox_env: None
) -> None:
    """The field is an unbounded string on the wire; the error body is not."""
    port = _FakePort()

    response = _post(client, api_key_header, _body(f"{CONTAINER_ID_PREFIX}{'A' * 100_000}"), port)

    assert response.status_code == 400, response.text
    message = response.json()["detail"]["error"]["message"]
    assert len(message) < 200
    assert "has expired or does not exist" in message


@pytest.fixture
def reuse_off_client(test_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    yield from build_test_client(test_config.model_copy(update={"sandbox_container_idle_ttl_sec": 0}))


def test_with_reuse_off_nothing_is_held_reported_or_resumable(
    reuse_off_client: TestClient, master_key_header: dict[str, str], sandbox_env: None
) -> None:
    """The pre-reuse behavior, byte for byte: a sandbox per request, released on exit.

    Also the difference between the two asks. ``auto`` is best-effort, so the
    caller who wanted a sandbox kept gets the run they would have had and no
    container; an id names specific files, so the same deployment refuses it
    rather than handing over an empty sandbox in their place.
    """
    key = reuse_off_client.post(f"{API_ROOT}/keys", json={"key_name": "k"}, headers=master_key_header)
    assert key.status_code == 200, key.text
    headers = {next(iter(master_key_header)): f"Bearer {key.json()['key']}"}
    port = _FakePort()

    response = _post(reuse_off_client, headers, _body(), port)

    assert response.status_code == 200, response.text
    assert "container" not in response.json()
    assert "X-Otari-Container-Id" not in response.headers
    assert port.opened == [{"resume": None, "keep_alive_s": None}]

    refused = _post(reuse_off_client, headers, _body(f"{CONTAINER_ID_PREFIX}anything"), port)
    assert refused.status_code == 400, refused.text
    assert "has expired or does not exist" in refused.json()["detail"]["error"]["message"]


@pytest.mark.asyncio
async def test_the_sweep_drops_rows_past_their_clocks(
    async_db: AsyncSession, client: TestClient, master_key_header: dict[str, str], test_user: dict[str, Any]
) -> None:
    workspaces = client.get(f"{API_ROOT}/workspaces", headers=master_key_header)
    assert workspaces.status_code == 200, workspaces.text
    workspace_id = uuid.UUID(workspaces.json()["data"][0]["id"])
    now = datetime.now(UTC)
    uow = UnitOfWork(async_db)

    def row(container_id: str, *, expires_at: datetime, hard_expires_at: datetime) -> SandboxContainerRow:
        return SandboxContainerRow(
            id=container_id,
            user_id=test_user["user_id"],
            workspace_id=workspace_id,
            provider="fake-provider",
            provider_session_id=f"sbx-{container_id[-1]}",
            created_at=now - timedelta(hours=2),
            last_used_at=now - timedelta(minutes=30),
            expires_at=expires_at,
            hard_expires_at=hard_expires_at,
        )

    soon, later, ago = now + timedelta(minutes=5), now + timedelta(hours=1), now - timedelta(minutes=1)
    async with uow:
        # Idle clock ran out; hard clock ran out; and one still live on both.
        await upsert_container_row(uow, row("otari_cntr_a", expires_at=ago, hard_expires_at=later))
        await upsert_container_row(uow, row("otari_cntr_b", expires_at=soon, hard_expires_at=ago))
        await upsert_container_row(uow, row("otari_cntr_c", expires_at=soon, hard_expires_at=later))

    async with uow:
        dropped = await sweep_expired_containers(uow, batch_size=10)

    assert dropped == 2
    async with uow:
        assert await get_container_row(uow, "otari_cntr_a") is None
        assert await get_container_row(uow, "otari_cntr_b") is None
        assert await get_container_row(uow, "otari_cntr_c") is not None
