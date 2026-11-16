"""``POST /api/v1/code/execute``: running code in the sandbox without a model.

The port is faked at the composition root, so the real admission, backend,
container registry, budget and usage row run, and only the sandbox itself is
stood in for: what it was asked to run, and how long it was given, is recorded,
and what it answers is set per test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from gateway.adapters.file_storage_adapter import LocalDirFileStore
from gateway.core.config import (
    API_KEY_HEADER,
    API_ROOT,
    COST_USD_HEADER,
    PRICING_SOURCE_HEADER,
    GatewayConfig,
    SandboxEnvironmentConfig,
)
from gateway.core.error_codes import ERROR_CODE_HEADER, INSUFFICIENT_FUNDS
from gateway.ports.billing_port import InsufficientFundsError
from gateway.ports.code_execution_port import SandboxFileEntry, SandboxNotReachableError, SandboxSessionGoneError
from gateway.services.code_execution import CONTAINER_ID_PREFIX
from gateway.types.code_execution import CodeExecutionResult, ResultBlock

_SANDBOX_URL = "http://127.0.0.1:9999/sandbox"
_ENDPOINT = f"{API_ROOT}/code/execute"
_USAGE_LABEL = "/v1/code/execute"


class _FakeSession:
    def __init__(self, port: _FakePort, session_id: str) -> None:
        self._port = port
        self.session_id = session_id
        self.holds_across_requests = True

    def discard(self) -> None:
        self.holds_across_requests = False

    async def execute(self, code: str, *, timeout_s: float) -> ResultBlock:
        self._port.executed.append({"code": code, "timeout_s": timeout_s, "session": self.session_id})
        if self._port.unreachable:
            raise SandboxNotReachableError("connect failed")
        self._port.workspace.update(self._port.writes)
        return ResultBlock(type="code_execution_tool_result", content=self._port.result)

    async def put_file(self, path: str, data: bytes, *, mime_type: str) -> None:
        self._port.seeded[path] = data
        self._port.workspace[path] = data

    async def list_files(self) -> list[SandboxFileEntry]:
        return [SandboxFileEntry(path=path, size_bytes=len(data)) for path, data in self._port.workspace.items()]

    async def read_file(self, path: str, *, budget_bytes: int) -> AsyncIterator[bytes]:
        yield self._port.workspace[path]


class _FakePort:
    label = "fake-provider"

    def __init__(self) -> None:
        self.opened: list[dict[str, Any]] = []
        self.executed: list[dict[str, Any]] = []
        self.gone: set[str] = set()
        self.unreachable = False
        # The provider down before a session opens, as against one that fails a run.
        self.unreachable_on_open = False
        self.result = CodeExecutionResult(stdout="2\n", stderr="", return_code=0)
        # What uploads were put in the working directory, what is in it, and what the next run writes there.
        self.seeded: dict[str, bytes] = {}
        self.workspace: dict[str, bytes] = {}
        self.writes: dict[str, bytes] = {}

    @asynccontextmanager
    async def open_session(
        self,
        *,
        image: str | None = None,
        timeout_s: float,
        session_ttl_s: float,
        resume: str | None = None,
        keep_alive_s: float | None = None,
    ) -> AsyncIterator[_FakeSession]:
        del session_ttl_s
        self.opened.append({"image": image, "timeout_s": timeout_s, "resume": resume, "keep_alive_s": keep_alive_s})
        if resume is not None and resume in self.gone:
            raise SandboxSessionGoneError(f"{resume} is gone")
        if self.unreachable_on_open:
            raise SandboxNotReachableError("connect failed")
        yield _FakeSession(self, resume or f"sbx-{len(self.opened)}")


@pytest.fixture
def port(monkeypatch: pytest.MonkeyPatch) -> Generator[_FakePort]:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    fake = _FakePort()
    with patch("gateway.container.build_code_execution_port", new=lambda _config, **_kwargs: fake):
        yield fake


@pytest.fixture
def tmp_file_store(client: TestClient, tmp_path: Path) -> None:
    """Point the app's blob store at a temp dir (the default writes to cwd)."""
    cast(Any, client.app).state.file_store = LocalDirFileStore(str(tmp_path))


def _rows(client: TestClient, master_key_header: dict[str, str], user_id: str) -> list[dict[str, Any]]:
    resp = client.get(
        f"{API_ROOT}/usage", params={"user_id": user_id, "endpoint": _USAGE_LABEL}, headers=master_key_header
    )
    assert resp.status_code == 200, resp.text
    return [dict(row) for row in resp.json()]


def _price(
    client: TestClient, master_key_header: dict[str, str], per_million: float, key: str = "otari:code_execution"
) -> None:
    """A flat per-run rate: the stored rate is USD per million runs."""
    resp = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": key,
            "input_price_per_million": per_million,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )
    assert resp.status_code in (200, 201), resp.text


def _set_policy(client: TestClient, master_key_header: dict[str, str], **policy: Any) -> None:
    workspace_id = client.get(f"{API_ROOT}/workspaces", headers=master_key_header).json()["data"][0]["id"]
    resp = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/code-execution-policy", json=policy, headers=master_key_header
    )
    assert resp.status_code == 200, resp.text


def test_runs_the_code_as_sent_and_bills_the_run(
    client: TestClient,
    port: _FakePort,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    _price(client, master_key_header, 5000.0)
    code = "x = 1 + 1\nprint( x )  # spacing kept\n"

    resp = client.post(_ENDPOINT, json={"code": code}, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "object": "code_execution.result",
        "stdout": "2\n",
        "stderr": "",
        "return_code": 0,
        "files": [],
    }
    assert [run["code"] for run in port.executed] == [code], "the code reaches the sandbox byte for byte"
    assert "Otari-Container-Id" not in resp.headers, "a run that asked for no container holds none"
    assert (resp.headers[COST_USD_HEADER], resp.headers[PRICING_SOURCE_HEADER]) == ("0.005000", "deployment")
    assert port.opened[0]["keep_alive_s"] is None
    (row,) = _rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["model"], row["provider"], row["status"]) == ("code_execution", "otari", "success")
    assert row["cost"] == pytest.approx(0.005)


def test_code_that_fails_is_still_billed(
    client: TestClient,
    port: _FakePort,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    _price(client, master_key_header, 5000.0)
    port.result = CodeExecutionResult(stdout="", stderr="ZeroDivisionError", return_code=1)

    resp = client.post(_ENDPOINT, json={"code": "1 / 0"}, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    assert (resp.json()["stderr"], resp.json()["return_code"]) == ("ZeroDivisionError", 1)
    assert resp.headers[COST_USD_HEADER] == "0.005000"
    (row,) = _rows(client, master_key_header, api_key_obj["user_id"])
    assert row["status"] == "success"
    assert row["cost"] == pytest.approx(0.005), "the sandbox ran the code, whatever it did"


def test_uploads_are_put_in_the_sandbox_and_written_files_come_back(
    client: TestClient, port: _FakePort, api_key_header: dict[str, str], tmp_file_store: None
) -> None:
    upload = client.post(
        f"{API_ROOT}/files",
        headers=api_key_header,
        files={"file": ("data.csv", b"a,b\n1,2\n", "text/csv")},
        data={"purpose": "user_data"},
    )
    assert upload.status_code == 200, upload.text
    port.writes = {"chart.png": b"png-bytes"}

    resp = client.post(
        _ENDPOINT, json={"code": "make_chart()", "file_ids": [upload.json()["id"]]}, headers=api_key_header
    )

    assert resp.status_code == 200, resp.text
    assert port.seeded == {"data.csv": b"a,b\n1,2\n"}
    (written,) = resp.json()["files"]
    assert written["filename"] == "chart.png", "the upload is not reported as something the run wrote"
    content = client.get(f"{API_ROOT}/files/{written['file_id']}/content", headers=api_key_header)
    assert content.status_code == 200, content.text
    assert content.content == b"png-bytes"


def test_an_upload_the_caller_cannot_reach_is_a_400(
    client: TestClient, port: _FakePort, api_key_header: dict[str, str], tmp_file_store: None
) -> None:
    resp = client.post(_ENDPOINT, json={"code": "print(1)", "file_ids": ["file-nope"]}, headers=api_key_header)

    assert resp.status_code == 400, resp.text
    assert "file-nope" in resp.json()["detail"]
    assert port.opened == []


def test_auto_holds_the_sandbox_and_its_id_runs_there_again(
    client: TestClient, port: _FakePort, api_key_header: dict[str, str]
) -> None:
    first = client.post(_ENDPOINT, json={"code": "x = 2", "container": "auto"}, headers=api_key_header)

    assert first.status_code == 200, first.text
    assert COST_USD_HEADER not in first.headers, "an unpriced run names no cost"
    assert PRICING_SOURCE_HEADER not in first.headers
    container_id = first.headers["Otari-Container-Id"]
    assert container_id.startswith(CONTAINER_ID_PREFIX)
    assert first.headers["Otari-Container-Expires-At"]

    second = client.post(_ENDPOINT, json={"code": "print(x)", "container": container_id}, headers=api_key_header)

    assert second.status_code == 200, second.text
    assert second.headers["Otari-Container-Id"] == container_id
    assert port.opened[1]["resume"] == "sbx-1", "the second run resumed the first one's sandbox"


def test_an_unknown_container_is_refused_before_any_sandbox_is_leased(
    client: TestClient, port: _FakePort, api_key_header: dict[str, str]
) -> None:
    resp = client.post(
        _ENDPOINT, json={"code": "print(1)", "container": f"{CONTAINER_ID_PREFIX}nope"}, headers=api_key_header
    )

    assert resp.status_code == 400, resp.text
    assert "has expired or does not exist" in resp.json()["detail"]
    assert port.opened == []


def test_a_container_the_provider_lost_is_a_400_and_is_forgotten(
    client: TestClient, port: _FakePort, api_key_header: dict[str, str]
) -> None:
    first = client.post(_ENDPOINT, json={"code": "x = 2", "container": "auto"}, headers=api_key_header)
    container_id = first.headers["Otari-Container-Id"]
    port.gone.add("sbx-1")

    lost = client.post(_ENDPOINT, json={"code": "print(x)", "container": container_id}, headers=api_key_header)
    again = client.post(_ENDPOINT, json={"code": "print(x)", "container": container_id}, headers=api_key_header)

    assert lost.status_code == 400, lost.text
    assert container_id in lost.json()["detail"]
    assert again.status_code == 400, again.text
    assert len(port.opened) == 2, "the forgotten id is refused at admission, without asking the provider again"


def test_a_workspace_with_code_execution_off_is_refused(
    client: TestClient,
    port: _FakePort,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    _set_policy(client, master_key_header, enabled=False)

    resp = client.post(_ENDPOINT, json={"code": "print(1)"}, headers=api_key_header)

    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "code execution is not enabled for this workspace"
    assert port.opened == []
    (row,) = _rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["status"], row["status_code"]) == ("error", 403)


def test_the_workspace_ceiling_and_the_request_narrow_the_timeout(
    client: TestClient, port: _FakePort, api_key_header: dict[str, str], master_key_header: dict[str, str]
) -> None:
    _set_policy(client, master_key_header, enabled=True, exec_timeout_s=20)

    client.post(_ENDPOINT, json={"code": "print(1)"}, headers=api_key_header)
    client.post(_ENDPOINT, json={"code": "print(1)", "timeout_seconds": 5}, headers=api_key_header)
    client.post(_ENDPOINT, json={"code": "print(1)", "timeout_seconds": 600}, headers=api_key_header)

    assert [run["timeout_s"] for run in port.executed] == [20.0, 5.0, 20.0]


def test_no_sandbox_is_a_400(client: TestClient, api_key_header: dict[str, str]) -> None:
    resp = client.post(_ENDPOINT, json={"code": "print(1)"}, headers=api_key_header)

    assert resp.status_code == 400, resp.text
    assert "no sandbox is configured" in resp.json()["detail"]


def test_an_unpriced_run_is_refused_where_pricing_is_required(
    client: TestClient,
    port: _FakePort,
    test_config: GatewayConfig,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(test_config, "require_pricing", True)

    resp = client.post(_ENDPOINT, json={"code": "print(1)"}, headers=api_key_header)

    assert resp.status_code == 402, resp.text
    assert "otari:code_execution" in resp.json()["detail"]
    assert COST_USD_HEADER not in resp.headers, "a refused run was not charged"
    assert port.opened == []


def test_a_run_without_funds_is_refused_before_the_sandbox_with_no_cost(
    client: TestClient, port: _FakePort, api_key_header: dict[str, str], master_key_header: dict[str, str]
) -> None:
    _price(client, master_key_header, 5000.0)
    with patch(
        "gateway.api.routes.code_execution.admit_deployment_paid",
        new=AsyncMock(side_effect=InsufficientFundsError(Decimal(0))),
    ):
        resp = client.post(_ENDPOINT, json={"code": "print(1)"}, headers=api_key_header)

    assert resp.status_code == 402, resp.text
    assert resp.headers[ERROR_CODE_HEADER] == INSUFFICIENT_FUNDS
    assert COST_USD_HEADER not in resp.headers
    assert PRICING_SOURCE_HEADER not in resp.headers
    assert port.opened == []


def test_an_unreachable_sandbox_is_a_502_and_refunds_the_hold(
    client: TestClient,
    port: _FakePort,
    master_key_header: dict[str, str],
    db_session_factory: Any,
) -> None:
    _price(client, master_key_header, 5000.0)
    created = client.post(f"{API_ROOT}/users", json={"user_id": "sandbox-down"}, headers=master_key_header)
    assert created.status_code == 200, created.text
    port.unreachable = True

    resp = client.post(_ENDPOINT, json={"code": "print(1)", "user": "sandbox-down"}, headers=master_key_header)

    assert resp.status_code == 502, resp.text
    assert "connect failed" not in resp.text
    assert COST_USD_HEADER not in resp.headers, "a run the sandbox never took is refunded, so it names no cost"
    (row,) = _rows(client, master_key_header, "sandbox-down")
    assert (row["status"], row["status_code"]) == ("error", 502)
    session = db_session_factory()
    try:
        reserved = session.execute(
            text("SELECT reserved FROM users WHERE user_id = :user"), {"user": "sandbox-down"}
        ).scalar_one()
    finally:
        session.close()
    assert float(reserved) == 0.0


def test_the_master_key_must_name_a_user(
    client: TestClient, port: _FakePort, master_key_header: dict[str, str]
) -> None:
    resp = client.post(_ENDPOINT, json={"code": "print(1)"}, headers=master_key_header)

    assert resp.status_code == 400, resp.text
    assert port.opened == []


def test_empty_code_is_a_422(client: TestClient, api_key_header: dict[str, str]) -> None:
    resp = client.post(_ENDPOINT, json={"code": ""}, headers=api_key_header)

    assert resp.status_code == 422, resp.text


def test_whitespace_only_code_is_a_422(client: TestClient, api_key_header: dict[str, str]) -> None:
    resp = client.post(_ENDPOINT, json={"code": "  \n\t "}, headers=api_key_header)

    assert resp.status_code == 422, resp.text


def test_a_resume_the_sandbox_cannot_take_gives_its_claim_back(
    client: TestClient, port: _FakePort, api_key_header: dict[str, str]
) -> None:
    """A session that never opened leaves no claim behind, so the container is not busy once the provider is back."""
    first = client.post(_ENDPOINT, json={"code": "x = 2", "container": "auto"}, headers=api_key_header)
    container_id = first.headers["Otari-Container-Id"]
    body = {"code": "print(x)", "container": container_id}

    port.unreachable_on_open = True
    down = client.post(_ENDPOINT, json=body, headers=api_key_header)
    port.unreachable_on_open = False
    back = client.post(_ENDPOINT, json=body, headers=api_key_header)

    assert down.status_code == 502, down.text
    assert back.status_code == 200, back.text
    assert back.headers["Otari-Container-Id"] == container_id


def test_runs_are_counted_by_the_usage_views_tool_filter_and_breakdown(
    client: TestClient,
    port: _FakePort,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    _price(client, master_key_header, 5000.0)
    client.post(_ENDPOINT, json={"code": "print(1)"}, headers=api_key_header)
    port.unreachable = True
    client.post(_ENDPOINT, json={"code": "print(1)"}, headers=api_key_header)
    user_id = api_key_obj["user_id"]

    filtered = client.get(
        f"{API_ROOT}/usage", params={"user_id": user_id, "tool": "code_execution"}, headers=master_key_header
    )
    summary = client.get(
        f"{API_ROOT}/usage/summary", params={"dimensions": "tool", "user_id": user_id}, headers=master_key_header
    )

    assert filtered.status_code == 200, filtered.text
    assert len(filtered.json()) == 2
    assert summary.status_code == 200, summary.text
    (row,) = summary.json()["by_tool"]
    assert (row["tool"], row["calls"], row["errors"]) == ("code_execution", 1, 1)
    assert row["cost"] == pytest.approx(0.005)


def test_a_service_key_bills_the_end_user_it_names(
    client: TestClient, port: _FakePort, master_key_header: dict[str, str]
) -> None:
    """``user`` on a service key's run names an end user, created with the key's end-user budget."""
    budget = client.post(
        f"{API_ROOT}/budgets", json={"request_limit": 1, "budget_duration_sec": 86400}, headers=master_key_header
    ).json()
    key = client.post(
        f"{API_ROOT}/keys",
        json={
            "key_name": "code-service",
            "user_id": "code-service",
            "is_service_key": True,
            "end_user_budget_id": budget["budget_id"],
        },
        headers=master_key_header,
    ).json()
    headers = {API_KEY_HEADER: f"Bearer {key['key']}"}

    first = client.post(_ENDPOINT, json={"code": "print(1)", "user": "fxa-123"}, headers=headers)
    second = client.post(_ENDPOINT, json={"code": "print(1)", "user": "fxa-123"}, headers=headers)
    other = client.post(_ENDPOINT, json={"code": "print(1)", "user": "fxa-456"}, headers=headers)

    assert first.status_code == 200, first.text
    assert second.status_code == 403, second.text
    assert other.status_code == 200, other.text
    users = client.get(f"{API_ROOT}/users", params={"parent_user_id": "code-service"}, headers=master_key_header).json()
    end_user = next(user for user in users if user["external_id"] == "fxa-123")
    assert end_user["budget_id"] == budget["budget_id"]
    (row,) = _rows(client, master_key_header, end_user["user_id"])
    assert row["status"] == "success"
    assert _rows(client, master_key_header, "code-service") == [], "the key's owner is billed for nothing"


def test_a_budget_on_a_run_naming_no_end_user_is_refused(
    client: TestClient, port: _FakePort, api_key_header: dict[str, str]
) -> None:
    headers = {**api_key_header, "Otari-End-User-Budget": "any-budget"}

    refused = client.post(_ENDPOINT, json={"code": "print(1)"}, headers=headers)

    assert refused.status_code == 400, refused.text
    assert refused.headers["Otari-Error-Code"] == "end_user_budget_without_end_user"
    assert port.executed == []


_ENVIRONMENTS = f"{API_ROOT}/code/environments"


@pytest.fixture
def environments(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Offer two environments, ``python`` the deployment's default, ``python-data`` described by the operator."""
    monkeypatch.setattr(test_config, "sandbox_session_image", "python")
    monkeypatch.setattr(test_config, "sandbox_allowed_session_images", "python-data")
    monkeypatch.setattr(
        test_config,
        "sandbox_environments",
        {"python-data": SandboxEnvironmentConfig(label="Python data", description="pandas and numpy, 2 vCPU")},
    )


def test_a_named_environment_runs_in_its_image_at_its_own_price(
    client: TestClient,
    port: _FakePort,
    environments: None,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    _price(client, master_key_header, 5000.0)
    _price(client, master_key_header, 9000.0, key="otari:code_execution:python-data")

    named = client.post(_ENDPOINT, json={"code": "print(1)", "environment": "python-data"}, headers=api_key_header)
    unnamed = client.post(_ENDPOINT, json={"code": "print(1)"}, headers=api_key_header)

    assert named.status_code == 200, named.text
    assert [opened["image"] for opened in port.opened] == ["python-data", "python"]
    assert named.headers[COST_USD_HEADER] == "0.009000"
    assert unnamed.headers[COST_USD_HEADER] == "0.005000", "an environment with no row of its own takes the tool's"
    costs = sorted(row["cost"] for row in _rows(client, master_key_header, api_key_obj["user_id"]))
    assert costs == [pytest.approx(0.005), pytest.approx(0.009)]


def test_an_environment_the_gateway_does_not_offer_is_a_400(
    client: TestClient, port: _FakePort, environments: None, api_key_header: dict[str, str]
) -> None:
    resp = client.post(_ENDPOINT, json={"code": "print(1)", "environment": "ruby"}, headers=api_key_header)

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == (
        f"Environment 'ruby' is not offered on this gateway. GET {API_ROOT}/code/environments lists those it offers."
    )
    assert port.opened == []


def test_an_environment_other_than_the_workspaces_pin_is_refused(
    client: TestClient,
    port: _FakePort,
    environments: None,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
) -> None:
    _set_policy(client, master_key_header, enabled=True, image="python-data")

    refused = client.post(_ENDPOINT, json={"code": "print(1)", "environment": "python"}, headers=api_key_header)
    pinned = client.post(_ENDPOINT, json={"code": "print(1)"}, headers=api_key_header)

    assert refused.status_code == 403, refused.text
    assert "'python-data' only" in refused.json()["detail"]
    assert pinned.status_code == 200, pinned.text
    assert [opened["image"] for opened in port.opened] == ["python-data"]


def test_a_held_container_keeps_its_environment_and_refuses_another(
    client: TestClient,
    port: _FakePort,
    environments: None,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
) -> None:
    _price(client, master_key_header, 5000.0)
    _price(client, master_key_header, 9000.0, key="otari:code_execution:python-data")
    first = client.post(
        _ENDPOINT, json={"code": "x = 2", "container": "auto", "environment": "python-data"}, headers=api_key_header
    )
    container_id = first.headers["Otari-Container-Id"]

    mismatch = client.post(
        _ENDPOINT, json={"code": "print(x)", "container": container_id, "environment": "python"}, headers=api_key_header
    )
    resumed = client.post(_ENDPOINT, json={"code": "print(x)", "container": container_id}, headers=api_key_header)

    assert mismatch.status_code == 400, mismatch.text
    assert mismatch.json()["detail"].startswith(
        f"Container '{container_id}' runs in the environment 'python-data', not 'python'."
    )
    assert resumed.status_code == 200, "the refusal gave the container back"
    assert resumed.headers[COST_USD_HEADER] == "0.009000", "a resumed sandbox is priced as the environment it runs"
    assert len(port.opened) == 2


def test_a_held_container_with_no_recorded_environment_refuses_a_named_one(
    client: TestClient,
    port: _FakePort,
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
) -> None:
    """A sandbox started with no image runs in a template the gateway cannot name, so it cannot be priced as one."""
    _price(client, master_key_header, 5000.0)
    _price(client, master_key_header, 9000.0, key="otari:code_execution:python-data")
    first = client.post(_ENDPOINT, json={"code": "x = 2", "container": "auto"}, headers=api_key_header)
    container_id = first.headers["Otari-Container-Id"]
    monkeypatch.setattr(test_config, "sandbox_allowed_session_images", "python-data")

    named = client.post(
        _ENDPOINT,
        json={"code": "print(x)", "container": container_id, "environment": "python-data"},
        headers=api_key_header,
    )
    resumed = client.post(_ENDPOINT, json={"code": "print(x)", "container": container_id}, headers=api_key_header)

    assert named.status_code == 400, named.text
    assert named.json()["detail"].startswith(
        f"Container '{container_id}' was created without a recorded environment, so it cannot run as 'python-data'."
    )
    assert "None" not in named.json()["detail"]
    assert resumed.status_code == 200, "the refusal gave the container back"
    assert resumed.headers[COST_USD_HEADER] == "0.005000"
    assert [opened["image"] for opened in port.opened] == [None, None]


def test_the_environment_listing_describes_and_prices_each_offered_environment(
    client: TestClient,
    port: _FakePort,
    environments: None,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
) -> None:
    _price(client, master_key_header, 9000.0, key="otari:code_execution:python-data")

    resp = client.get(_ENVIRONMENTS, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "object": "list",
        "data": [
            {
                "object": "code_execution.environment",
                "name": "python",
                "label": "python",
                "description": None,
                "price_per_run": None,
                "default": True,
            },
            {
                "object": "code_execution.environment",
                "name": "python-data",
                "label": "Python data",
                "description": "pandas and numpy, 2 vCPU",
                "price_per_run": pytest.approx(0.009),
                "default": False,
            },
        ],
    }


def test_the_listing_falls_back_to_the_tools_price(
    client: TestClient,
    port: _FakePort,
    environments: None,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
) -> None:
    _price(client, master_key_header, 5000.0)

    data = client.get(_ENVIRONMENTS, headers=api_key_header).json()["data"]

    assert [env["price_per_run"] for env in data] == [pytest.approx(0.005), pytest.approx(0.005)]


def test_the_listing_follows_the_workspaces_pin_and_its_switch(
    client: TestClient,
    port: _FakePort,
    environments: None,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
) -> None:
    _set_policy(client, master_key_header, enabled=True, image="python-data")
    pinned = client.get(_ENVIRONMENTS, headers=api_key_header).json()["data"]
    _set_policy(client, master_key_header, enabled=False)
    off = client.get(_ENVIRONMENTS, headers=api_key_header).json()["data"]

    assert [(env["name"], env["default"]) for env in pinned] == [("python-data", True)]
    assert off == []


def test_a_master_key_lists_the_deployments_environments_whatever_a_workspace_sets(
    client: TestClient,
    port: _FakePort,
    environments: None,
    master_key_header: dict[str, str],
) -> None:
    _set_policy(client, master_key_header, enabled=False)

    data = client.get(_ENVIRONMENTS, headers=master_key_header).json()["data"]

    assert [(env["name"], env["default"]) for env in data] == [("python", True), ("python-data", False)]


def test_a_gateway_with_no_sandbox_lists_no_environment(
    client: TestClient, environments: None, api_key_header: dict[str, str]
) -> None:
    resp = client.get(_ENVIRONMENTS, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"object": "list", "data": []}


def test_the_listing_needs_a_key(client: TestClient, port: _FakePort) -> None:
    assert client.get(_ENVIRONMENTS).status_code == 401


def test_the_listing_pages_in_the_operators_order(
    client: TestClient, port: _FakePort, environments: None, api_key_header: dict[str, str]
) -> None:
    first = client.get(_ENVIRONMENTS, params={"limit": 1}, headers=api_key_header).json()["data"]
    second = client.get(_ENVIRONMENTS, params={"skip": 1, "limit": 1}, headers=api_key_header).json()["data"]
    past = client.get(_ENVIRONMENTS, params={"skip": 2}, headers=api_key_header).json()["data"]

    assert [env["name"] for env in first] == ["python"]
    assert [(env["name"], env["default"]) for env in second] == [("python-data", False)]
    assert past == []


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 1001}, {"skip": -1}])
def test_the_listing_refuses_a_page_out_of_range(
    client: TestClient, port: _FakePort, environments: None, api_key_header: dict[str, str], params: dict[str, int]
) -> None:
    assert client.get(_ENVIRONMENTS, params=params, headers=api_key_header).status_code == 422
