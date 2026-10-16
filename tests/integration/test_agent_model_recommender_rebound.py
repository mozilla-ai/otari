"""A rebound ``AgentModelRecommenderPort`` is what the recommend route answers with, at the price it quotes.

The overlay hook is the whole point of the port: a hosted build binds a
recommender of its own, one the caller never configures, and the route, the
usage row, the budgets and the billing port follow it without an Otari file
changing. The probe bound here always picks ``haiku`` with no decision provider
configured, and quotes a price of its own; a recording billing probe bound
beside it shows that price held before the call and the final figure charged
after it. The probe's ledger holds what landed: it writes on the request
session as a port does, and an entry reaches the ledger when that session
commits and is dropped when it rolls back or closes. A decisions call on the
same app never reaches the billing probe, because a pass-through call quotes no
price.
"""

import itertools
import logging
import sys
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.log_config import logger as gateway_logger

from .conftest import build_test_client

MODULE = "probe_rebound_recommender"
PROBE_BOOTSTRAP = """
from decimal import Decimal

from gateway.container import Container
from gateway.ports.agent_model_recommender_port import (
    AgentModelRecommenderPort,
    ModelRecommendation,
    Quote,
    RecommendationFailedError,
    RecommendationUsage,
)
from gateway.ports.billing_port import BillingPort, InsufficientFundsError

ASKED = []
LEDGER = []  # What landed: a session's entries move here when it commits.
PENDING = {}  # Entries written on a session, by its id, until it commits, rolls back or closes.
RECORD = {"commits": False}
RELEASES = []  # Every release_hold call, landed or not.
SESSIONS = []
FUNDS = {"available": Decimal("1")}
FAULTS = set()


class ProbeRecommender:
    def quote(self):
        return Quote(provider="probe", model="recommender", charge=Decimal("0.002"))

    async def recommend(self, spawn, *, candidates):
        ASKED.append((spawn.agent_type, spawn.requested_model, sorted(candidates)))
        if "recommend" in FAULTS:
            raise RecommendationFailedError("the probe is down", upstream_status=503)
        charge = Decimal("0.005") if "overcharge" in FAULTS else Decimal("0.0015")
        return ModelRecommendation(
            model="haiku",
            reason="the probe always says haiku",
            probabilities=None,
            usage=RecommendationUsage(charge=charge),
        )


class ProbeBilling:
    def __init__(self, session):
        self.session = session
        SESSIONS.append(session)

    def write(self, entry):
        PENDING.setdefault(id(self.session), []).append(entry)

    async def apply_due_credit(self, *, organization_id):
        pass

    async def require_funds_on_deposit(self, *, organization_id):
        pass

    async def require_unheld_funds(self, *, organization_id):
        pass

    async def hold(self, *, organization_id, amount):
        if FUNDS["available"] < amount:
            raise InsufficientFundsError(FUNDS["available"])
        self.write(("hold", str(organization_id), amount))

    async def release_hold(self, *, organization_id, amount):
        RELEASES.append(amount)
        # A faulted release has written before it raises, as a port that fails midway would.
        self.write(("release", str(organization_id), amount))
        if "release" in FAULTS or ("release-once" in FAULTS and len(RELEASES) == 1):
            raise RuntimeError("ledger down")

    async def charge(self, *, organization_id, amount, description=None, actor_user_id=None, api_key_id=None):
        if "charge" in FAULTS:
            raise RuntimeError("ledger down")
        self.write(("charge", str(organization_id), amount, description, api_key_id))


def register(container: Container) -> None:
    container.bind(AgentModelRecommenderPort, lambda _session: ProbeRecommender())
    container.bind(BillingPort, ProbeBilling)
"""
MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
RECOMMEND = f"{API_ROOT}/routing/recommend"
SPAWN: dict[str, Any] = {
    "harness": "claude-code",
    "session_id": "bf05abe2-5ff2-4eb0-8459-ab578d5c9468",
    "tool_use_id": "toolu_01",
    "agent_type": "Explore",
    "description": "List files",
    "prompt": "List the files under src/ and report what each module does.",
    "parent_model": "claude-opus-5",
    "requested_model": "requested-xyz",
}
DECISION: dict[str, Any] = {
    "model": "typesafe:jev-latest",
    "state": "I've been trying to connect Stripe for 3 days.",
    "questions": {"urgency": {"type": "noul", "instructions": "Does this message express urgency?"}},
}
DECISION_ANSWER: dict[str, Any] = {
    "model": "jev-1.13.0",
    "answers": {"urgency": {"type": "noul", "noul": 0.97}},
    "usage": {"input_tokens": 100, "output_tokens": 1},
}

# One module name per test: the app is cached per distinct config, and a
# fresh name is what gets each test an app whose container imported the probe
# module the test then reads its records from.
_NAMES = (f"{MODULE}_{n}" for n in itertools.count())


@contextmanager
def _session_ledger(probe: ModuleType) -> Iterator[None]:
    """Land a probe session's pending entries when it commits, and drop them when it rolls back or closes."""
    commit, rollback, close = AsyncSession.commit, AsyncSession.rollback, AsyncSession.close

    async def committed(self: AsyncSession) -> None:
        await commit(self)
        if any(self is session for session in probe.SESSIONS):
            probe.LEDGER.extend(probe.PENDING.pop(id(self), []))
            if probe.RECORD["commits"]:
                probe.LEDGER.append(("commit",))

    async def rolled_back(self: AsyncSession) -> None:
        await rollback(self)
        probe.PENDING.pop(id(self), None)

    async def closed(self: AsyncSession) -> None:
        await close(self)
        probe.PENDING.pop(id(self), None)

    with (
        patch.object(AsyncSession, "commit", committed),
        patch.object(AsyncSession, "rollback", rolled_back),
        patch.object(AsyncSession, "close", closed),
    ):
        yield


@pytest.fixture
def rebound(
    postgres_url: str, clean_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, ModuleType]]:
    name = next(_NAMES)
    (tmp_path / f"{name}.py").write_text(PROBE_BOOTSTRAP)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        # A price the recommender quotes satisfies this; a pass-through call still needs a row.
        require_pricing=True,
        decision_providers={"typesafe": {"api_key": "ts-secret"}},
        bootstrap=f"{name}:register",
    )
    try:
        for client in build_test_client(config):
            probe = sys.modules[name]
            with _session_ledger(probe):
                yield client, probe
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def gateway_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """The gateway logger does not propagate, so caplog sees it only with its handler attached."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.ERROR, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


def _key(client: TestClient) -> dict[str, Any]:
    created = client.post(f"{API_ROOT}/keys", json={"key_name": "probe"}, headers=MASTER)
    assert created.status_code == status.HTTP_200_OK, created.text
    key: dict[str, Any] = created.json()
    return key


def _usage_rows(client: TestClient, user_id: str) -> list[dict[str, Any]]:
    rows = client.get(f"{API_ROOT}/usage", params={"user_id": user_id, "endpoint": "/v1/decisions"}, headers=MASTER)
    assert rows.status_code == status.HTTP_200_OK, rows.text
    return [dict(row) for row in rows.json()]


def test_the_bound_recommender_answers_at_the_price_it_quotes(rebound: tuple[TestClient, ModuleType]) -> None:
    client, probe = rebound
    key = _key(client)

    response = client.post(RECOMMEND, json=SPAWN, headers={API_KEY_HEADER: f"Bearer {key['key']}"})

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json() == {"model": "haiku", "reason": "the probe always says haiku", "probabilities": None}
    assert probe.ASKED == [("Explore", "requested-xyz", ["haiku", "opus", "sonnet"])]

    (row,) = _usage_rows(client, key["user_id"])
    assert (row["status"], row["provider"], row["model"]) == ("success", "probe", "recommender")
    assert float(row["cost"]) == pytest.approx(0.0015)
    user = client.get(f"{API_ROOT}/users/{key['user_id']}", headers=MASTER).json()
    assert user["spend"] == pytest.approx(0.0015)
    assert user["reserved"] == pytest.approx(0.0)

    # The quote is held, the hold is released, and the final figure is charged, on one organization.
    assert [(entry[0], entry[2]) for entry in probe.LEDGER] == [
        ("hold", Decimal("0.002")),
        ("release", Decimal("0.002")),
        ("charge", Decimal("0.0015")),
    ]
    assert len({entry[1] for entry in probe.LEDGER}) == 1
    assert probe.LEDGER[2][3:] == ("probe:recommender", key["id"])


def test_no_funds_is_refused_before_the_recommender_is_asked(rebound: tuple[TestClient, ModuleType]) -> None:
    client, probe = rebound
    key = _key(client)
    probe.FUNDS["available"] = Decimal("0")

    response = client.post(RECOMMEND, json=SPAWN, headers={API_KEY_HEADER: f"Bearer {key['key']}"})

    assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
    assert "Insufficient funds" in response.json()["detail"]
    assert probe.ASKED == []
    assert probe.LEDGER == []
    (row,) = _usage_rows(client, key["user_id"])
    assert (row["status"], row["status_code"], row["cost"]) == ("error", 402, None)
    user = client.get(f"{API_ROOT}/users/{key['user_id']}", headers=MASTER).json()
    assert user["reserved"] == pytest.approx(0.0)


def test_a_charge_that_fails_settles_nothing(rebound: tuple[TestClient, ModuleType]) -> None:
    """The hold ends and the charge lands before the budget settles, so a failed charge finds the reservation."""
    client, probe = rebound
    key = _key(client)
    probe.FAULTS.add("charge")

    with pytest.raises(RuntimeError, match="ledger down"):
        client.post(RECOMMEND, json=SPAWN, headers={API_KEY_HEADER: f"Bearer {key['key']}"})

    assert len(probe.ASKED) == 1
    # The release before the failed charge went with the rollback and was done again; one landed.
    assert len(probe.RELEASES) == 2
    assert [(entry[0], entry[2]) for entry in probe.LEDGER] == [
        ("hold", Decimal("0.002")),
        ("release", Decimal("0.002")),
    ]
    assert _usage_rows(client, key["user_id"]) == []
    user = client.get(f"{API_ROOT}/users/{key['user_id']}", headers=MASTER).json()
    assert user["spend"] == pytest.approx(0.0)
    assert user["reserved"] == pytest.approx(0.0)


def test_a_release_that_fails_midway_is_rolled_back_and_done_again(rebound: tuple[TestClient, ModuleType]) -> None:
    """What the port wrote before raising goes with the rollback; the second release starts from the committed hold."""
    client, probe = rebound
    key = _key(client)
    probe.FAULTS.add("release-once")

    with pytest.raises(RuntimeError, match="ledger down"):
        client.post(RECOMMEND, json=SPAWN, headers={API_KEY_HEADER: f"Bearer {key['key']}"})

    assert len(probe.RELEASES) == 2
    assert [(entry[0], entry[2]) for entry in probe.LEDGER] == [
        ("hold", Decimal("0.002")),
        ("release", Decimal("0.002")),
    ]
    assert _usage_rows(client, key["user_id"]) == []
    user = client.get(f"{API_ROOT}/users/{key['user_id']}", headers=MASTER).json()
    assert user["spend"] == pytest.approx(0.0)
    assert user["reserved"] == pytest.approx(0.0)


def test_a_hold_still_claimed_after_the_second_release_is_logged(
    rebound: tuple[TestClient, ModuleType], gateway_logs: pytest.LogCaptureFixture
) -> None:
    """The budget is refunded all the same, and the hold is named for releasing by hand."""
    client, probe = rebound
    key = _key(client)
    probe.FAULTS.add("release")

    with pytest.raises(RuntimeError, match="ledger down"):
        client.post(RECOMMEND, json=SPAWN, headers={API_KEY_HEADER: f"Bearer {key['key']}"})

    assert len(probe.RELEASES) == 2
    assert [(entry[0], entry[2]) for entry in probe.LEDGER] == [("hold", Decimal("0.002"))]
    messages = [record.getMessage() for record in gateway_logs.records]
    assert any("0.002" in message and "still claimed" in message for message in messages), messages
    user = client.get(f"{API_ROOT}/users/{key['user_id']}", headers=MASTER).json()
    assert user["reserved"] == pytest.approx(0.0)


def test_a_charge_past_the_quote_is_charged_as_the_quote(rebound: tuple[TestClient, ModuleType]) -> None:
    client, probe = rebound
    key = _key(client)
    probe.FAULTS.add("overcharge")

    response = client.post(RECOMMEND, json=SPAWN, headers={API_KEY_HEADER: f"Bearer {key['key']}"})

    assert response.status_code == status.HTTP_200_OK, response.text
    assert [(entry[0], entry[2]) for entry in probe.LEDGER] == [
        ("hold", Decimal("0.002")),
        ("release", Decimal("0.002")),
        ("charge", Decimal("0.002")),
    ]
    (row,) = _usage_rows(client, key["user_id"])
    assert float(row["cost"]) == pytest.approx(0.002)
    user = client.get(f"{API_ROOT}/users/{key['user_id']}", headers=MASTER).json()
    assert user["spend"] == pytest.approx(0.002)


@contextmanager
def _commits_recorded(probe: ModuleType) -> Iterator[None]:
    """Write a ``commit`` entry into the probe's ledger for each commit of a session the billing probe was built on."""
    probe.RECORD["commits"] = True
    try:
        yield
    finally:
        probe.RECORD["commits"] = False


def test_the_release_is_committed_with_the_settlement(rebound: tuple[TestClient, ModuleType]) -> None:
    """A port writing on the request session keeps its release only if a commit follows before the session closes."""
    client, probe = rebound
    key = _key(client)

    with _commits_recorded(probe):
        response = client.post(RECOMMEND, json=SPAWN, headers={API_KEY_HEADER: f"Bearer {key['key']}"})

    assert response.status_code == status.HTTP_200_OK, response.text
    steps = [entry[0] for entry in probe.LEDGER]
    assert steps[steps.index("hold") :] == ["hold", "commit", "release", "charge", "commit"]


def test_a_failed_recommendation_commits_the_release_with_the_refund(rebound: tuple[TestClient, ModuleType]) -> None:
    client, probe = rebound
    key = _key(client)
    probe.FAULTS.add("recommend")

    with _commits_recorded(probe):
        response = client.post(RECOMMEND, json=SPAWN, headers={API_KEY_HEADER: f"Bearer {key['key']}"})

    assert response.status_code == status.HTTP_502_BAD_GATEWAY, response.text
    steps = [entry[0] for entry in probe.LEDGER]
    assert steps[steps.index("hold") :] == ["hold", "commit", "release", "commit"]


def test_a_pass_through_decision_never_reaches_the_billing_port(rebound: tuple[TestClient, ModuleType]) -> None:
    client, probe = rebound
    key = _key(client)
    priced = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": "typesafe:jev-latest", "input_price_per_million": 2.0, "output_price_per_million": 10.0},
        headers=MASTER,
    )
    assert priced.status_code == status.HTTP_200_OK, priced.text

    with patch("gateway.api.routes._passthrough.request_decision", AsyncMock(return_value=DECISION_ANSWER)):
        response = client.post(f"{API_ROOT}/decisions", json=DECISION, headers={API_KEY_HEADER: f"Bearer {key['key']}"})

    assert response.status_code == status.HTTP_200_OK, response.text
    assert probe.LEDGER == []
    (row,) = _usage_rows(client, key["user_id"])
    assert (row["status"], row["provider"], row["model"]) == ("success", "typesafe", "jev-latest")


def test_the_container_says_what_was_rebound(rebound: tuple[TestClient, ModuleType]) -> None:
    client, _ = rebound

    summary = client.app.state.container.summary  # type: ignore[attr-defined]

    assert "rebound AgentModelRecommenderPort, BillingPort" in summary
