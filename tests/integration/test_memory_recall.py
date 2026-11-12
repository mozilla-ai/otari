"""Automatic recall through the gateway, and the experiment that measures what memory saves."""

import uuid
from collections.abc import Callable, Collection, Generator, Sequence
from typing import Any, cast
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, CONVERSATION_HEADER, GatewayConfig
from gateway.models.tenancy import Workspace
from gateway.models.usage import UsageLog
from gateway.repositories.memory import EntryRepository
from gateway.services.memory import MemoryModels, ModelUse, RecallVectors
from gateway.services.memory._models import RERANK_ENDPOINT, MemoryModelError
from gateway.services.memory._tools import RECALL_OPEN
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client
from .test_guardrails_route import _ROUTES, _streamed
from .test_memory_mcp import DEPLOY, _call, _consolidate, _embed, _key
from .test_memory_mcp import memory_client as memory_client  # noqa: F401 - the fixture, used by name
from .test_memory_tool_loop import _chat, _completion, _offered, _provider, _switches

ASK = "how do I deploy the api from main?"


def _workspace_id(db_session_factory: Callable[[], Session]) -> str:
    with db_session_factory() as session:
        workspace = session.scalars(select(Workspace)).first()
        assert workspace is not None
        return str(workspace.id)


def _remembered(client: TestClient, headers: dict[str, str]) -> None:
    proposed = _call(client, headers, "memory_propose", text=DEPLOY, kind="procedure")
    assert proposed["isError"] is False
    assert _consolidate(client) == 1


def _last_user_text(sent: dict[str, Any]) -> str:
    content = sent["messages"][-1]["content"]
    return content if isinstance(content, str) else "\n".join(part.get("text", "") for part in content)


def test_recall_adds_what_memory_holds_to_the_newest_turn_without_a_tool_call(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)
    _remembered(memory_client, headers)
    _switches(memory_client, db_session_factory, master_key_header, auto_recall=True)

    with _provider(_completion(None)) as sent:
        response = _chat(memory_client, headers, messages=[{"role": "user", "content": ASK}])

    assert response.status_code == 200, response.text
    assert response.headers["Otari-Memory-Recalled"] == "1"
    text = _last_user_text(sent[0])
    assert text.startswith(ASK) and RECALL_OPEN in text and DEPLOY in text
    assert not _offered(sent[0]), "recall needs no tools"
    assert len(sent) == 1, "and no extra round"


def test_recall_leaves_alone_what_is_not_alike_a_tool_turn_and_a_request_that_declines(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)
    _remembered(memory_client, headers)
    _switches(memory_client, db_session_factory, master_key_header, auto_recall=True)
    tool_turn = [
        {"role": "user", "content": ASK},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c", "content": "done"},
    ]

    with _provider(_completion(None), _completion(None), _completion(None)) as sent:
        unlike = _chat(memory_client, headers, messages=[{"role": "user", "content": "write a test to push"}])
        after_tool = _chat(memory_client, headers, messages=tool_turn)
        declined = _chat(memory_client, {**headers, "Otari-Memory": "off"}, messages=[{"role": "user", "content": ASK}])

    assert unlike.headers["Otari-Memory-Recalled"] == "0"
    assert "Otari-Memory-Recalled" not in after_tool.headers
    assert "Otari-Memory-Recalled" not in declined.headers
    assert all(RECALL_OPEN not in str(call["messages"]) for call in sent)


def test_a_follow_up_finds_what_its_conversation_is_about(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)
    _remembered(memory_client, headers)
    _switches(memory_client, db_session_factory, master_key_header, auto_recall=True)
    follow_up = {"role": "user", "content": "and on which branch?"}
    conversation = [{"role": "user", "content": ASK}, {"role": "assistant", "content": "With make."}, follow_up]

    with _provider(_completion(None), _completion(None)) as sent:
        alone = _chat(memory_client, headers, messages=[follow_up])
        in_conversation = _chat(memory_client, headers, messages=conversation)

    assert alone.headers["Otari-Memory-Recalled"] == "0", "the follow-up alone names too little"
    assert in_conversation.headers["Otari-Memory-Recalled"] == "1"
    assert DEPLOY in _last_user_text(sent[1])
    assert sent[1]["messages"][0] == {"role": "user", "content": ASK}, "the earlier turns are searched, not changed"


def test_recall_leaves_out_what_is_far_less_alike_than_the_best(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)
    _remembered(memory_client, headers)
    nearby = "Run the api test suite on main before a deploy"
    assert _call(memory_client, headers, "memory_propose", text=nearby, kind="procedure")["isError"] is False
    assert _consolidate(memory_client) == 1
    _switches(memory_client, db_session_factory, master_key_header, auto_recall=True, recall_min_similarity=0.7)

    with _provider(_completion(None)) as sent:
        response = _chat(
            memory_client,
            headers,
            messages=[{"role": "user", "content": "how do I deploy the api from the main branch?"}],
        )

    # Both clear the floor, but the second only shares some words with what is asked.
    assert response.headers["Otari-Memory-Recalled"] == "1"
    text = _last_user_text(sent[0])
    assert DEPLOY in text and nearby not in text


# -- Checking relevance with the deployment's rerank model ------------------

NEARBY = "Run the api test suite on main before a deploy"
# Alike to DEPLOY in one word alone, so similarity leaves it out, but it is what DEPLOY answers.
WHICH_BRANCH = "and on which branch?"


async def _rerank_on_branch(self: MemoryModels, query: str, documents: Sequence[str]) -> tuple[list[float], ModelUse]:
    """Scores what speaks of the branch the query asks for as relevant, anything else as not."""
    scores = [0.9 if "branch" in query and "branch" in document else 0.1 for document in documents]
    return scores, ModelUse(RERANK_ENDPOINT, "cohere:rerank", "cohere", True, 11, None, 3)


async def _rerank_down(self: MemoryModels, query: str, documents: Sequence[str]) -> tuple[list[float], ModelUse]:
    raise MemoryModelError(
        "the memory rerank call failed", ModelUse(RERANK_ENDPOINT, "cohere:rerank", "cohere", False, None, None, 3)
    )


@pytest.fixture
def reranking_client(
    test_config: GatewayConfig, clean_database: None, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    """A deployment that names a rerank model."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.setattr(MemoryModels, "embed", _embed)
    monkeypatch.setattr(MemoryModels, "rerank", _rerank_on_branch)
    config = test_config.model_copy(
        update={
            "memory_enabled": True,
            "memory_embedding_model": "openai:embed",
            "memory_rerank_model": "cohere:rerank",
        }
    )
    yield from build_test_client(config)


def _both_remembered(client: TestClient, headers: dict[str, str]) -> None:
    _remembered(client, headers)
    assert _call(client, headers, "memory_propose", text=NEARBY, kind="procedure")["isError"] is False
    assert _consolidate(client) == 1


def _rerank_calls(db_session_factory: Callable[[], Session]) -> list[bool]:
    with db_session_factory() as session:
        return list(session.scalars(select(UsageLog.status == "success").where(UsageLog.endpoint == RERANK_ENDPOINT)))


def test_checking_relevance_hands_over_what_the_reranker_scores_relevant_whatever_its_similarity(
    reranking_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(reranking_client, master_key_header, db_session_factory)
    _both_remembered(reranking_client, headers)
    _switches(reranking_client, db_session_factory, master_key_header, auto_recall=True, recall_rerank=True)
    memory = reranking_client.get(
        f"{API_ROOT}/workspaces/{_workspace_id(db_session_factory)}/memory", headers=master_key_header
    ).json()
    assert memory["rerank_available"] is True and memory["recall_min_relevance"] == 0.8

    with _provider(_completion(None), _completion(None)) as sent:
        branch = _chat(reranking_client, headers, messages=[{"role": "user", "content": WHICH_BRANCH}])
        deploy = _chat(reranking_client, headers, messages=[{"role": "user", "content": ASK}])

    assert branch.headers["Otari-Memory-Recalled"] == "1", "below the similarity floor, yet relevant"
    text = _last_user_text(sent[0])
    assert DEPLOY in text and NEARBY not in text
    assert deploy.headers["Otari-Memory-Recalled"] == "0", "alike, yet scored not relevant"
    assert _rerank_calls(db_session_factory) == [True, True], "each check is metered as memory's own call"


def test_the_reranker_reads_the_turn_first_and_the_conversation_after_it(
    reranking_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = _key(reranking_client, master_key_header, db_session_factory)
    _remembered(reranking_client, headers)
    _switches(reranking_client, db_session_factory, master_key_header, auto_recall=True, recall_rerank=True)
    queries: list[str] = []

    async def capture(self: MemoryModels, query: str, documents: Sequence[str]) -> tuple[list[float], ModelUse]:
        queries.append(query)
        return await _rerank_on_branch(self, query, documents)

    monkeypatch.setattr(MemoryModels, "rerank", capture)
    conversation = [{"role": "user", "content": ASK}, {"role": "assistant", "content": "With make."}]

    with _provider(_completion(None), _completion(None)):
        _chat(reranking_client, headers, messages=[{"role": "user", "content": WHICH_BRANCH}])
        _chat(reranking_client, headers, messages=[*conversation, {"role": "user", "content": WHICH_BRANCH}])

    assert queries == [WHICH_BRANCH, f"{WHICH_BRANCH}\n(Said earlier in the conversation: {ASK})"]


def test_recall_falls_back_to_similarity_when_relevance_cannot_be_checked(
    reranking_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = _key(reranking_client, master_key_header, db_session_factory)
    _remembered(reranking_client, headers)
    _switches(reranking_client, db_session_factory, master_key_header, auto_recall=True, recall_rerank=True)
    monkeypatch.setattr(MemoryModels, "rerank", _rerank_down)

    with _provider(_completion(None)) as sent:
        response = _chat(reranking_client, headers, messages=[{"role": "user", "content": ASK}])

    assert response.status_code == 200, response.text
    assert response.headers["Otari-Memory-Recalled"] == "1"
    assert DEPLOY in _last_user_text(sent[0])
    assert _rerank_calls(db_session_factory) == [False]


def test_a_deployment_without_a_rerank_model_keeps_the_switch_and_recalls_by_similarity(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)
    _remembered(memory_client, headers)
    _switches(
        memory_client,
        db_session_factory,
        master_key_header,
        auto_recall=True,
        recall_rerank=True,
        recall_min_relevance=0.8,
    )
    memory = memory_client.get(
        f"{API_ROOT}/workspaces/{_workspace_id(db_session_factory)}/memory", headers=master_key_header
    ).json()
    assert memory["rerank_available"] is False
    assert memory["recall_rerank"] is True and memory["recall_min_relevance"] == 0.8

    with _provider(_completion(None)):
        response = _chat(memory_client, headers, messages=[{"role": "user", "content": WHICH_BRANCH}])

    assert response.headers["Otari-Memory-Recalled"] == "0", "too little alike, and nothing checks relevance"


def test_the_experiment_serves_its_control_sessions_without_memory_and_reports_both_arms(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)
    _remembered(memory_client, headers)
    _switches(
        memory_client,
        db_session_factory,
        master_key_header,
        auto_recall=True,
        offer_to_every_request=True,
        experiment_holdout_percent=50,
    )
    sessions = [f"task-{index}" for index in range(12)]

    with _provider(*(_completion(None) for _ in range(len(sessions) + 1))) as sent:
        responses = [
            _chat(memory_client, {**headers, CONVERSATION_HEADER: session}, messages=[{"role": "user", "content": ASK}])
            for session in sessions
        ]
        # A request that names no session is in no arm: it gets memory as usual, and is left out of the comparison.
        without_session = _chat(memory_client, headers, messages=[{"role": "user", "content": ASK}])

    arms = [response.headers["Otari-Memory-Arm"] for response in responses]
    assert set(arms) == {"control", "memory"}, "twelve sessions at an even split land in both arms"
    for arm, call in zip(arms, sent[: len(arms)], strict=True):
        with_memory = arm == "memory"
        assert (RECALL_OPEN in _last_user_text(call)) is with_memory
        assert bool(_offered(call)) is with_memory
    assert "Otari-Memory-Arm" not in without_session.headers
    assert RECALL_OPEN in _last_user_text(sent[-1])

    workspace_id = _workspace_id(db_session_factory)
    report = memory_client.get(f"{API_ROOT}/workspaces/{workspace_id}/memory/experiment", headers=master_key_header)
    assert report.status_code == 200, report.text
    body = report.json()
    assert body["holdout_percent"] == 50 and body["ended_at"] is None
    assert body["control"]["sessions"] == arms.count("control")
    assert body["memory"]["sessions"] == arms.count("memory")
    assert body["control"]["tokens_per_session"] == 15
    assert body["tokens"] is None, "a dozen sessions is too few to estimate"
    # Recall counts every turn it searched for in the window, the session-less request's too.
    assert body["recalls"] == arms.count("memory") + 1
    assert body["recalls_with_results"] == arms.count("memory") + 1
    assert "/v1/memory/recall" in body["memory_calls"]


def test_stopping_the_experiment_keeps_its_results_and_restarting_deals_afresh(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    workspace_id = _workspace_id(db_session_factory)
    _key(memory_client, master_key_header, db_session_factory)
    url = f"{API_ROOT}/workspaces/{workspace_id}/memory"

    started = memory_client.put(
        url, json={"enabled": True, "experiment_holdout_percent": 20}, headers=master_key_header
    )
    assert started.status_code == 200, started.text
    began = started.json()["experiment_started_at"]
    kept = memory_client.put(url, json={"enabled": True, "experiment_holdout_percent": 20}, headers=master_key_header)
    assert kept.json()["experiment_started_at"] == began, "the same holdout keeps it running"

    stopped = memory_client.put(url, json={"enabled": True}, headers=master_key_header).json()
    assert stopped["experiment_holdout_percent"] == 0
    assert stopped["experiment_started_at"] == began and stopped["experiment_ended_at"] is not None
    report = memory_client.get(f"{url}/experiment", headers=master_key_header).json()
    assert report["holdout_percent"] == 20 and report["ended_at"] == stopped["experiment_ended_at"]

    restarted = memory_client.put(
        url, json={"enabled": True, "experiment_holdout_percent": 30}, headers=master_key_header
    ).json()
    assert restarted["experiment_started_at"] != began and restarted["experiment_ended_at"] is None


def test_the_experiment_report_is_for_the_workspaces_managers(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    workspace_id = _workspace_id(db_session_factory)
    member = _key(memory_client, master_key_header, db_session_factory)

    refused = memory_client.get(f"{API_ROOT}/workspaces/{workspace_id}/memory/experiment", headers=member)

    assert refused.status_code in (401, 403)


# -- What recall holds of the entries it opened ------------------------------


@pytest.fixture
def opened_embeddings(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """How many sealed embeddings each recall read from the database to open."""
    reads: list[int] = []
    original = EntryRepository.sealed_embeddings

    async def counted(self: EntryRepository, workspace_id: uuid.UUID, entry_ids: Collection[uuid.UUID]) -> Any:
        reads.append(len(entry_ids))
        return await original(self, workspace_id, entry_ids)

    monkeypatch.setattr(EntryRepository, "sealed_embeddings", counted)
    return reads


def _held(client: TestClient) -> int:
    held = getattr(cast(FastAPI, client.app).state, "memory_recall_vectors", None)
    return len(held) if held is not None else 0


def _hold_nothing_yet(client: TestClient) -> None:
    """The app outlives a test, and what it holds with it."""
    cast(FastAPI, client.app).state.memory_recall_vectors = RecallVectors()


def _retire_every_entry(db_session_factory: Callable[[], Session], workspace_id: str) -> None:
    """As another process would, past this one's service: what this process holds of them is not told."""
    with db_session_factory() as session:
        session.execute(text("SELECT set_config('otari.memory_workspace', :w, true)"), {"w": workspace_id})
        session.execute(
            text("UPDATE memory_entries SET status = 'retired' WHERE workspace_id = :w"), {"w": workspace_id}
        )
        session.commit()


def test_recall_opens_an_entry_once_and_hands_over_only_what_the_database_still_lists(
    memory_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    opened_embeddings: list[int],
) -> None:
    _hold_nothing_yet(memory_client)
    headers = _key(memory_client, master_key_header, db_session_factory)
    _remembered(memory_client, headers)
    _switches(memory_client, db_session_factory, master_key_header, auto_recall=True)

    with _provider(_completion(None), _completion(None), _completion(None)):
        first = _chat(memory_client, headers, messages=[{"role": "user", "content": ASK}])
        second = _chat(memory_client, headers, messages=[{"role": "user", "content": ASK}])
        _retire_every_entry(db_session_factory, _workspace_id(db_session_factory))
        after = _chat(memory_client, headers, messages=[{"role": "user", "content": ASK}])

    assert first.headers["Otari-Memory-Recalled"] == second.headers["Otari-Memory-Recalled"] == "1"
    assert opened_embeddings == [1], "the second turn opens nothing"
    assert after.headers["Otari-Memory-Recalled"] == "0", "held, yet no longer listed"
    assert _held(memory_client) == 1


def test_erasing_a_workspace_drops_what_recall_holds_of_it(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _hold_nothing_yet(memory_client)
    headers = _key(memory_client, master_key_header, db_session_factory)
    _remembered(memory_client, headers)
    _switches(memory_client, db_session_factory, master_key_header, auto_recall=True)
    with _provider(_completion(None)):
        _chat(memory_client, headers, messages=[{"role": "user", "content": ASK}])
    assert _held(memory_client) == 1

    erased = memory_client.delete(
        f"{API_ROOT}/workspaces/{_workspace_id(db_session_factory)}/memory", headers=master_key_header
    )

    assert erased.status_code == 200, erased.text
    assert _held(memory_client) == 0


@pytest.fixture
def uncached_client(
    test_config: GatewayConfig, clean_database: None, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    """A deployment that holds no opened embeddings."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.setattr(MemoryModels, "embed", _embed)
    config = test_config.model_copy(
        update={"memory_enabled": True, "memory_embedding_model": "openai:embed", "memory_recall_cache": False}
    )
    yield from build_test_client(config)


def test_a_deployment_that_holds_nothing_opens_every_entry_each_turn(
    uncached_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    opened_embeddings: list[int],
) -> None:
    headers = _key(uncached_client, master_key_header, db_session_factory)
    _remembered(uncached_client, headers)
    _switches(uncached_client, db_session_factory, master_key_header, auto_recall=True)

    with _provider(_completion(None), _completion(None)):
        for _ in range(2):
            response = _chat(uncached_client, headers, messages=[{"role": "user", "content": ASK}])
            assert response.headers["Otari-Memory-Recalled"] == "1"

    assert opened_embeddings == [1, 1]
    assert _held(uncached_client) == 0


def test_a_streamed_answer_says_what_memory_did(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)
    _remembered(memory_client, headers)
    _switches(memory_client, db_session_factory, master_key_header, auto_recall=True)
    _, target, _ = _ROUTES["chat"]

    with patch(target, side_effect=_streamed("chat")):
        response = _chat(memory_client, headers, messages=[{"role": "user", "content": ASK}], stream=True)
        response.read()

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["Otari-Memory-Recalled"] == "1"
