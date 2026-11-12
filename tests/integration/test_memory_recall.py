"""Automatic recall through the gateway, and the experiment that measures what memory saves."""

from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, CONVERSATION_HEADER
from gateway.models.tenancy import Workspace
from gateway.services.memory._tools import RECALL_OPEN

from .test_memory_mcp import DEPLOY, _call, _consolidate, _key
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
