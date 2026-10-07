"""Request tags: read from a request's ``metadata``, stored on its usage rows, queried back."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
)
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from conftest import seed_workspace_id
from gateway.core.config import API_ROOT
from gateway.models.usage import UsageLog
from gateway.models.users import User

from .conftest import MODEL_NAME

_LITELLM_METADATA = {"spend_logs_metadata": {"purpose": "chat", "country_code": "DE"}}
_TAGS = {"purpose": "chat", "country_code": "DE"}


class _ProviderDown(Exception):
    pass


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-tags",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _chat(client: TestClient, headers: dict[str, str], user: str, **body: Any) -> Any:
    client.post(f"{API_ROOT}/users", json={"user_id": user}, headers=headers)
    return client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}], "user": user, **body},
        headers=headers,
    )


def _row(db: Session, user: str) -> UsageLog:
    log = db.query(UsageLog).filter(UsageLog.user_id == user).one()
    db.refresh(log)
    return log


def test_chat_tags_land_on_the_usage_row_and_never_reach_the_provider(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        captured.update(kwargs)
        return _completion()

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        response = _chat(client, master_key_header, "tags-chat", metadata=_LITELLM_METADATA)

    assert response.status_code == 200, response.text
    assert "metadata" not in captured
    assert _row(db_session, "tags-chat").tags == _TAGS


def test_streamed_chat_tags_land_on_the_usage_row(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    async def chunks() -> AsyncIterator[ChatCompletionChunk]:
        yield ChatCompletionChunk(
            id="c",
            object="chat.completion.chunk",
            created=0,
            model=MODEL_NAME,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason="stop")],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def mock_acompletion(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        return chunks()

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        response = _chat(client, master_key_header, "tags-stream", stream=True, metadata={"purpose": "summarize"})
        response.read()

    assert response.status_code == 200, response.text
    assert _row(db_session, "tags-stream").tags == {"purpose": "summarize"}


def test_a_failed_request_is_tagged_too(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        raise _ProviderDown

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        response = _chat(client, master_key_header, "tags-failed", metadata={"purpose": "chat"})

    assert response.status_code >= 500
    row = _row(db_session, "tags-failed")
    assert row.status == "error"
    assert row.tags == {"purpose": "chat"}


def test_a_request_without_metadata_has_no_tags(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        return _completion()

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        assert _chat(client, master_key_header, "tags-none").status_code == 200

    assert _row(db_session, "tags-none").tags is None


def test_metadata_that_is_not_tags_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = _chat(client, master_key_header, "tags-bad", metadata={"purpose": {"nested": "object"}})
    assert response.status_code == 422
    assert "tag value must be a string" in response.text


def test_messages_forwards_anthropic_user_id_and_tags_the_rest(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    client.post(f"{API_ROOT}/users", json={"user_id": "tags-messages"}, headers=master_key_header)
    captured: dict[str, Any] = {}

    async def mock_amessages(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise _ProviderDown

    with patch("gateway.api.routes.messages.amessages", new=mock_amessages):
        client.post(
            f"{API_ROOT}/messages",
            json={
                "model": MODEL_NAME,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {"user_id": "tags-messages", **_LITELLM_METADATA},
            },
            headers=master_key_header,
        )

    assert captured, "amessages was never called"
    assert captured["metadata"] == {"user_id": "tags-messages"}
    assert _row(db_session, "tags-messages").tags == _TAGS


# ---------------------------------------------------------------------------
# Usage API
# ---------------------------------------------------------------------------


def _seed(db: Session, tags: dict[str, str] | None, cost: float) -> None:
    if db.query(User).filter(User.user_id == "tags-api").first() is None:
        db.add(User(user_id="tags-api", alias="tags-api", spend=0.0, blocked=False))
        db.flush()
    db.add(
        UsageLog(
            id=str(uuid.uuid4()),
            workspace_id=seed_workspace_id(db),
            user_id="tags-api",
            timestamp=datetime.now(UTC),
            model="gpt-4",
            provider="openai",
            endpoint="/v1/chat/completions",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=cost,
            status="success",
            tags=tags,
        )
    )


def _seed_rows(db: Session) -> None:
    _seed(db, {"purpose": "chat", "country": "DE"}, 1.0)
    _seed(db, {"purpose": "chat", "country": "US"}, 2.0)
    _seed(db, {"purpose": "summarize", "country": "DE"}, 4.0)
    _seed(db, None, 8.0)
    db.commit()


def _costs(client: TestClient, headers: dict[str, str], tags: list[str]) -> list[float]:
    response = client.get(f"{API_ROOT}/usage", params={"tag": tags}, headers=headers)
    assert response.status_code == 200, response.text
    return sorted(row["cost"] for row in response.json())


def test_usage_list_filters_by_tag(client: TestClient, master_key_header: dict[str, str], db_session: Session) -> None:
    _seed_rows(db_session)

    assert _costs(client, master_key_header, ["purpose:chat"]) == [1.0, 2.0]
    # Values for one key match any of them; different keys must all match.
    assert _costs(client, master_key_header, ["purpose:chat", "purpose:summarize"]) == [1.0, 2.0, 4.0]
    assert _costs(client, master_key_header, ["purpose:chat", "country:DE"]) == [1.0]
    assert _costs(client, master_key_header, ["purpose:missing"]) == []

    rows = client.get(f"{API_ROOT}/usage", params={"tag": "country:US"}, headers=master_key_header).json()
    assert rows[0]["tags"] == {"purpose": "chat", "country": "US"}

    count = client.get(f"{API_ROOT}/usage/count", params={"tag": "country:DE"}, headers=master_key_header)
    assert count.json() == {"total": 2}


def test_a_tag_filter_without_a_value_separator_is_refused(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.get(f"{API_ROOT}/usage", params={"tag": "purpose"}, headers=master_key_header)
    assert response.status_code == 422


def test_usage_summary_groups_by_a_tag(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _seed_rows(db_session)

    response = client.get(
        f"{API_ROOT}/usage/summary",
        params={"group_by_tag": "purpose", "dimensions": "none"},
        headers=master_key_header,
    )

    assert response.status_code == 200, response.text
    by_tag = {row["key"]: row["cost"] for row in response.json()["by_tag"]}
    assert by_tag == {None: 8.0, "summarize": 4.0, "chat": 3.0}


def test_usage_summary_without_group_by_tag_has_an_empty_breakdown(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _seed_rows(db_session)
    response = client.get(f"{API_ROOT}/usage/summary", headers=master_key_header)
    assert response.json()["by_tag"] == []
