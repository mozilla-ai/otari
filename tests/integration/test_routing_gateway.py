from typing import Any
from unittest.mock import patch

from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient


def _mock_completion(model: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-routed",
        object="chat.completion",
        created=1700000000,
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="routed"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
    )


def test_routing_policy_and_project_roundtrip(client: TestClient, master_key_header: dict[str, str]) -> None:
    policy_response = client.post(
        "/v1/routing-policies",
        json={
            "name": "Default cost router",
            "strategy": "lowest_cost",
            "is_default": True,
            "config": {
                "candidates": [
                    {
                        "model": "openai:gpt-4o",
                        "input_price_per_million": 5.0,
                        "output_price_per_million": 15.0,
                    },
                    {
                        "model": "openai:gpt-4o-mini",
                        "input_price_per_million": 0.15,
                        "output_price_per_million": 0.60,
                    },
                ]
            },
        },
        headers=master_key_header,
    )
    assert policy_response.status_code == 200
    policy = policy_response.json()
    assert policy["is_default"] is True

    project_response = client.post(
        "/v1/projects",
        json={
            "project_id": "proj_test",
            "name": "Test project",
            "routing_policy_id": policy["policy_id"],
            "metadata": {"team": "infra"},
        },
        headers=master_key_header,
    )
    assert project_response.status_code == 200
    project = project_response.json()
    assert project["routing_policy_id"] == policy["policy_id"]

    list_response = client.get("/v1/routing-policies", headers=master_key_header)
    assert list_response.status_code == 200
    assert list_response.json()[0]["policy_id"] == policy["policy_id"]


def test_default_routing_selects_lowest_cost_model_and_records_trace(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    policy_response = client.post(
        "/v1/routing-policies",
        json={
            "name": "Default cost router",
            "strategy": "lowest_cost",
            "is_default": True,
            "config": {
                "candidates": [
                    {
                        "model": "openai:gpt-4o",
                        "input_price_per_million": 5.0,
                        "output_price_per_million": 15.0,
                    },
                    {
                        "model": "openai:gpt-4o-mini",
                        "input_price_per_million": 0.15,
                        "output_price_per_million": 0.60,
                    },
                ]
            },
        },
        headers=master_key_header,
    )
    assert policy_response.status_code == 200
    policy = policy_response.json()

    project_response = client.post(
        "/v1/projects",
        json={"project_id": "proj_route", "routing_policy_id": policy["policy_id"]},
        headers=master_key_header,
    )
    assert project_response.status_code == 200

    captured_kwargs: list[dict[str, Any]] = []

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        captured_kwargs.append(kwargs)
        return _mock_completion(str(kwargs["model"]))

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "default_routing",
                "project_id": "proj_route",
                "tags": {"environment": "test"},
                "messages": [{"role": "user", "content": "Say hello"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200
    assert captured_kwargs[0]["model"] == "openai:gpt-4o-mini"
    assert response.headers["X-Routed-Model"] == "openai:gpt-4o-mini"
    assert response.headers["X-Routing-Policy-ID"] == policy["policy_id"]

    traces_response = client.get("/v1/route-traces?project_id=proj_route", headers=master_key_header)
    assert traces_response.status_code == 200
    traces = traces_response.json()
    assert len(traces) == 1
    assert traces[0]["status"] == "success"
    assert traces[0]["selected_model"] == "openai:gpt-4o-mini"
    assert traces[0]["strategy"] == "lowest_cost"
    assert traces[0]["tags"] == {"environment": "test"}
    assert traces[0]["attempts"][0]["status"] == "success"


def test_default_routing_falls_back_after_provider_error(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    policy_response = client.post(
        "/v1/routing-policies",
        json={
            "name": "Priority router",
            "strategy": "priority",
            "is_default": True,
            "config": {
                "candidates": [
                    "openai:gpt-4o-mini",
                    "anthropic:claude-3-5-haiku-latest",
                ]
            },
        },
        headers=master_key_header,
    )
    assert policy_response.status_code == 200

    captured_models: list[str] = []

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        model = str(kwargs["model"])
        captured_models.append(model)
        if len(captured_models) == 1:
            raise RuntimeError("temporary provider failure")
        return _mock_completion(model)

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "default_routing",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200
    assert captured_models == ["openai:gpt-4o-mini", "anthropic:claude-3-5-haiku-latest"]
    assert response.headers["X-Routed-Model"] == "anthropic:claude-3-5-haiku-latest"

    traces_response = client.get("/v1/route-traces", headers=master_key_header)
    assert traces_response.status_code == 200
    trace = traces_response.json()[0]
    assert trace["status"] == "success"
    assert [attempt["status"] for attempt in trace["attempts"]] == ["error", "success"]
