"""Which Otari routes serve a model, from the protocols its models.dev ``api`` block names."""

import json
from pathlib import Path
from typing import get_args

import pytest

from gateway.core.config import API_ROOT
from gateway.types.model_endpoint import ModelEndpoint, models_dev_endpoints

SPEC = Path(__file__).resolve().parents[2] / "docs" / "public" / "openapi.json"


@pytest.mark.parametrize(
    ("protocol", "endpoint"),
    [
        ("chat-completions", "/chat/completions"),
        ("responses", "/responses"),
        ("messages", "/messages"),
        ("embeddings", "/embeddings"),
        ("speech", "/audio/speech"),
        ("transcriptions", "/audio/transcriptions"),
        ("systemone", "/systemone"),
        ("rerank", "/rerank"),
        ("decisions", "/decisions"),
        ("moderations", "/moderations"),
    ],
)
def test_a_protocol_otari_serves_names_its_route(protocol: str, endpoint: str) -> None:
    assert models_dev_endpoints([protocol]) == (endpoint,)


def test_the_images_protocol_names_generation_and_edit() -> None:
    assert models_dev_endpoints(["images"]) == ("/images/generations", "/images/edits")


@pytest.mark.parametrize(
    "protocol",
    [
        # A vendor's native protocol, which any-llm calls behind the OpenAI-shaped routes.
        "converse",
        "generate-content",
        "workers-ai-run",
        "interactions",
        "cohere-chat",
        "embed-content",
        "contextualized-embeddings",
        # A live, two-way session.
        "realtime",
        "bidi-generate-content",
        "live",
        "bidi-generate-music",
        # No route yet.
        "fim",
        "ocr",
        "a-protocol-models-dev-adds-later",
    ],
)
def test_a_protocol_otari_does_not_serve_names_no_route(protocol: str) -> None:
    assert models_dev_endpoints([protocol]) == ()


def test_the_videos_protocol_names_the_videos_route() -> None:
    assert models_dev_endpoints(["videos"]) == ("/videos",)


def test_the_routes_follow_models_dev_order_and_appear_once() -> None:
    protocols = ["messages", "converse", "chat-completions", "messages", "responses"]

    assert models_dev_endpoints(protocols) == ("/messages", "/chat/completions", "/responses")


def test_a_route_two_protocols_name_appears_once() -> None:
    assert models_dev_endpoints(["images", "chat-completions", "images"]) == (
        "/images/generations",
        "/images/edits",
        "/chat/completions",
    )


def test_no_protocols_name_no_route() -> None:
    assert models_dev_endpoints([]) == ()


def test_every_route_is_a_path_in_the_api() -> None:
    paths = json.loads(SPEC.read_text("utf-8"))["paths"]

    missing = [endpoint for endpoint in get_args(ModelEndpoint) if f"{API_ROOT}{endpoint}" not in paths]

    assert missing == []
