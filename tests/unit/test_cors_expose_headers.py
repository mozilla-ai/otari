"""Every ``Otari-*`` header a response carries is readable by a browser on another origin.

The Fetch spec hides a response header from cross-origin script unless
``Access-Control-Expose-Headers`` names it, so a header left off
``_CORS_EXPOSE_HEADERS`` reaches a browser and stays invisible to the page that
asked for it. Each ``Otari-*`` literal under ``src/gateway`` is therefore
either exposed or named below as a header only requests carry, and anything
unclassified fails, so a new response header cannot be added without that
decision being made.

The request bucket only shrinks: an entry whose literal has left the source
fails until it leaves the bucket too.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.main import _CORS_EXPOSE_HEADERS, create_app

GATEWAY_ROOT = Path(__file__).resolve().parents[2] / "src" / "gateway"
OTARI_LITERAL = re.compile(r"""["'](Otari-[A-Za-z0-9-]+)["']""")

# Headers a client sends and no response carries, so exposing them would name
# nothing a browser could read.
REQUEST_HEADERS = {
    "Otari-Key": "The credential header.",
    "Otari-Router": "Names the routing policy a request runs under.",
    "Otari-Conversation-Id": "Groups a request into a conversation.",
    "Otari-Trace-Content": "Asks for a request's content to be traced.",
    "Otari-Tags": "Request tags for clients that cannot add a body field.",
    "Otari-Region": "Picks the otari.ai region a dashboard request goes to.",
    "Otari-Code-Execution": "Turns on the code execution tool for a request.",
    "Otari-Web-Search": "Turns on the web search tool for a request.",
    "Otari-Web-Fetch": "Turns on the web fetch tool for a request.",
}

MASTER_KEY = "sk-test-master"
PAGE = "https://hackathon-team.example.net"


def _otari_literals() -> set[str]:
    found: set[str] = set()
    for path in GATEWAY_ROOT.rglob("*.py"):
        found.update(OTARI_LITERAL.findall(path.read_text(encoding="utf-8")))
    return found


def test_every_otari_header_is_exposed_or_request_only() -> None:
    unclassified = _otari_literals() - set(_CORS_EXPOSE_HEADERS) - REQUEST_HEADERS.keys()

    assert not unclassified, (
        f"{sorted(unclassified)} reach the wire but are neither in _CORS_EXPOSE_HEADERS (src/gateway/main.py), "
        "which a response header belongs in, nor in REQUEST_HEADERS here."
    )


def test_request_headers_are_not_exposed() -> None:
    assert not REQUEST_HEADERS.keys() & set(_CORS_EXPOSE_HEADERS)


def test_request_bucket_names_only_live_literals() -> None:
    stale = REQUEST_HEADERS.keys() - _otari_literals()

    assert not stale, f"{sorted(stale)} no longer appear under src/gateway; remove them from REQUEST_HEADERS."


@pytest.fixture(params=["cors_allow_origins", "cors_inference_origins"])
def client(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[TestClient]:
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'cors.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
        **{request.param: [PAGE]},  # type: ignore[arg-type]
    )
    with TestClient(create_app(config)) as client:
        yield client


@pytest.mark.parametrize("header", ["Otari-Request-ID", "Otari-Model", "Otari-Guardrails", "Retry-After"])
def test_a_cross_origin_response_exposes_the_header(client: TestClient, header: str) -> None:
    response = client.post(
        f"{API_ROOT}/chat/completions",
        headers={"Origin": PAGE, "Authorization": "Bearer not-a-key"},
        json={"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )

    exposed = {name.strip().lower() for name in response.headers["access-control-expose-headers"].split(",")}
    assert header.lower() in exposed
