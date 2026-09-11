"""Unit tests for the guardrail catalog (``fetch_guardrail_catalog``).

Stubs the guardrails service ``GET /profiles`` contract with an
``httpx.MockTransport``. The parameter half is not stubbed: it comes from the
installed ``any_guardrail`` registry, which is the point of the join, so these
assert against real entries in it rather than against a fixture that could agree
with a schema nobody ships.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx
import pytest

from gateway.log_config import logger as gateway_logger
from gateway.services.guardrail_catalog import fetch_guardrail_catalog

_URL = "http://anyguardrails:8000"


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    """Replace the module's ``httpx.AsyncClient`` with one backed by ``handler``."""
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient  # captured before patching, to avoid recursion

    def factory(*_args: object, **_kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=transport)

    monkeypatch.setattr("gateway.services.guardrail_catalog.httpx.AsyncClient", factory)


def _profiles_handler(rows: object, status_code: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/profiles"
        return httpx.Response(status_code, json=rows)

    return handler


@pytest.mark.asyncio
async def test_lists_the_services_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(
        monkeypatch,
        _profiles_handler(
            [
                {"name": "prompt-injection", "guardrail_name": "injec_guard", "model_id": "leolee99/InjecGuard"},
                {"name": "house-policy", "guardrail_name": "any_llm", "model_id": None},
            ]
        ),
    )

    catalog = await fetch_guardrail_catalog(_URL)

    assert catalog.available is True
    assert catalog.reason is None
    # Sorted by profile, so the picker's order does not depend on a dict's.
    assert [profile.profile for profile in catalog.profiles] == ["house-policy", "prompt-injection"]
    assert catalog.profiles[1].model_id == "leolee99/InjecGuard"


@pytest.mark.asyncio
async def test_types_the_validate_kwargs_a_profile_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(monkeypatch, _profiles_handler([{"name": "house-policy", "guardrail_name": "any_llm"}]))

    catalog = await fetch_guardrail_catalog(_URL)

    parameters = {parameter.name: parameter for parameter in catalog.profiles[0].parameters}
    assert catalog.profiles[0].parameters_known is True
    # any_llm judges against a policy the caller supplies, and refuses without
    # one, which is exactly the guardrail a free-text profile box cannot set up.
    assert parameters["policy"].required is True
    assert parameters["policy"].type == "string"
    assert parameters["policy"].description is not None
    # An enum arrives with the choices a picker needs rather than as free text.
    assert parameters["prompt_version"].type == "enum"
    assert parameters["prompt_version"].choices


@pytest.mark.asyncio
async def test_omits_the_constructor_stage_the_operators_yaml_owns(monkeypatch: pytest.MonkeyPatch) -> None:
    """``create`` kwargs are the sidecar's own boot config and have no target here."""
    _patch_transport(monkeypatch, _profiles_handler([{"name": "policy", "guardrail_name": "any_llm"}]))

    catalog = await fetch_guardrail_catalog(_URL)

    # any_llm takes `provider`/`model_id` style constructor arguments upstream;
    # none of them may reach a form whose only write target is validate_kwargs.
    assert {parameter.name for parameter in catalog.profiles[0].parameters} == {
        "policy",
        "model_id",
        "system_prompt",
        "prompt_version",
    }


@pytest.mark.asyncio
async def test_keeps_a_profile_this_gateway_has_no_schema_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sidecar on a newer any-guardrail still gets a selectable profile."""
    _patch_transport(monkeypatch, _profiles_handler([{"name": "novel", "guardrail_name": "not_shipped_yet"}]))

    catalog = await fetch_guardrail_catalog(_URL)

    assert catalog.available is True
    assert catalog.profiles[0].parameters_known is False
    assert catalog.profiles[0].parameters == []


@pytest.mark.asyncio
async def test_drops_only_the_rows_it_cannot_read(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(
        monkeypatch,
        _profiles_handler([{"name": "kept", "guardrail_name": "injec_guard"}, {"guardrail_name": "injec_guard"}, 7]),
    )

    catalog = await fetch_guardrail_catalog(_URL)

    assert catalog.available is True
    assert [profile.profile for profile in catalog.profiles] == ["kept"]


@pytest.mark.asyncio
async def test_no_service_configured_is_a_reason_not_an_error() -> None:
    catalog = await fetch_guardrail_catalog(None)

    assert catalog.available is False
    assert catalog.profiles == []
    assert catalog.reason is not None and "configured" in catalog.reason


@pytest.mark.asyncio
async def test_a_service_without_the_endpoint_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(monkeypatch, _profiles_handler({"detail": "Not Found"}, status_code=404))

    catalog = await fetch_guardrail_catalog(_URL)

    assert catalog.available is False
    assert catalog.reason is not None and "/profiles" in catalog.reason


@pytest.mark.asyncio
async def test_an_unreachable_service_never_names_its_address(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    _patch_transport(monkeypatch, handler)

    catalog = await fetch_guardrail_catalog("https://guardrails.internal.example")

    assert catalog.available is False
    assert catalog.reason is not None
    # The endpoint goes to the log, for the reason a 502 body keeps it out.
    assert "guardrails.internal.example" not in catalog.reason


@pytest.mark.asyncio
async def test_a_non_list_answer_is_malformed_rather_than_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(monkeypatch, _profiles_handler({"profiles": []}))

    catalog = await fetch_guardrail_catalog(_URL)

    assert catalog.available is False
    assert catalog.profiles == []


@pytest.mark.asyncio
async def test_trailing_slash_does_not_double_up(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json=[])

    _patch_transport(monkeypatch, handler)
    await fetch_guardrail_catalog(f"{_URL}/")

    assert seen == ["/profiles"]


@pytest.mark.asyncio
async def test_the_log_line_masks_a_url_password(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``guardrails_url`` may carry userinfo, which the settings endpoints already mask."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    _patch_transport(monkeypatch, handler)
    # The gateway logger does not propagate, so caplog has to be attached to it.
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")
    try:
        catalog = await fetch_guardrail_catalog("https://otari:hunter2@guardrails.example")
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert catalog.available is False
    assert "hunter2" not in caplog.text
    assert "guardrails.example" in caplog.text


@pytest.mark.asyncio
async def test_a_body_past_the_cap_is_not_held(monkeypatch: pytest.MonkeyPatch) -> None:
    """The five-second timeout bounds how long the answer takes, not its size."""
    oversized = b'[{"name": "x", "guardrail_name": "injec_guard"}]' + b" " * (2 * 1024 * 1024)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=oversized)

    _patch_transport(monkeypatch, handler)

    catalog = await fetch_guardrail_catalog(_URL)

    assert catalog.available is False
    assert catalog.profiles == []
    assert catalog.reason is not None


@pytest.mark.asyncio
async def test_more_profiles_than_a_picker_could_serve_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"name": f"p{index}", "guardrail_name": "injec_guard"} for index in range(501)]
    _patch_transport(monkeypatch, _profiles_handler(rows))

    catalog = await fetch_guardrail_catalog(_URL)

    assert catalog.available is False
    assert catalog.profiles == []


@pytest.mark.asyncio
async def test_an_unusable_configured_url_is_a_reason_not_a_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the dashboard PATCH validates ``guardrails_url``; env and YAML reach this raw.

    ``httpx.InvalidURL`` is not an ``HTTPError``, so it is named in the handler
    alongside one. Raised from the client here rather than reached through a
    malformed address, because the stub transport below is what would otherwise
    answer it: the real client refuses the protocol before any transport sees
    the request, which is the arm that catches it either way.
    """

    def factory(*_args: object, **_kwargs: object) -> httpx.AsyncClient:
        raise httpx.InvalidURL("no host")

    monkeypatch.setattr("gateway.services.guardrail_catalog.httpx.AsyncClient", factory)

    catalog = await fetch_guardrail_catalog("http://")

    assert catalog.available is False
    assert catalog.profiles == []
    assert catalog.reason is not None
