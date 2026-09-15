"""Unit tests for the guardrail catalog (``fetch_guardrail_catalog``).

Stubs the guardrails service ``GET /profiles`` contract with an
``httpx.MockTransport``. The parameter half is not stubbed: it comes from the
installed ``any_guardrail`` registry, which is the point of the join, so these
assert against real entries in it rather than against a fixture that could agree
with a schema nobody ships.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterator

import httpx
import pytest
from any_guardrail.base import GuardrailName
from any_guardrail.parameters import ParameterType as UpstreamParameterType
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import BackendType, OutputShape
from any_guardrail.taxonomy import GuardrailCategory as UpstreamCategory
from any_guardrail.taxonomy import GuardrailStage as UpstreamStage

from gateway.log_config import logger as gateway_logger
from gateway.services.guardrail_catalog import (
    _BACKEND_PACKAGES,
    _KNOWN_TYPES,
    LOCAL_GUARDRAILS_EXTRA,
    BuiltInGuardrailCatalog,
    BuiltInGuardrailSpec,
    GuardrailParameterSpec,
    _backend_availability,
    _installed,
    build_builtin_guardrail_catalog,
    fetch_guardrail_catalog,
)

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


# ---------------------------------------------------------------------------
# The built-in catalog (``build_builtin_guardrail_catalog``).
#
# Reads the installed any-guardrail registry with nothing stubbed, for the reason
# the tests above leave the parameter half real: a fixture here could agree with a
# schema nobody ships. Only the backend probe is faked, so an assertion about
# `runnable` does not depend on which extras this environment happens to hold.
# ---------------------------------------------------------------------------


def _spec(catalog: BuiltInGuardrailCatalog, guardrail_name: str) -> BuiltInGuardrailSpec:
    return next(spec for spec in catalog.guardrails if spec.guardrail_name == guardrail_name)


def _force_probe(monkeypatch: pytest.MonkeyPatch, *, installed: bool) -> None:
    """Answer every module probe the same way, whatever this environment installed."""
    monkeypatch.setattr("gateway.services.guardrail_catalog._installed", lambda _package: installed)


@pytest.fixture(autouse=True)
def _clear_backend_cache() -> Iterator[None]:
    """The probe is cached for the process; a test must not inherit another's answer."""
    _backend_availability.cache_clear()
    yield
    _backend_availability.cache_clear()


def test_lists_only_the_guardrails_a_hosted_api_reaches() -> None:
    """The set is derived from upstream's metadata, so an addition there reaches it."""
    listed = {spec.guardrail_name for spec in build_builtin_guardrail_catalog().guardrails}

    assert listed == {
        name.value
        for name, metadata in GUARDRAIL_METADATA.items()
        if BackendType.HOSTED_API in ({metadata.backend} | metadata.alternate_backends)
    }


def test_omits_a_guardrail_that_would_load_model_weights() -> None:
    """The whole point. Otari builds none of these, so offering them is offering nothing."""
    listed = {spec.guardrail_name for spec in build_builtin_guardrail_catalog().guardrails}

    assert not listed & {"llama_guard", "prompt_guard", "injec_guard", "lettuce_detect"}


def test_lists_a_local_guardrail_that_also_has_a_hosted_path() -> None:
    """SusFactor is why the rule reads `alternate_backends` and not `backend` alone."""
    listed = {spec.guardrail_name for spec in build_builtin_guardrail_catalog().guardrails}

    assert "susfactor" in listed


def test_orders_the_catalog_for_a_picker() -> None:
    catalog = build_builtin_guardrail_catalog()

    names = [spec.display_name for spec in catalog.guardrails]
    assert names == sorted(names, key=str.casefold)


def test_publishes_the_constructor_stage_a_stored_guardrail_owns() -> None:
    """The create stage is the point: it is where a vendor API key lives."""
    api_key = next(
        parameter
        for parameter in _spec(build_builtin_guardrail_catalog(), "lakera_guard").create_parameters
        if parameter.name == "api_key"
    )

    assert api_key.secret
    assert api_key.storable
    # Optional in the signature and read from LAKERA_API_KEY, so only upstream's
    # effectively-required flag stops the form rendering it as skippable.
    assert api_key.required


def test_publishes_both_stages_of_one_guardrail() -> None:
    spec = _spec(build_builtin_guardrail_catalog(), "any_llm")

    assert {parameter.name for parameter in spec.validate_parameters} == {
        "policy",
        "model_id",
        "system_prompt",
        "prompt_version",
    }
    # any_llm is the one hosted guardrail taking no constructor arguments, which
    # is why the two stages are published as separate lists rather than merged.
    assert spec.create_parameters == []


def test_marks_a_live_object_secret_as_unstorable() -> None:
    """A json-typed secret is an authenticated client, not a value to write down."""
    session = next(
        parameter
        for parameter in _spec(build_builtin_guardrail_catalog(), "bedrock_guardrails").create_parameters
        if parameter.name == "boto3_session"
    )

    assert session.secret
    assert session.type == "json"
    assert not session.storable


def test_a_plain_secret_stays_storable() -> None:
    key = next(
        parameter
        for parameter in _spec(build_builtin_guardrail_catalog(), "watsonx_guardian").create_parameters
        if parameter.name == "api_key"
    )

    assert key.secret
    assert key.storable


def test_carries_the_metadata_a_picker_groups_by() -> None:
    spec = _spec(build_builtin_guardrail_catalog(), "lakera_guard")

    assert spec.backend == "hosted_api"
    assert spec.primary_category == "prompt_injection"
    assert spec.requires_api_key
    assert spec.display_name
    assert spec.description
    assert spec.vendor
    assert spec.default_license
    assert spec.stages


def test_names_the_environment_variable_that_fills_a_parameter() -> None:
    """So a form can offer "or set this" rather than demanding a key the host already has."""
    spec = _spec(build_builtin_guardrail_catalog(), "openai_moderation")

    api_key = next(parameter for parameter in spec.create_parameters if parameter.name == "api_key")

    assert api_key.env_var == "OPENAI_API_KEY"
    assert api_key.secret


def test_does_not_say_whether_that_environment_variable_is_set() -> None:
    """The catalog is readable by any dashboard session, so it names the variable and stops there."""
    published = set(GuardrailParameterSpec.model_fields)

    assert "env_var" in published
    assert not published & {"env_var_set", "env_var_value", "value"}


def test_publishes_a_one_of_requirement_no_single_parameter_can_express() -> None:
    """Watsonx needs a project or a space, which every parameter alone reads as optional."""
    spec = _spec(build_builtin_guardrail_catalog(), "watsonx_guardian")

    project_or_space = next(group for group in spec.requirement_groups if "project_id" in group.parameters)

    assert set(project_or_space.parameters) == {"project_id", "space_id", "api_client"}
    assert set(project_or_space.env_vars) == {"WATSONX_PROJECT_ID", "WATSONX_SPACE_ID"}
    assert project_or_space.description
    # Each member reads optional on its own, which is the whole reason the group exists.
    for name in ("project_id", "space_id"):
        assert not next(p for p in spec.create_parameters if p.name == name).required


def test_leaves_requirement_groups_empty_for_a_guardrail_without_one() -> None:
    """The majority. An empty list must not read as "constraints unknown"."""
    assert _spec(build_builtin_guardrail_catalog(), "lakera_guard").requirement_groups == []


def test_reports_a_second_way_to_run_the_same_guardrail() -> None:
    """Susfactor also has a hosted path, which one runnable flag cannot express."""
    spec = _spec(build_builtin_guardrail_catalog(), "susfactor")

    assert spec.model_dump(mode="json")["alternate_backends"] == ["hosted_api"]


def test_a_guardrail_whose_backend_is_installed_is_runnable(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_probe(monkeypatch, installed=True)

    spec = _spec(build_builtin_guardrail_catalog(), "azure_content_safety")

    assert spec.runnable
    assert spec.missing_extra is None


def test_a_guardrail_whose_backend_is_absent_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_probe(monkeypatch, installed=False)

    spec = _spec(build_builtin_guardrail_catalog(), "azure_content_safety")

    assert not spec.runnable
    assert spec.missing_extra == LOCAL_GUARDRAILS_EXTRA


def test_a_hosted_guardrail_needs_no_extra_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """The base install reaches Lakera over `requests`, so nothing is probed."""
    _force_probe(monkeypatch, installed=False)

    spec = _spec(build_builtin_guardrail_catalog(), "lakera_guard")

    assert spec.runnable
    assert spec.missing_extra is None


def test_a_guardrail_with_no_backend_information_is_a_gap_not_a_guess(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newer any-guardrail could ship one; reporting it runnable would be a lie."""
    monkeypatch.delitem(_BACKEND_PACKAGES, GuardrailName.LAKERA_GUARD)

    spec = _spec(build_builtin_guardrail_catalog(), "lakera_guard")

    assert not spec.runnable
    assert spec.missing_extra is None


def test_every_listed_guardrail_has_backend_information() -> None:
    """A guardrail upstream adds must be given a probe, not left to the gap above."""
    listed = {GuardrailName(spec.guardrail_name) for spec in build_builtin_guardrail_catalog().guardrails}

    assert listed <= set(_BACKEND_PACKAGES)


def test_a_missing_module_is_not_installed() -> None:
    assert not _installed("a_module_no_one_ships")
    # A dotted probe whose parent is absent raises rather than answering None.
    assert not _installed("a_module_no_one_ships.deeper")


def test_listing_the_catalog_never_loads_a_model_backend() -> None:
    """The whole point of reading the registry rather than constructing anything."""
    build_builtin_guardrail_catalog()

    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_publishes_every_taxonomy_value_upstream_can_report() -> None:
    """The taxonomy fields are upstream's enums, so a member it adds is carried, not dropped."""
    published = BuiltInGuardrailSpec.model_json_schema()["$defs"]

    assert set(published["BackendType"]["enum"]) == {member.value for member in BackendType}
    assert set(published["GuardrailCategory"]["enum"]) == {member.value for member in UpstreamCategory}
    assert set(published["GuardrailStage"]["enum"]) == {member.value for member in UpstreamStage}
    assert set(published["OutputShape"]["enum"]) == {member.value for member in OutputShape}


def test_names_every_parameter_type_upstream_can_report() -> None:
    """The one taxonomy still spelled out here, so an upstream addition fails loudly.

    Unlike the four above it degrades rather than widening, to "json", so nothing
    else would report a member this gateway has never seen.
    """
    assert {member.value for member in UpstreamParameterType} <= _KNOWN_TYPES


def test_taxonomy_values_serialize_as_their_wire_strings() -> None:
    """An enum field must reach a client as the same string a Literal did."""
    spec = next(
        spec
        for spec in build_builtin_guardrail_catalog().guardrails
        if spec.guardrail_name == GuardrailName.LAKERA_GUARD
    )

    dumped = spec.model_dump(mode="json")

    assert dumped["backend"] == "hosted_api"
    assert dumped["primary_category"] == "prompt_injection"
    assert all(isinstance(stage, str) for stage in dumped["stages"])


def test_orders_the_taxonomy_lists_deterministically() -> None:
    """Upstream holds these as frozensets, whose iteration order a JSON artifact cannot inherit.

    Subclassing its model is what makes them sorted: the serializers doing it are
    upstream's, so nothing here sorts anything.
    """
    for spec in build_builtin_guardrail_catalog().guardrails:
        dumped = spec.model_dump(mode="json")

        for field in ("categories", "stages", "output_shapes", "alternate_backends"):
            assert dumped[field] == sorted(dumped[field]), field
        assert dumped["variant_licenses"] == sorted(dumped["variant_licenses"], key=lambda v: v["model_id"])
