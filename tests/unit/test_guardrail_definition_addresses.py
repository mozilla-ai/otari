"""The addresses an organization's guardrail definition may name, when it is saved and when it is built.

Three rules, each checked at both moments so a row written before the rule
existed is refused by the runner rather than dialed:

- a guardrail whose vendor library dials its own connection, and follows
  redirects, is reached only on the vendor's own hosts;
- a definition that names its own address carries its own credentials, since
  the operator's environment would otherwise fill the gap and send its key there;
- every address is public, as it always was at save.
"""

from __future__ import annotations

import ipaddress
import json
from datetime import UTC, datetime
from typing import Any

import pytest
from any_guardrail import AnyGuardrail

from gateway.exceptions.guardrails_exceptions import (
    OrganizationGuardrailDefinitionArgumentsError,
    OrganizationGuardrailDefinitionUnsafeUrlError,
)
from gateway.models.guardrails import OrganizationGuardrailDefinition
from gateway.services import url_safety
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.tenancy import organization_guardrail_runner as runner
from gateway.services.tenancy.organization_guardrail_definition_service import (
    _definable_spec,
    _validate_arguments,
    stored_arguments_problem,
)

PUBLIC = "93.184.216.34"


@pytest.fixture(autouse=True)
def _names_resolve_publicly(monkeypatch: pytest.MonkeyPatch) -> None:
    async def public(_host: str) -> list[object]:
        return [ipaddress.ip_address(PUBLIC)]

    monkeypatch.setattr(url_safety, "_resolve_all_async", public)


# Arguments these guardrails require whatever else a definition holds, none of them an address.
_REQUIRED: dict[str, dict[str, Any]] = {
    "patronus": {"evaluators": [{"evaluator": "lynx"}]},
    "alinia": {"detection_config": {"security": True}},
}


def _save(guardrail_name: str, arguments: dict[str, Any]) -> None:
    _validate_arguments(_definable_spec(guardrail_name), {**_REQUIRED.get(guardrail_name, {}), **arguments})


# --------------------------------------------------------------------------- #
# The vendor's own hosts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("guardrail_name", "endpoint"),
    [
        ("lakera_guard", "https://api.lakera.ai/v2/guard"),
        ("lakera_guard", "https://eu.api.lakera.ai/v2/guard"),
        ("lakera_guard", "https://ap-southeast-1.api.lakera.ai/v2/guard"),
        ("patronus", "https://api.patronus.ai/v1/evaluate"),
        ("azure_content_safety", "https://acme.cognitiveservices.azure.com"),
        ("azure_prompt_shields", "https://acme-eu.cognitiveservices.azure.com/"),
    ],
)
def test_an_endpoint_on_the_vendors_own_host_is_accepted(guardrail_name: str, endpoint: str) -> None:
    _save(guardrail_name, {"endpoint": endpoint, "api_key": "k"})


@pytest.mark.parametrize(
    ("guardrail_name", "endpoint"),
    [
        ("lakera_guard", "https://their-server.example/v2/guard"),
        ("lakera_guard", "https://api.lakera.ai.their-server.example/v2/guard"),
        ("lakera_guard", "https://other.api.lakera.ai/v2/guard"),
        ("patronus", "https://their-server.example/v1/evaluate"),
        ("azure_content_safety", "https://cognitiveservices.azure.com.their-server.example"),
        ("azure_content_safety", "https://theircognitiveservices.azure.com"),
        ("azure_prompt_shields", "https://their-server.example"),
        ("alinia", "https://api.alinia.ai/v1/detect"),
        ("alinia", "https://their-server.example/detect"),
    ],
)
def test_an_endpoint_off_the_vendors_hosts_is_refused(guardrail_name: str, endpoint: str) -> None:
    """Such a vendor library follows a redirect anywhere, so a public server of the tenant's could send it inside."""
    with pytest.raises(OrganizationGuardrailDefinitionUnsafeUrlError, match="endpoint"):
        _save(guardrail_name, {"endpoint": endpoint, "api_key": "k"})


def test_an_address_nested_in_an_argument_is_held_to_the_vendors_hosts_too() -> None:
    with pytest.raises(OrganizationGuardrailDefinitionUnsafeUrlError, match="detection_config"):
        _save("alinia", {"api_key": "k", "detection_config": {"callback": "https://their-server.example/"}})


def test_a_definition_naming_no_endpoint_reaches_the_vendor_the_operator_configured() -> None:
    _save("alinia", {"api_key": "k"})
    _save("lakera_guard", {})


# --------------------------------------------------------------------------- #
# The operator's credentials stay with the operator's endpoints
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("guardrail_name", "arguments", "needed"),
    [
        ("lakera_guard", {"endpoint": "https://api.lakera.ai/v2/guard"}, "api_key"),
        ("patronus", {"endpoint": "https://api.patronus.ai/v1/evaluate"}, "api_key"),
        ("azure_content_safety", {"endpoint": "https://acme.cognitiveservices.azure.com"}, "api_key"),
        ("openai_moderation", {"base_url": "https://moderation.example.com/v1"}, "api_key"),
        ("watsonx_guardian", {"url": "https://us-south.ml.cloud.ibm.com", "project_id": "p"}, "api_key"),
        ("watsonx_guardian", {"url": "https://us-south.ml.cloud.ibm.com", "api_key": "k"}, "project_id or space_id"),
    ],
)
def test_a_definition_naming_its_own_endpoint_needs_its_own_credentials(
    monkeypatch: pytest.MonkeyPatch, guardrail_name: str, arguments: dict[str, Any], needed: str
) -> None:
    """With the operator's variables set, as on a deployment that uses them for its own guardrails."""
    for variable in ("LAKERA_API_KEY", "PATRONUS_API_KEY", "CONTENT_SAFETY_KEY", "OPENAI_API_KEY", "WATSONX_APIKEY"):
        monkeypatch.setenv(variable, "the-operators-key")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "the-operators-project")

    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError, match=needed):
        _save(guardrail_name, arguments)


def test_watsonx_never_reads_the_operators_variables_for_a_definitions_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATSONX_APIKEY", "the-operators-key")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "the-operators-project")

    with pytest.raises(ValueError, match="its own api_key"):
        runner._watsonx_on_a_pinned_client({"url": "https://us-south.ml.cloud.ibm.com"})


# --------------------------------------------------------------------------- #
# A row saved before the rules is refused when built
# --------------------------------------------------------------------------- #


def _stored(monkeypatch: pytest.MonkeyPatch, guardrail_name: str, plain: dict[str, Any], **secrets: str) -> Any:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    return OrganizationGuardrailDefinition(
        name="legacy",
        guardrail_name=guardrail_name,
        create_kwargs=plain,
        encrypted_create_secrets=encrypt_secret(json.dumps(secrets, sort_keys=True)) if secrets else None,
        updated_at=datetime.now(UTC),
    )


@pytest.fixture
def no_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a refused row was built")

    monkeypatch.setattr(AnyGuardrail, "create", refuse)


@pytest.mark.parametrize(
    ("guardrail_name", "plain", "secrets"),
    [
        ("lakera_guard", {"endpoint": "http://127.0.0.1:8000/v2/guard"}, {"api_key": "k"}),
        ("lakera_guard", {"endpoint": "https://127.0.0.1/v2/guard"}, {"api_key": "k"}),
        ("patronus", {"endpoint": "https://their-server.example/v1/evaluate"}, {"api_key": "k"}),
        ("azure_prompt_shields", {"endpoint": "https://localhost:8443"}, {"api_key": "k"}),
        ("alinia", {"endpoint": "https://their-server.example/detect"}, {"api_key": "k"}),
        ("openai_moderation", {"base_url": "https://127.1/v1"}, {"api_key": "k"}),
        ("lakera_guard", {"endpoint": "https://api.lakera.ai/v2/guard"}, {}),
    ],
)
@pytest.mark.asyncio
async def test_a_row_stored_before_the_fix_is_refused_when_built(
    monkeypatch: pytest.MonkeyPatch, no_build: None, guardrail_name: str, plain: dict[str, Any], secrets: dict[str, str]
) -> None:
    definition = _stored(monkeypatch, guardrail_name, plain, **secrets)

    held = await runner._build(definition)

    assert held is not None
    assert held.guardrail is None


@pytest.mark.asyncio
async def test_a_stored_row_whose_name_resolves_inside_is_refused_when_built(
    monkeypatch: pytest.MonkeyPatch, no_build: None
) -> None:
    async def loopback(_host: str) -> list[object]:
        return [ipaddress.ip_address("127.0.0.1")]

    monkeypatch.setattr(url_safety, "_resolve_all_async", loopback)
    definition = _stored(
        monkeypatch, "openai_moderation", {"base_url": "https://moderation.example.com/v1"}, api_key="k"
    )

    held = await runner._build(definition)

    assert held is not None
    assert held.guardrail is None


@pytest.mark.asyncio
async def test_a_stored_row_whose_name_does_not_resolve_is_tried_again_rather_than_held_refused(
    monkeypatch: pytest.MonkeyPatch, no_build: None
) -> None:
    async def nothing(_host: str) -> list[object]:
        return []

    monkeypatch.setattr(url_safety, "_resolve_all_async", nothing)
    definition = _stored(
        monkeypatch, "openai_moderation", {"base_url": "https://moderation.example.com/v1"}, api_key="k"
    )

    assert await runner._build(definition) is None


@pytest.mark.asyncio
async def test_a_stored_row_within_the_rules_passes_the_check() -> None:
    assert (
        await stored_arguments_problem(
            "lakera_guard", {"endpoint": "https://eu.api.lakera.ai/v2/guard", "api_key": "k"}
        )
        is None
    )
    assert await stored_arguments_problem("lakera_guard", {}) is None


# --------------------------------------------------------------------------- #
# Only text is a credential
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("api_key", [False, 0, [], {"key": "k"}, 1.5])
@pytest.mark.parametrize(
    ("guardrail_name", "address"),
    [
        ("openai_moderation", {"base_url": "https://moderation.example.com/v1"}),
        ("lakera_guard", {"endpoint": "https://api.lakera.ai/v2/guard"}),
        ("azure_content_safety", {"endpoint": "https://acme.cognitiveservices.azure.com"}),
    ],
)
def test_a_credential_that_is_not_text_is_refused_at_save(
    monkeypatch: pytest.MonkeyPatch, guardrail_name: str, address: dict[str, Any], api_key: Any
) -> None:
    """A vendor reads ``api_key or os.getenv(...)``, so a falsy value would hand it the operator's key."""
    monkeypatch.setenv("OPENAI_API_KEY", "the-operators-key")

    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError, match="api_key"):
        _save(guardrail_name, {**address, "api_key": api_key})


@pytest.mark.parametrize("api_key", ["", " ", "\t"])
def test_a_blank_credential_beside_an_endpoint_is_refused(api_key: str) -> None:
    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError, match="names its own endpoint"):
        _save("openai_moderation", {"base_url": "https://moderation.example.com/v1", "api_key": api_key})


@pytest.mark.parametrize("api_key", [False, 0, [], " "])
@pytest.mark.asyncio
async def test_a_stored_row_whose_credential_is_not_text_is_refused_when_built(
    monkeypatch: pytest.MonkeyPatch, no_build: None, api_key: Any
) -> None:
    definition = _stored(
        monkeypatch, "openai_moderation", {"base_url": "https://moderation.example.com/v1", "api_key": api_key}
    )

    held = await runner._build(definition)

    assert held is not None
    assert held.guardrail is None
    assert held.refusal is not None
    assert "api_key" in held.refusal


# --------------------------------------------------------------------------- #
# An address two HTTP libraries could read differently
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("guardrail_name", "endpoint"),
    [
        ("azure_content_safety", "https://evil.example\\x.cognitiveservices.azure.com"),
        ("azure_prompt_shields", "https://evil.example\\@acme.cognitiveservices.azure.com"),
        ("lakera_guard", "https://evil.example\\@api.lakera.ai/v2/guard"),
        ("lakera_guard", "https://user@api.lakera.ai/v2/guard"),
        ("lakera_guard", "https://api.lakera.ai /v2/guard"),
        ("lakera_guard", "https://api.lakera.ai\t/v2/guard"),
        ("patronus", "https://api%2epatronus.ai/v1/evaluate"),
        ("patronus", "https://api_patronus.ai/v1/evaluate"),
    ],
)
def test_an_address_the_vendor_library_could_read_differently_is_refused_at_save(
    guardrail_name: str, endpoint: str
) -> None:
    with pytest.raises(OrganizationGuardrailDefinitionUnsafeUrlError, match="endpoint"):
        _save(guardrail_name, {"endpoint": endpoint, "api_key": "k"})


@pytest.mark.asyncio
async def test_an_ambiguous_stored_address_is_refused_when_built(
    monkeypatch: pytest.MonkeyPatch, no_build: None
) -> None:
    definition = _stored(
        monkeypatch,
        "azure_content_safety",
        {"endpoint": "https://evil.example\\x.cognitiveservices.azure.com"},
        api_key="k",
    )

    held = await runner._build(definition)

    assert held is not None
    assert held.guardrail is None
    assert held.refusal is not None
    assert "backslash" in held.refusal


@pytest.mark.parametrize("url", ["https://[::1/", "https://api.lakera.ai:99999/v2/guard", "https:///v2/guard"])
def test_a_malformed_address_is_refused_as_a_bad_request(url: str) -> None:
    """Not a ValueError out of the URL parser, which the API would answer with a 500."""
    with pytest.raises(OrganizationGuardrailDefinitionUnsafeUrlError):
        _save("lakera_guard", {"endpoint": url, "api_key": "k"})


def test_an_internationalized_name_is_read_in_its_encoded_form() -> None:
    _save("openai_moderation", {"base_url": "https://bücher.example/v1", "api_key": "k"})


# --------------------------------------------------------------------------- #
# The refusal reaches the admin, not only the log
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_refused_build_says_why_in_the_definitions_build_state(
    monkeypatch: pytest.MonkeyPatch, no_build: None
) -> None:
    import uuid

    from gateway.services.tenancy.organization_guardrail_definition_service import (
        OrganizationGuardrailDefinitionPublic,
    )

    definition = _stored(
        monkeypatch, "lakera_guard", {"endpoint": "https://their-server.example/v2/guard"}, api_key="k"
    )
    definition.id = uuid.uuid4()
    definition.organization_id = uuid.uuid4()
    definition.created_at = definition.updated_at
    definition.enabled = True
    held = await runner._build(definition)
    assert held is not None
    monkeypatch.setitem(runner._held, (definition.organization_id, definition.id), held)

    refusal = runner.build_refusal(definition.organization_id, definition.id, definition.updated_at)
    public = OrganizationGuardrailDefinitionPublic.from_model(
        definition, secret_names=["api_key"], secrets_decryptable=True, build_state="failed", build_refusal=refusal
    )

    assert refusal is not None
    assert "vendor's own hosts" in refusal
    assert public.build_refusal == refusal
    assert "k" not in refusal.split()
