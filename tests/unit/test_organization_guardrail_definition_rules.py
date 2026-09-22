"""The rules a stored guardrail definition has to pass, and the plain/secret split.

Almost every rule is read off `build_builtin_guardrail_catalog`, so most cases
here name a real guardrail and assert against the installed any-guardrail rather
than a fixture. The two that are not are the Bedrock key pair and the address
check, which reads the value of an argument because upstream types every URL as
a plain string. Two rules refuse nothing today, and those are exercised against a
hand-built spec: the point of writing them was the release that arrives with the
shape, so a test that only asserted "nothing is refused" would pass with the rule
deleted.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from any_guardrail.parameters import RequirementGroup

from gateway.models.guardrails import OrganizationGuardrailDefinition
from gateway.services.guardrail_catalog import (
    BuiltInGuardrailSpec,
    GuardrailParameterSpec,
    build_builtin_guardrail_catalog,
    builtin_guardrail_spec,
)
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.tenancy.errors import (
    OrganizationGuardrailDefinitionArgumentsError,
    OrganizationGuardrailDefinitionUnsafeUrlError,
    OrganizationGuardrailNotBuildableError,
    OrganizationGuardrailNotDefinableError,
)
from gateway.services.tenancy.organization_guardrail_definition_service import (
    OrganizationGuardrailDefinitionUpdate,
    _arguments_after,
    _definable_spec,
    _dialable_urls,
    _refuse_unconfigurable,
    _split_by_secret_flag,
    _validate_argument_urls,
    _validate_arguments,
    build_arguments,
)


def _spec(guardrail_name: str) -> BuiltInGuardrailSpec:
    spec = builtin_guardrail_spec(guardrail_name)
    assert spec is not None, f"{guardrail_name} is expected to be in the built-in catalog"
    return spec


def _synthetic(
    *,
    guardrail_name: str = "invented",
    create: list[GuardrailParameterSpec] | None = None,
    validate: list[GuardrailParameterSpec] | None = None,
    groups: list[RequirementGroup] | None = None,
) -> BuiltInGuardrailSpec:
    """A catalog row for a guardrail any-guardrail does not have, yet.

    Copied from a real row and overridden, rather than built from a literal, so
    a field upstream adds to its metadata does not have to be invented here.
    Only the parameter lists and the groups reach the rules under test.
    """
    return _spec("lakera_guard").model_copy(
        update={
            "guardrail_name": guardrail_name,
            "create_parameters": create or [],
            "validate_parameters": validate or [],
            "requirement_groups": groups or [],
        }
    )


def _parameter(name: str, **overrides: Any) -> GuardrailParameterSpec:
    fields: dict[str, Any] = {"name": name, "type": "string", "required": False}
    fields.update(overrides)
    return GuardrailParameterSpec(**fields)


# --------------------------------------------------------------------------- #
# Which guardrails may be defined at all
# --------------------------------------------------------------------------- #


def test_refuses_a_guardrail_this_gateway_cannot_build() -> None:
    """One answer for a name the registry never heard of and for one that holds model weights.

    A caller can do nothing different with either, and both come back from the
    catalog lookup as the same ``None``.
    """
    for guardrail_name in ("not_a_guardrail", "", "susfactor"):
        with pytest.raises(OrganizationGuardrailNotBuildableError):
            _definable_spec(guardrail_name)


def test_refuses_any_llm_although_the_catalog_lists_it() -> None:
    """The billing rule, and a separate refusal from the one above on purpose.

    `any_llm` is buildable and would run on whatever LLM key the process holds,
    outside the budget the request reserved. The message has to say so, because
    "not available" would send an admin looking for a missing dependency.
    """
    with pytest.raises(OrganizationGuardrailNotDefinableError) as refused:
        _definable_spec("any_llm")

    assert "metering" in str(refused.value)


def test_accepts_every_other_guardrail_the_catalog_lists() -> None:
    """Eight of the nine listed, and the store's own list is the catalog's."""
    listed = [spec.guardrail_name for spec in build_builtin_guardrail_catalog().guardrails]

    definable = [name for name in listed if name != "any_llm"]
    assert len(definable) == 8
    for guardrail_name in definable:
        assert _definable_spec(guardrail_name).guardrail_name == guardrail_name


def test_refuses_a_guardrail_whose_required_argument_cannot_be_written_down() -> None:
    """Empty against the installed library, which is why it is written against an invented one.

    A required live object has no field to arrive in, so no stored row could ever
    describe such a guardrail. Refused at the name rather than per argument: a
    caller cannot fix it by sending something else.
    """
    # ``storable`` is spelled out because the catalog derives it while reading
    # upstream's registry, and this spec never went through that.
    spec = _synthetic(create=[_parameter("session", type="json", required=True, secret=True, storable=False)])

    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError) as refused:
        _refuse_unconfigurable(spec)

    assert "session" in str(refused.value)
    assert "live object" in str(refused.value)


def test_refuses_a_guardrail_that_needs_an_argument_on_every_check() -> None:
    """The other empty arm. A definition stores what a guardrail is *built* with.

    `any_llm.policy` is this shape today, and it never reaches here because the
    billing rule refuses it first. Written anyway, so the next guardrail with a
    required per-call argument cannot be stored as a row that fails at call time.
    """
    spec = _synthetic(validate=[_parameter("policy", required=True)])

    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError) as refused:
        _refuse_unconfigurable(spec)

    assert "policy" in str(refused.value)

    assert [parameter.name for parameter in _spec("any_llm").validate_parameters if parameter.required] == ["policy"]


# --------------------------------------------------------------------------- #
# The build arguments
# --------------------------------------------------------------------------- #


def test_refuses_an_argument_the_guardrail_does_not_declare() -> None:
    """No guardrail takes extra keyword arguments, so this would fail at build time anyway.

    Refusing it here is what keeps a credential under an unexpected name from
    reaching a column, the split having no flag to read for a name no spec
    declares.
    """
    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError) as refused:
        _validate_arguments(_spec("lakera_guard"), {"api_key": "k", "endpiont": "https://api.lakera.ai"})

    assert "endpiont" in str(refused.value)


def test_refuses_an_argument_that_is_a_live_object_and_names_the_path_that_works() -> None:
    """Both of them, and each message read off the catalog rather than written per guardrail.

    Bedrock's session and watsonx's client are authenticated objects, not
    configuration. The alternatives come from the requirement groups where there
    are any, and from the guardrail's own storable secrets where there are not.
    """
    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError) as bedrock:
        _validate_arguments(
            _spec("bedrock_guardrails"),
            {"guardrail_identifier": "gr-1", "boto3_session": {"profile": "default"}},
        )

    assert "boto3_session" in str(bedrock.value)
    assert "aws_access_key_id, aws_secret_access_key" in str(bedrock.value)

    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError) as watsonx:
        _validate_arguments(_spec("watsonx_guardian"), {"api_client": {"token": "t"}})

    assert "api_client" in str(watsonx.value)
    assert "api_key, project_id, space_id, url" in str(watsonx.value)


def test_refuses_a_required_argument_no_environment_variable_can_supply() -> None:
    """Three of these exist today, one per guardrail that has one."""
    required_without_env = {
        "alinia": "detection_config",
        "bedrock_guardrails": "guardrail_identifier",
        "patronus": "evaluators",
    }

    for guardrail_name, parameter_name in required_without_env.items():
        spec = _spec(guardrail_name)
        assert parameter_name in {
            parameter.name
            for parameter in spec.create_parameters
            if parameter.required and not parameter.env_var and parameter.storable
        }

        with pytest.raises(OrganizationGuardrailDefinitionArgumentsError) as refused:
            _validate_arguments(spec, {})

        assert parameter_name in str(refused.value)


def test_accepts_a_required_credential_an_environment_variable_can_supply() -> None:
    """The carve-out, and the reason rule 4 cannot read `required` alone.

    `required` folds in upstream's effectively-required flag, so Lakera's key
    reads required although `LAKERA_API_KEY` supplies it. Demanding it would
    refuse a deployment that keeps the credential in its environment, which is
    five of the eight definable guardrails.
    """
    key_from_the_environment = _spec("lakera_guard").create_parameters[0]
    assert key_from_the_environment.name == "api_key"
    assert key_from_the_environment.required
    assert key_from_the_environment.env_var == "LAKERA_API_KEY"

    _validate_arguments(_spec("lakera_guard"), {})


def test_never_asks_whether_an_environment_variable_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """The process that writes a row is not the one that builds the guardrail.

    The row may outlive both, so an answer that moved with this process's
    environment would be a rule about the wrong host. The catalog withholds the
    same fact for the same reason.
    """
    monkeypatch.delenv("LAKERA_API_KEY", raising=False)
    _validate_arguments(_spec("lakera_guard"), {})

    monkeypatch.setenv("LAKERA_API_KEY", "set-on-this-host")
    _validate_arguments(_spec("lakera_guard"), {})

    monkeypatch.delenv("ALINIA_API_KEY", raising=False)
    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError):
        _validate_arguments(_spec("alinia"), {})

    # Still refused, and for `detection_config`, which no variable supplies
    # either way. Setting the key changes nothing about the answer.
    monkeypatch.setenv("ALINIA_API_KEY", "set-on-this-host")
    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError) as refused:
        _validate_arguments(_spec("alinia"), {})
    assert "detection_config" in str(refused.value)


def test_refuses_a_requirement_group_nothing_satisfies() -> None:
    """Refuses nothing today: every group upstream declares carries environment variables."""
    spec = _synthetic(
        create=[_parameter("primary"), _parameter("secondary")],
        groups=[RequirementGroup(description="Supply primary or secondary.", parameters=("primary", "secondary"))],
    )

    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError) as refused:
        _validate_arguments(spec, {})
    assert "Supply primary or secondary." in str(refused.value)

    _validate_arguments(spec, {"secondary": "value"})


def test_leaves_a_requirement_group_an_environment_variable_can_satisfy() -> None:
    """Watsonx, whose three groups are the only ones that exist.

    Each offers the unstorable `api_client` as its alternative, so the path that
    works is the other members. Whether the variables are set is not this
    process's business, which is what makes the row acceptable with nothing sent.
    """
    spec = _spec("watsonx_guardian")
    assert [sorted(group.parameters) for group in spec.requirement_groups] == [
        ["api_client", "api_key"],
        ["api_client", "url"],
        ["api_client", "project_id", "space_id"],
    ]
    assert all(group.env_vars for group in spec.requirement_groups)

    _validate_arguments(spec, {})
    _validate_arguments(spec, {"api_key": "k", "url": "https://eu-de.ml.cloud.ibm.com", "project_id": "p"})


def test_refuses_a_bedrock_definition_without_both_aws_keys() -> None:
    """The one rule not read off the catalog, because upstream marks both keys optional.

    With neither key boto3 falls back to the instance role of the host this
    gateway runs on, so the organization's guardrail would spend the operator's
    AWS account. One key alone falls back the same way.
    """
    spec = _spec("bedrock_guardrails")
    assert not any(parameter.required for parameter in spec.create_parameters if parameter.name.startswith("aws_")), (
        "upstream marks them optional, which is why the rule is written out here"
    )

    for arguments in (
        {"guardrail_identifier": "gr-1"},
        {"guardrail_identifier": "gr-1", "aws_access_key_id": "AKIA"},
        {"guardrail_identifier": "gr-1", "aws_secret_access_key": "secret"},
        {"guardrail_identifier": "gr-1", "aws_access_key_id": "", "aws_secret_access_key": "secret"},
    ):
        with pytest.raises(OrganizationGuardrailDefinitionArgumentsError) as refused:
            _validate_arguments(spec, arguments)
        assert "host this gateway runs on" in str(refused.value)

    _validate_arguments(
        spec,
        {"guardrail_identifier": "gr-1", "aws_access_key_id": "AKIA", "aws_secret_access_key": "secret"},
    )


# --------------------------------------------------------------------------- #
# The split
# --------------------------------------------------------------------------- #


def test_splits_the_arguments_by_the_catalogs_flag_and_not_by_the_name() -> None:
    """Inverted on purpose, because every real secret is also named like one.

    `models/secret_fields` has to guess from a name, having no schema for a
    free-form settings dict. Here the guardrail's own schema says which argument
    is a credential, so a vendor key cannot reach the plain column by being named
    unusually and an endpoint cannot be hidden by being named `api_key`.
    """
    spec = _synthetic(
        create=[_parameter("endpoint", secret=True), _parameter("api_key")],
    )

    plain, secrets = _split_by_secret_flag(spec, {"endpoint": "https://vendor.example", "api_key": "not-a-secret"})

    assert plain == {"api_key": "not-a-secret"}
    assert secrets == {"endpoint": "https://vendor.example"}


def test_splits_a_real_guardrails_arguments() -> None:
    """Lakera, whose key is secret and whose endpoint and project id are not."""
    arguments = {"api_key": "lakera-key", "endpoint": "https://api.lakera.ai", "project_id": "proj"}

    plain, secrets = _split_by_secret_flag(_spec("lakera_guard"), arguments)

    assert plain == {"endpoint": "https://api.lakera.ai", "project_id": "proj"}
    assert secrets == {"api_key": "lakera-key"}


# --------------------------------------------------------------------------- #
# What an edit does to a secret the caller was never shown
# --------------------------------------------------------------------------- #


def _stored(monkeypatch: pytest.MonkeyPatch, **secrets: str) -> OrganizationGuardrailDefinition:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    return OrganizationGuardrailDefinition(
        name="prod-lakera",
        guardrail_name="lakera_guard",
        create_kwargs={"endpoint": "https://api.lakera.ai"},
        encrypted_create_secrets=encrypt_secret(json.dumps(secrets, sort_keys=True)) if secrets else None,
    )


def test_an_edit_that_mentions_no_arguments_reads_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """The case that lets a broken row be repaired rather than only deleted.

    An admin on a deployment whose `OTARI_SECRET_KEY` has moved can still turn
    the guardrail off. Proven by moving the key out from under the row: the
    answer is the same, because nothing decrypts.
    """
    definition = _stored(monkeypatch, api_key="lakera-key")
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    request = OrganizationGuardrailDefinitionUpdate(enabled=False)

    assert _arguments_after(definition, request, "lakera_guard") is None


def test_changing_the_guardrail_re_splits_the_stored_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plain/secret split is the old class's answer, so it cannot be carried over.

    A name plain under one guardrail can be secret under the next. Merging the
    two stored halves and re-splitting is what keeps the columns honest, and it
    is the one edit that has to decrypt without being asked to.
    """
    definition = _stored(monkeypatch, api_key="lakera-key")
    request = OrganizationGuardrailDefinitionUpdate(guardrail_name="openai_moderation")

    assert _arguments_after(definition, request, "openai_moderation") == {
        "endpoint": "https://api.lakera.ai",
        "api_key": "lakera-key",
    }


def test_arguments_sent_without_a_mask_are_taken_as_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """And still decrypt nothing, which is how an unreadable row is repaired.

    The stored secrets are unreadable here, the key having moved. A caller who
    retypes the credential must succeed anyway.
    """
    definition = _stored(monkeypatch, api_key="lakera-key")
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    request = OrganizationGuardrailDefinitionUpdate(
        create_kwargs={"api_key": "retyped", "endpoint": "https://eu.example"}
    )

    assert _arguments_after(definition, request, "lakera_guard") == {
        "api_key": "retyped",
        "endpoint": "https://eu.example",
    }


def test_a_masked_argument_keeps_the_stored_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """What a form that never saw the credential sends back when it edits the endpoint."""
    definition = _stored(monkeypatch, api_key="lakera-key")
    request = OrganizationGuardrailDefinitionUpdate(
        create_kwargs={"api_key": "***", "endpoint": "https://eu.api.lakera.ai"}
    )

    assert _arguments_after(definition, request, "lakera_guard") == {
        "api_key": "lakera-key",
        "endpoint": "https://eu.api.lakera.ai",
    }


def test_a_secret_left_out_of_a_sent_map_is_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    """``create_kwargs`` replaces the arguments whole, so an omission is a decision.

    The alternative is a credential that cannot be removed without deleting the
    definition, and the mask is what makes the deliberate keep available.
    """
    definition = _stored(monkeypatch, api_key="lakera-key")
    request = OrganizationGuardrailDefinitionUpdate(create_kwargs={"endpoint": "https://api.lakera.ai", "x": "***"})

    assert _arguments_after(definition, request, "lakera_guard") == {
        "endpoint": "https://api.lakera.ai",
        "x": "***",
    }, "a mask under a name nothing is stored under stays what the caller sent, and rule 3 refuses it next"


def test_an_unreadable_secret_map_asks_for_the_credentials_rather_than_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rename case is the one that must decrypt, so it is the one that can refuse.

    A 400 naming the way out, not a 500: the caller fixes it by sending
    ``create_kwargs`` with the credentials in it.
    """
    definition = _stored(monkeypatch, api_key="lakera-key")
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    request = OrganizationGuardrailDefinitionUpdate(guardrail_name="openai_moderation")

    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError) as refused:
        _arguments_after(definition, request, "openai_moderation")

    assert "OTARI_SECRET_KEY" in str(refused.value)
    assert "create_kwargs" in str(refused.value)


# --------------------------------------------------------------------------- #
# The endpoints a definition may name
#
# Every address here is an IP literal, in the public range example.com has used
# for years, so the check under test runs no resolver: a suite that needs DNS to
# agree with it fails for reasons that have nothing to do with the rule.
# --------------------------------------------------------------------------- #


def test_an_address_is_recognized_by_its_value_wherever_it_sits() -> None:
    """Nested too, because two real build arguments are containers.

    `alinia.detection_config` is a dict and `patronus.evaluators` a list, so a
    scan of the top level alone would walk past an address inside either.
    """
    found = dict(
        _dialable_urls(
            {
                "endpoint": "https://93.184.216.34",
                "detection_config": {"sink": "https://93.184.216.35/report"},
                "evaluators": [{"callback": "http://93.184.216.36"}],
                "api_key": "lakera-key",
                "project_id": "proj-1",
                "retries": 3,
                "enabled": True,
            }
        )
    )

    assert found == {
        "endpoint": "https://93.184.216.34",
        "detection_config.sink": "https://93.184.216.35/report",
        "evaluators[0].callback": "http://93.184.216.36",
    }


def test_a_value_with_no_scheme_in_it_is_not_an_address() -> None:
    """Including the mask, which is why the update path needs no special case."""
    assert list(_dialable_urls({"api_key": "***", "project_id": "proj-1", "space_id": ""})) == []


@pytest.mark.asyncio
async def test_accepts_a_public_endpoint() -> None:
    await _validate_argument_urls({"endpoint": "https://93.184.216.34/v2", "api_key": "lakera-key"})


@pytest.mark.asyncio
async def test_refuses_an_endpoint_inside_the_network_this_gateway_runs_in() -> None:
    with pytest.raises(OrganizationGuardrailDefinitionUnsafeUrlError) as refused:
        await _validate_argument_urls({"endpoint": "https://169.254.169.254/latest/meta-data/"})

    assert "endpoint" in str(refused.value)
    assert "link-local" in str(refused.value)


@pytest.mark.asyncio
async def test_refuses_an_endpoint_that_would_carry_a_credential_in_clear() -> None:
    with pytest.raises(OrganizationGuardrailDefinitionUnsafeUrlError) as refused:
        await _validate_argument_urls({"endpoint": "http://93.184.216.34/v2"})

    assert "https" in str(refused.value)


@pytest.mark.asyncio
async def test_refuses_a_scheme_this_gateway_would_not_dial() -> None:
    with pytest.raises(OrganizationGuardrailDefinitionUnsafeUrlError):
        await _validate_argument_urls({"endpoint": "file://etc/passwd"})


@pytest.mark.asyncio
async def test_a_refusal_names_the_nested_argument_it_came_from() -> None:
    """The path, not just the top-level name, because that is what a form has to fix."""
    with pytest.raises(OrganizationGuardrailDefinitionUnsafeUrlError) as refused:
        await _validate_argument_urls({"detection_config": {"sink": "https://10.0.0.5/collect"}})

    assert "detection_config.sink" in str(refused.value)


@pytest.mark.asyncio
async def test_a_refusal_carries_the_host_and_not_the_url() -> None:
    """An endpoint can hold a credential in its userinfo, and the answer is public."""
    with pytest.raises(OrganizationGuardrailDefinitionUnsafeUrlError) as refused:
        await _validate_argument_urls({"endpoint": "https://user:pa55word@10.0.0.5/collect"})

    assert "pa55word" not in str(refused.value)
    assert "10.0.0.5" in str(refused.value)


@pytest.mark.asyncio
async def test_the_operators_own_override_reaches_this_check_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """One flag governs every caller of the shared check, and that is worth pinning."""
    monkeypatch.setenv("OTARI_MCP_ALLOW_PRIVATE_HOSTS", "true")
    await _validate_argument_urls({"endpoint": "https://10.0.0.5/collect"})


# --------------------------------------------------------------------------- #
# The arguments a build would be handed
# --------------------------------------------------------------------------- #


def test_the_build_arguments_are_the_two_halves_put_back_together(monkeypatch: pytest.MonkeyPatch) -> None:
    """What `AnyGuardrail.create` takes, which is the whole row and not either column.

    The split exists so a credential never reaches the plain column. A build
    needs both halves, so exactly one function undoes it, and the runner calls
    that one rather than decrypting a second time on its own.
    """
    definition = _stored(monkeypatch, api_key="lakera-key")

    assert build_arguments(definition) == {"endpoint": "https://api.lakera.ai", "api_key": "lakera-key"}


def test_a_definition_with_no_secrets_still_yields_its_plain_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A row with nothing encrypted must not need a readable key to be built."""
    definition = _stored(monkeypatch)
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)

    assert build_arguments(definition) == {"endpoint": "https://api.lakera.ai"}
