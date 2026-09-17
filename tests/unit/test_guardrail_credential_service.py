"""The rules a stored guardrail definition is held to, and the secret split.

Everything here is pure: the catalog is a property of the installed
``any_guardrail`` and needs no database and no network. The session-bound half of
the service is covered by ``tests/integration/test_guardrail_credentials_api.py``.

The guardrails named below are chosen for the shape each one proves, not as a
list worth keeping in step with upstream: Lakera for one env-backed secret,
Bedrock for several secrets and an unstorable one, watsonx for requirement
groups, Alinia for a required argument nothing else can supply, and AnyLlm for a
guardrail with no constructor arguments at all.
"""

import sys

import pytest

from gateway.exceptions.guardrail_credentials import (
    MissingGuardrailParameterError,
    UnknownGuardrailError,
    UnknownGuardrailParameterError,
    UnstorableGuardrailParameterError,
)
from gateway.models.guardrails import GuardrailCredential
from gateway.services.guardrail_credential_service import (
    decrypt_create_secrets,
    definition_from_row,
    split_create_kwargs,
    stored_guardrail_config,
    stored_secret_names,
    validate_guardrail_kwargs,
)
from gateway.services.secret_box import SecretDecryptionError, encrypt_secret, generate_secret_key

_LAKERA = {"api_key": "lak-secret", "endpoint": "https://api.lakera.ai/v2/guard"}


def test_a_secret_and_a_plain_argument_go_to_different_halves() -> None:
    """The split is the catalog's ``secret`` flag and nothing else."""
    plain, secrets = split_create_kwargs("lakera_guard", _LAKERA)

    assert plain == {"endpoint": "https://api.lakera.ai/v2/guard"}
    assert secrets == {"api_key": "lak-secret"}


def test_several_secrets_share_one_map() -> None:
    """Bedrock has two storable ones; nothing about the shape is per-guardrail."""
    plain, secrets = split_create_kwargs(
        "bedrock_guardrails",
        {
            "guardrail_identifier": "gr-1",
            "aws_access_key_id": "AKIA",
            "aws_secret_access_key": "shh",
        },
    )

    assert plain == {"guardrail_identifier": "gr-1"}
    assert secrets == {"aws_access_key_id": "AKIA", "aws_secret_access_key": "shh"}


def test_a_guardrail_with_no_constructor_arguments_splits_to_nothing() -> None:
    assert split_create_kwargs("any_llm", {}) == ({}, {})


def test_a_guardrail_the_catalog_does_not_offer_is_refused() -> None:
    with pytest.raises(UnknownGuardrailError):
        validate_guardrail_kwargs("not_a_guardrail", create_kwargs={}, validate_kwargs={})


def test_a_guardrail_that_would_load_model_weights_is_refused() -> None:
    """``llama_guard`` is a real any-guardrail class and still not one this stores.

    It runs by holding weights in the process, so the catalog does not offer it
    and the store must not accept it either. Refusing through the same catalog
    lookup is what keeps the two from disagreeing.
    """
    with pytest.raises(UnknownGuardrailError):
        validate_guardrail_kwargs("llama_guard", create_kwargs={}, validate_kwargs={})


def test_a_guardrail_whose_hosted_path_needs_a_live_object_is_refused() -> None:
    """``susfactor`` names a hosted alternate that a stored definition cannot select.

    Reaching it means passing a ``provider=`` object, which is not an argument
    upstream publishes, so what this would build is the local encoder.
    """
    with pytest.raises(UnknownGuardrailError):
        validate_guardrail_kwargs("susfactor", create_kwargs={}, validate_kwargs={})


def test_an_argument_the_guardrail_does_not_take_is_refused() -> None:
    """No guardrail accepts ``**kwargs``, so this would fail when it was built."""
    with pytest.raises(UnknownGuardrailParameterError, match="no create argument"):
        validate_guardrail_kwargs("lakera_guard", create_kwargs={**_LAKERA, "nope": 1}, validate_kwargs={})


def test_an_unknown_per_call_argument_is_refused_too() -> None:
    with pytest.raises(UnknownGuardrailParameterError, match="no validate argument"):
        validate_guardrail_kwargs("patronus", create_kwargs={"evaluators": []}, validate_kwargs={"nope": 1})


def test_a_live_object_cannot_be_stored() -> None:
    """``boto3_session`` is a real Bedrock argument, so only the storable flag stops it."""
    with pytest.raises(UnstorableGuardrailParameterError) as caught:
        validate_guardrail_kwargs(
            "bedrock_guardrails",
            create_kwargs={"guardrail_identifier": "gr-1", "boto3_session": {}},
            validate_kwargs={},
        )

    assert "aws_access_key_id" in str(caught.value)
    assert "aws_secret_access_key" in str(caught.value)


def test_the_other_live_object_is_refused_and_its_neighbor_is_not() -> None:
    with pytest.raises(UnstorableGuardrailParameterError):
        validate_guardrail_kwargs("watsonx_guardian", create_kwargs={"api_client": {}}, validate_kwargs={})

    validate_guardrail_kwargs("watsonx_guardian", create_kwargs={"api_key": "k"}, validate_kwargs={})


def test_a_required_argument_nothing_else_supplies_must_be_given() -> None:
    """Alinia's ``detection_config`` is required and names no environment variable."""
    with pytest.raises(MissingGuardrailParameterError, match="detection_config"):
        validate_guardrail_kwargs(
            "alinia",
            create_kwargs={"api_key": "k", "endpoint": "https://example.invalid"},
            validate_kwargs={},
        )


def test_a_null_does_not_satisfy_a_required_argument() -> None:
    """A key is not a value. ``{"detection_config": null}`` would build nothing."""
    with pytest.raises(MissingGuardrailParameterError, match="detection_config"):
        validate_guardrail_kwargs(
            "alinia",
            create_kwargs={"api_key": "k", "endpoint": "https://example.invalid", "detection_config": None},
            validate_kwargs={},
        )


def test_a_falsy_value_does_satisfy_a_required_argument() -> None:
    """Only ``None`` counts as missing: zero and false are values a guardrail may take."""
    validate_guardrail_kwargs(
        "openai_moderation",
        create_kwargs={"api_key": "sk-x", "threshold": 0},
        validate_kwargs={},
    )


def test_a_required_argument_an_environment_variable_supplies_may_be_omitted() -> None:
    """``lakera_guard.api_key`` reads required because the catalog folds in
    upstream's effectively-required flag, but ``LAKERA_API_KEY`` can supply it.

    Demanding it would refuse a legitimate row on a deployment that sets the
    variable, which is why the rule carves out a parameter that names one. The
    store does not look at the environment to decide: whether the variable is
    set belongs to the process that builds the guardrail, not to this one.
    """
    validate_guardrail_kwargs("lakera_guard", create_kwargs={}, validate_kwargs={})


def test_a_requirement_group_that_names_environment_variables_is_not_enforced() -> None:
    """All three of watsonx's groups name one, so an empty row is accepted."""
    validate_guardrail_kwargs("watsonx_guardian", create_kwargs={}, validate_kwargs={})


def test_a_valid_definition_passes_every_rule() -> None:
    validate_guardrail_kwargs("lakera_guard", create_kwargs=_LAKERA, validate_kwargs={})
    validate_guardrail_kwargs(
        "openai_moderation",
        create_kwargs={"api_key": "sk-x", "threshold": 0.7},
        validate_kwargs={},
    )


def test_a_row_with_no_secret_map_decrypts_to_nothing() -> None:
    row = GuardrailCredential(name="n", guardrail_name="any_llm", create_kwargs={}, validate_kwargs={})

    assert decrypt_create_secrets(row) == {}
    assert stored_secret_names(row) == (frozenset(), True)


def test_a_stored_map_decrypts_to_its_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    row = GuardrailCredential(
        name="n",
        guardrail_name="lakera_guard",
        create_kwargs={},
        validate_kwargs={},
        encrypted_create_secrets=encrypt_secret('{"api_key": "lak-secret"}'),
    )

    assert decrypt_create_secrets(row) == {"api_key": "lak-secret"}
    assert stored_secret_names(row) == (frozenset({"api_key"}), True)


def test_an_unreadable_map_costs_the_names_and_not_the_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong key must not take the whole list down, so the row reports no names."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    ciphertext = encrypt_secret('{"api_key": "lak-secret"}')
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    row = GuardrailCredential(
        name="n",
        guardrail_name="lakera_guard",
        create_kwargs={},
        validate_kwargs={},
        encrypted_create_secrets=ciphertext,
    )

    assert stored_secret_names(row) == (frozenset(), False)


def test_a_definition_puts_the_two_halves_of_a_row_back_together(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guardrail being built knows nothing about where its API key was kept."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    row = GuardrailCredential(
        name="prompt-injection",
        guardrail_name="lakera_guard",
        create_kwargs={"endpoint": "https://api.lakera.ai/v2/guard"},
        validate_kwargs={"breakdown": True},
        encrypted_create_secrets=encrypt_secret('{"api_key": "lak-secret"}'),
    )

    definition = definition_from_row(row)

    assert definition.guardrail_name == "lakera_guard"
    assert definition.create_kwargs == {
        "endpoint": "https://api.lakera.ai/v2/guard",
        "api_key": "lak-secret",
    }
    assert definition.validate_kwargs == {"breakdown": True}


def test_a_stored_secret_wins_over_a_plain_key_of_the_same_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plain half cannot shadow a credential, however it came to hold that name.

    A row rewritten under a guardrail that classifies the argument differently
    could leave both halves carrying it, and the encrypted one is the real value.
    """
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    row = GuardrailCredential(
        name="n",
        guardrail_name="lakera_guard",
        create_kwargs={"api_key": "stale-plain-copy"},
        validate_kwargs={},
        encrypted_create_secrets=encrypt_secret('{"api_key": "lak-secret"}'),
    )

    assert definition_from_row(row).create_kwargs == {"api_key": "lak-secret"}


def test_a_definition_from_a_row_with_no_secrets_is_its_plain_half() -> None:
    row = GuardrailCredential(
        name="judge", guardrail_name="any_llm", create_kwargs={"model_id": "gpt-4o"}, validate_kwargs={}
    )

    assert definition_from_row(row).create_kwargs == {"model_id": "gpt-4o"}


def test_a_definition_refuses_a_row_whose_secrets_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the plain half alone would be a client with no API key."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    ciphertext = encrypt_secret('{"api_key": "lak-secret"}')
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    row = GuardrailCredential(
        name="n",
        guardrail_name="lakera_guard",
        create_kwargs={},
        validate_kwargs={},
        encrypted_create_secrets=ciphertext,
    )

    with pytest.raises(SecretDecryptionError):
        definition_from_row(row)


def test_storing_a_guardrail_never_loads_a_model_backend() -> None:
    """The catalog reads an import-free registry, and this module must not widen that."""
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def _row(**kwargs: object) -> GuardrailCredential:
    """A stored row with only the columns the config is built from set."""
    defaults: dict[str, object] = {
        "name": "prompt-injection",
        "guardrail_name": "lakera_guard",
        "create_kwargs": {},
        "validate_kwargs": {},
        "enabled": True,
        "mode": "block",
        "on_unavailable": "block",
        "applies_to_all_workspaces": False,
    }
    return GuardrailCredential(**(defaults | kwargs))


def test_the_row_is_the_profile_a_request_is_checked_against() -> None:
    """``name`` is the profile, and the row's per-call arguments travel with it."""
    config = stored_guardrail_config(_row(validate_kwargs={"threshold": 0.8}))

    assert config.profile == "prompt-injection"
    assert config.validate_kwargs == {"threshold": 0.8}
    # Never an endpoint: a stored definition runs in this process, and a URL here
    # would send it to a sidecar instead.
    assert config.url is None


def test_allow_becomes_the_legacy_monitor() -> None:
    """The one place the two vocabularies meet."""
    assert stored_guardrail_config(_row(on_unavailable="allow")).on_unavailable == "monitor"


def test_block_survives_the_translation() -> None:
    assert stored_guardrail_config(_row(on_unavailable="block")).on_unavailable == "block"


@pytest.mark.parametrize("column", ["mode", "on_unavailable"])
def test_a_value_written_around_the_schema_resolves_to_block(column: str) -> None:
    """Fail closed, as ``organization_guardrail_service._stored_mode`` does.

    Both columns are plain strings whose only writers are ``Literal`` fields, so
    this is unreachable through the API. It resolves to the enforcing side
    because the alternative is a security control that silently stops enforcing.
    """
    config = stored_guardrail_config(_row(**{column: "nonsense"}))

    assert getattr(config, column) == "block"


def test_a_monitor_definition_reports_rather_than_refuses() -> None:
    assert stored_guardrail_config(_row(mode="monitor")).mode == "monitor"
