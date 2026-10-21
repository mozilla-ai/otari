"""Unit tests for named provider-instance resolution (mozilla-ai/otari#213)."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
import yaml
from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import AnyLLMError, MissingApiKeyError

from gateway.api.routes.pricing import _candidate_model_keys
from gateway.core.config import (
    KEYLESS_SELF_HOSTED_PROVIDERS,
    GatewayConfig,
    ModelCapabilityConfig,
    provider_credential_env_names,
)
from gateway.log_config import logger as gateway_logger
from gateway.services import provider_kwargs
from gateway.services.model_capabilities import resolve_capabilities
from gateway.services.provider_kwargs import (
    _KEYLESS_PLACEHOLDER_API_KEY,
    AMBIENT_CREDENTIAL_PROVIDERS,
    BEDROCK_READ_TIMEOUT_SECONDS,
    get_provider_kwargs,
    keyless_placeholder_api_key,
    missing_declared_credential,
    normalize_pricing_key,
    provider_key,
    require_declared_credential,
    resolve_provider_selector,
    split_selector,
    warn_about_ignored_provider_variables,
)

# ---------------------------------------------------------------------------
# config.provider_instance_type
# ---------------------------------------------------------------------------


def test_instance_type_defaults_to_key() -> None:
    config = GatewayConfig(providers={"openai": {"api_key": "sk"}})
    assert config.provider_instance_type("openai") == "openai"


def test_instance_type_uses_declared_provider_type() -> None:
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai", "api_base": "http://x/v1"}})
    assert config.provider_instance_type("home_lab") == "openai"


def test_instance_type_normalizes_openai_compatible_alias() -> None:
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai-compatible"}})
    assert config.provider_instance_type("home_lab") == "openai"
    config2 = GatewayConfig(providers={"home_lab": {"provider_type": "openai_compatible"}})
    assert config2.provider_instance_type("home_lab") == "openai"


def test_instance_type_normalizes_anthropic_compatible_alias() -> None:
    config = GatewayConfig(providers={"proxy": {"provider_type": "anthropic-compatible"}})
    assert config.provider_instance_type("proxy") == "anthropic"
    config2 = GatewayConfig(providers={"proxy": {"provider_type": "anthropic_compatible"}})
    assert config2.provider_instance_type("proxy") == "anthropic"


def test_instance_type_unknown_instance_returns_input() -> None:
    assert GatewayConfig().provider_instance_type("anthropic") == "anthropic"


# ---------------------------------------------------------------------------
# config.provider_pricing_implementation
# ---------------------------------------------------------------------------


def test_pricing_implementation_uses_a_declared_provider_type() -> None:
    config = GatewayConfig(providers={"aws-prod": {"provider_type": "bedrock"}})
    assert config.provider_pricing_implementation("aws-prod") == "bedrock"


@pytest.mark.parametrize(
    "declared",
    ["openai-compatible", "openai_compatible", "anthropic-compatible", "anthropic_compatible"],
)
def test_pricing_implementation_ignores_wire_protocol_aliases(declared: str) -> None:
    """A ``*-compatible`` type says how to talk to an endpoint, not who runs it.

    ``provider_instance_type`` normalizes it to a real implementation because that
    is which SDK to dispatch with. Pricing must not follow: a self-hosted endpoint
    declared ``openai-compatible`` would otherwise bill at OpenAI's list rate for
    any OpenAI model name it happens to expose.
    """
    config = GatewayConfig(providers={"local": {"provider_type": declared}})
    assert config.provider_pricing_implementation("local") is None


def test_pricing_implementation_none_without_a_declared_type() -> None:
    """No declared type means the instance name is all pricing has to go on."""
    config = GatewayConfig(providers={"openai": {"api_key": "sk"}})
    assert config.provider_pricing_implementation("openai") is None
    assert GatewayConfig().provider_pricing_implementation("anthropic") is None


# ---------------------------------------------------------------------------
# config.validate_provider_instances
# ---------------------------------------------------------------------------


def test_validate_rejects_unknown_provider_type() -> None:
    config = GatewayConfig(providers={"home_lab": {"provider_type": "not-a-real-provider"}})
    with pytest.raises(ValueError, match="not a known provider"):
        config.validate_provider_instances()


def test_validate_accepts_alias_provider_type() -> None:
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai-compatible"}})
    config.validate_provider_instances()  # no raise


def test_validate_rejects_non_list_models() -> None:
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai", "models": "deepseek"}})
    with pytest.raises(ValueError, match="must be a list"):
        config.validate_provider_instances()


@pytest.mark.parametrize(
    ("providers", "match"),
    [
        ({"bedrock": {"client_args": {"config": {"read_timeot": 120}}}}, "not a valid botocore Config"),
        ({"aws": {"provider_type": "bedrock", "client_args": {"config": 120}}}, "must be a mapping"),
    ],
)
def test_validate_rejects_a_bedrock_config_botocore_cannot_build(providers: dict[str, Any], match: str) -> None:
    """Every Bedrock client merges this config in, so a typo would fail each request instead."""
    with pytest.raises(ValueError, match=match):
        GatewayConfig(providers=providers).validate_provider_instances()


def test_validate_accepts_a_bedrock_config_botocore_can_build() -> None:
    GatewayConfig(
        providers={"bedrock": {"client_args": {"config": {"read_timeout": 120}}}}
    ).validate_provider_instances()  # no raise


def test_validate_allows_instance_without_provider_type() -> None:
    # Backward compatible: keys that are real providers need no provider_type and
    # are not hard-validated here.
    GatewayConfig(providers={"openai": {"api_key": "sk"}}).validate_provider_instances()


def test_valueless_entry_loads_as_empty_config() -> None:
    # Regression (mozilla-ai/otari#389): a keyless local backend has no credential
    # to declare, so `ollama:` with no body is the natural config. YAML parses that
    # as None, which the dict[str, dict] annotation rejected outright.
    config = GatewayConfig(providers={"ollama": None})  # type: ignore[dict-item]
    assert config.providers == {"ollama": {}}
    config.validate_provider_instances()  # no raise


def test_valueless_providers_block_loads_as_no_providers() -> None:
    # Commenting out every entry leaves a bare `providers:`, which YAML also reads
    # as None. That should mean "no providers", not the same pydantic type error.
    config = GatewayConfig(**yaml.safe_load("providers:\n"))
    assert config.providers == {}


def test_valueless_entry_from_yaml_is_a_configured_instance() -> None:
    # The whole point of the entry is opting the local backend into discovery,
    # which is scoped to config.providers, so the key must survive the load.
    config = GatewayConfig(**yaml.safe_load("providers:\n  ollama:\n"))
    assert "ollama" in config.providers
    assert config.provider_instance_type("ollama") == "ollama"


def _capture_gateway_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Route the ``gateway`` logger (which does not propagate) into caplog."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")


def _validate_capturing_warnings(config: GatewayConfig, caplog: pytest.LogCaptureFixture) -> str:
    _capture_gateway_logs(caplog)
    try:
        config.validate_provider_instances()  # no raise
    finally:
        gateway_logger.removeHandler(caplog.handler)
    return caplog.text


def test_bare_keyless_local_entry_warns_nothing(caplog: pytest.LogCaptureFixture) -> None:
    # A keyless backend has no credential to declare, so the bare entry is the
    # intended config: it must stay silent, and stay discoverable.
    config = GatewayConfig(**yaml.safe_load("providers:\n  ollama:\n  llamacpp:\n  llamafile:\n"))
    assert _validate_capturing_warnings(config, caplog) == ""
    assert set(config.providers) == {"ollama", "llamacpp", "llamafile"}


def test_bare_keyed_entry_warns(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    # `openai:` with nothing beneath it used to be a startup type error. It now
    # loads, so warn: with no credential anywhere it is far more likely a
    # truncated YAML edit than an intentionally uncredentialed instance.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = GatewayConfig(**yaml.safe_load("providers:\n  openai:\n"))
    text = _validate_capturing_warnings(config, caplog)
    assert "providers.openai" in text
    assert "OPENAI_API_KEY" in text


def test_bare_keyed_entry_with_env_credential_warns_nothing_at_load(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Not a load-time warning: the startup prune reports the entry as ignored,
    # with the api_key line to add (test_entry_naming_no_key_lacks_one_although_the_variable_is_set).
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    config = GatewayConfig(**yaml.safe_load("providers:\n  openai:\n"))
    assert _validate_capturing_warnings(config, caplog) == ""


def test_bare_entry_of_unknown_instance_warns_nothing(caplog: pytest.LogCaptureFixture) -> None:
    # An instance that is not a known any-llm implementation is left alone here
    # (validate_provider_instances is lenient about it), so there is no basis for
    # claiming it needs a credential.
    config = GatewayConfig(**yaml.safe_load("providers:\n  mystery_box:\n"))
    assert _validate_capturing_warnings(config, caplog) == ""


def test_entry_with_settings_but_no_credential_warns_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Only a settings-less entry looks like a truncated edit; an entry with a body
    # is a deliberate config that predates this check.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = GatewayConfig(providers={"openai": {"models": ["gpt-4o"]}})
    assert _validate_capturing_warnings(config, caplog) == ""


# ---------------------------------------------------------------------------
# provider_credential_env_names
# ---------------------------------------------------------------------------


def test_credential_env_names_single() -> None:
    assert provider_credential_env_names("openai") == ("OPENAI_API_KEY",)


def test_credential_env_names_splits_alternatives() -> None:
    # gemini declares "GEMINI_API_KEY/GOOGLE_API_KEY"; either one authenticates.
    assert provider_credential_env_names("gemini") == ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def test_credential_env_names_empty_for_keyless_backends() -> None:
    # any-llm spells "no credential" as the literal string "None".
    for keyless in ("ollama", "llamafile"):
        assert provider_credential_env_names(keyless) == ()


def test_credential_env_names_none_for_unknown_provider() -> None:
    # Not "keyless": unknowable, which callers treat as no basis for a warning.
    assert provider_credential_env_names("not-a-real-provider") is None


# Providers that override `_verify_and_set_api_key` and still genuinely require a
# credential, so `credential_ladder_exhausted` is right to treat them as keyed.
# gemini raises `MissingApiKeyError` when neither GEMINI_API_KEY nor
# GOOGLE_API_KEY resolves; azure and azureopenai tolerate a missing key only for
# an Entra ID deployment, which still needs an endpoint from config and so never
# reaches the port with empty kwargs. typesafe tolerates one only off its hosted
# API, and serves no chat or Messages request otari could route to it anyway.
_KEYED_DESPITE_OVERRIDING = frozenset({"gemini", "azure", "azureopenai", "typesafe"})


def test_the_uncredentialed_provider_roster_has_not_drifted() -> None:
    """Pin which providers any-llm calls without a credential otari can see.

    `credential_ladder_exhausted` decides whether a candidate that resolved no
    credential is genuinely unserved, and a wrong answer is not visible from
    otari: on a build with an overlay bound to `ModelProviderPort` it silently
    routes a working self-hosted or IAM-authenticated request to somebody else's
    fleet. `provider_credential_env_names` cannot answer it, because it sees the
    *declaration* and these providers declare a variable they do not insist on,
    so the two sets are written out by hand in `config.py` and `provider_kwargs.py`.

    This is the drift guard for both directions, which is why it asserts on the
    whole roster rather than only on the names already listed. A provider that
    stops overriding fails here (it now demands a key, and treating it as keyless
    would break it); more importantly, an any-llm release that adds a *new*
    keyless provider also fails here, instead of leaving a request that works
    today quietly claimed by a fleet. Either way the fix is a human deciding
    which set the name belongs in.

    It reads a private method because that is where the fact lives: the public
    surface reports the declaration, and the declaration is what misleads.
    """
    overriding = {
        provider.value
        for provider in LLMProvider
        if AnyLLM.get_provider_class(provider.value)._verify_and_set_api_key is not AnyLLM._verify_and_set_api_key
    }
    # The `()`-declaring providers are handled by `provider_credential_env_names`
    # itself and need no hand-written entry, so they are expected here but not in
    # either set.
    declares_nothing = {name for name in overriding if provider_credential_env_names(name) == ()}
    classified = KEYLESS_SELF_HOSTED_PROVIDERS | AMBIENT_CREDENTIAL_PROVIDERS | _KEYED_DESPITE_OVERRIDING

    assert overriding - declares_nothing == classified, (
        "any-llm's uncredentialed-provider roster changed. Every provider overriding "
        "_verify_and_set_api_key must be classified in config.py "
        "(KEYLESS_SELF_HOSTED_PROVIDERS), provider_kwargs.py (AMBIENT_CREDENTIAL_PROVIDERS) or here "
        "(_KEYED_DESPITE_OVERRIDING). Unclassified names are treated as keyed, which is "
        "the direction that hands a working request to a hosted fleet."
    )
    # And nothing in either set has quietly started demanding a key.
    for name in KEYLESS_SELF_HOSTED_PROVIDERS | AMBIENT_CREDENTIAL_PROVIDERS:
        assert provider_credential_env_names(name), f"{name} no longer declares a credential variable"
        assert name in overriding, f"any-llm now unconditionally requires a credential for {name}"


def test_validate_rejects_instance_name_with_separator() -> None:
    # A name containing ':' or '/' could never match the selector split and would
    # be silently unreachable; reject it at startup instead.
    for bad in ("my:lab", "my/lab"):
        with pytest.raises(ValueError, match="must not contain"):
            GatewayConfig(providers={bad: {"provider_type": "openai"}}).validate_provider_instances()


# ---------------------------------------------------------------------------
# split_selector
# ---------------------------------------------------------------------------


def test_split_selector_colon_first() -> None:
    assert split_selector("home_lab:deepseek-v4") == ("home_lab", "deepseek-v4")


def test_split_selector_slash() -> None:
    assert split_selector("openai/gpt-4o") == ("openai", "gpt-4o")


def test_split_selector_no_delimiter() -> None:
    assert split_selector("gpt-4o") is None


# ---------------------------------------------------------------------------
# resolve_provider_selector
# ---------------------------------------------------------------------------


def test_resolve_plain_provider_unchanged() -> None:
    config = GatewayConfig(providers={"openai": {"api_key": "sk-real"}})
    resolved = resolve_provider_selector(config, "openai:gpt-4o")
    assert resolved.instance == "openai"
    assert resolved.provider == LLMProvider.OPENAI
    assert resolved.model == "gpt-4o"
    assert resolved.dispatch_model == "openai:gpt-4o"
    assert resolved.kwargs["api_key"] == "sk-real"


def test_resolve_named_instance_routes_to_implementation() -> None:
    config = GatewayConfig(
        providers={
            "openai": {"api_key": "sk-real"},
            "home_lab": {
                "provider_type": "openai",
                "api_base": "https://box.ts.net/v1",
                "api_key": "home-token",
            },
        }
    )
    resolved = resolve_provider_selector(config, "home_lab:deepseek-v4-flash")
    # any-llm is dispatched against the implementation, never the instance name.
    assert resolved.provider == LLMProvider.OPENAI
    assert resolved.model == "deepseek-v4-flash"
    assert resolved.dispatch_model == "openai:deepseek-v4-flash"
    # ...but billing/pricing key on the instance, with the instance's credentials.
    assert resolved.instance == "home_lab"
    assert resolved.kwargs["api_base"] == "https://box.ts.net/v1"
    assert resolved.kwargs["api_key"] == "home-token"


def test_resolve_two_instances_do_not_collide() -> None:
    config = GatewayConfig(
        providers={
            "openai": {"api_key": "sk-real", "api_base": "https://api.openai.com/v1"},
            "home_lab": {"provider_type": "openai", "api_base": "https://box/v1", "api_key": "ht"},
        }
    )
    real = resolve_provider_selector(config, "openai:gpt-4o")
    local = resolve_provider_selector(config, "home_lab:gpt-4o")
    assert real.kwargs["api_key"] == "sk-real"
    assert local.kwargs["api_key"] == "ht"
    assert real.dispatch_model == local.dispatch_model == "openai:gpt-4o"


def test_resolve_alias_instance() -> None:
    config = GatewayConfig(providers={"vllm_box": {"provider_type": "openai-compatible", "api_base": "http://v/v1"}})
    resolved = resolve_provider_selector(config, "vllm_box:qwen3")
    assert resolved.provider == LLMProvider.OPENAI
    assert resolved.dispatch_model == "openai:qwen3"


def test_resolve_registry_only_provider_remains_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Otari provider resolution remains enum-backed despite any-llm's wider split type."""
    monkeypatch.setattr(AnyLLM, "split_model_provider", lambda _selector: ("registry-only", "model"))

    with pytest.raises(ValueError, match="registry-only"):
        resolve_provider_selector(GatewayConfig(), "registry-only:model")


def test_resolve_unknown_provider_raises() -> None:
    # An unconfigured prefix that is not a real provider surfaces any-llm's error
    # (caught as (ValueError, AnyLLMError) by the budget gate).
    with pytest.raises(AnyLLMError):
        resolve_provider_selector(GatewayConfig(), "not_a_provider:model")


# ---------------------------------------------------------------------------
# get_provider_kwargs strips instance-only metadata
# ---------------------------------------------------------------------------


def test_get_provider_kwargs_strips_provider_type_and_models() -> None:
    config = GatewayConfig(
        providers={
            "home_lab": {
                "provider_type": "openai",
                "models": ["a", "b"],
                "api_base": "http://x/v1",
                "api_key": "k",
            }
        }
    )
    kwargs = get_provider_kwargs(config, LLMProvider.OPENAI, instance="home_lab")
    assert "provider_type" not in kwargs
    assert "models" not in kwargs
    assert kwargs == {"api_base": "http://x/v1", "api_key": "k"}


def test_bedrock_uses_aws_region_when_default_region_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    kwargs = get_provider_kwargs(GatewayConfig(), LLMProvider.BEDROCK)

    assert _without_read_timeout(kwargs["client_args"]) == {"region_name": "us-east-1"}


def test_bedrock_region_fallback_preserves_other_client_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    config = GatewayConfig(providers={"bedrock": {"client_args": {"retries": {"max_attempts": 2}}}})

    kwargs = get_provider_kwargs(config, LLMProvider.BEDROCK)

    assert _without_read_timeout(kwargs["client_args"]) == {
        "retries": {"max_attempts": 2},
        "region_name": "us-east-1",
    }


def test_bedrock_preserves_explicit_region_and_aws_default_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    explicit_config = GatewayConfig(providers={"bedrock": {"client_args": {"region_name": "ap-southeast-2"}}})

    explicit_kwargs = get_provider_kwargs(explicit_config, LLMProvider.BEDROCK)

    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    default_config = GatewayConfig(providers={"bedrock": {"client_args": {"retries": {"max_attempts": 2}}}})
    default_kwargs = get_provider_kwargs(default_config, LLMProvider.BEDROCK)

    assert _without_read_timeout(explicit_kwargs["client_args"]) == {"region_name": "ap-southeast-2"}
    assert _without_read_timeout(default_kwargs["client_args"]) == {"retries": {"max_attempts": 2}}


def _without_read_timeout(client_args: dict[str, Any]) -> dict[str, Any]:
    """``client_args`` less the read-timeout ``config`` every Bedrock client gets."""
    rest = dict(client_args)
    assert rest.pop("config").read_timeout == BEDROCK_READ_TIMEOUT_SECONDS
    return rest


def test_bedrock_outlasts_botocores_default_read_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """botocore's 60s default fails a stream whose model goes quiet for a minute."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    config = GatewayConfig(providers={"bedrock": {"client_args": {"config": {"connect_timeout": 5}}}})

    client_config = get_provider_kwargs(config, LLMProvider.BEDROCK)["client_args"]["config"]

    assert client_config.read_timeout == BEDROCK_READ_TIMEOUT_SECONDS
    assert client_config.connect_timeout == 5
    assert config.providers["bedrock"]["client_args"]["config"] == {"connect_timeout": 5}


def test_aws_region_fallback_is_limited_to_bedrock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    kwargs = get_provider_kwargs(GatewayConfig(), LLMProvider.OPENAI)

    assert "client_args" not in kwargs


# ---------------------------------------------------------------------------
# get_provider_kwargs: keyless custom-endpoint placeholder (mozilla-ai/otari#421)
# ---------------------------------------------------------------------------


def test_keyless_custom_endpoint_gets_placeholder_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # A custom OpenAI-compatible endpoint with no key would otherwise be rejected
    # by any-llm before it is even dialed; the placeholder makes the key optional.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai", "api_base": "http://x/v1"}})
    kwargs = get_provider_kwargs(config, LLMProvider.OPENAI, instance="home_lab")
    assert kwargs == {"api_base": "http://x/v1", "api_key": _KEYLESS_PLACEHOLDER_API_KEY}


def test_explicit_key_is_not_overridden_by_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = GatewayConfig(
        providers={"home_lab": {"provider_type": "openai", "api_base": "http://x/v1", "api_key": "sk-real"}}
    )
    kwargs = get_provider_kwargs(config, LLMProvider.OPENAI, instance="home_lab")
    assert kwargs["api_key"] == "sk-real"


def test_no_placeholder_without_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    # A default hosted endpoint (no custom api_base) needs a real key, so a
    # placeholder there would only be sent to the provider and rejected.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = GatewayConfig(providers={"openai": {}})
    kwargs = get_provider_kwargs(config, LLMProvider.OPENAI, instance="openai")
    assert "api_key" not in kwargs


def test_a_keyless_endpoint_never_takes_the_provider_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    # The entry names no key, so OPENAI_API_KEY must never be sent to its api_base.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai", "api_base": "http://x/v1"}})
    kwargs = get_provider_kwargs(config, LLMProvider.OPENAI, instance="home_lab")
    assert kwargs["api_key"] == _KEYLESS_PLACEHOLDER_API_KEY


def test_organization_endpoint_never_takes_the_deployment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    # An organization's keyless endpoint is not the deployment's declaration, so
    # the deployment's own OPENAI_API_KEY must never be sent to it.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    monkeypatch.setattr(provider_kwargs, "cached_org_provider_kwargs", lambda _ws, _p: {"api_base": "http://x/v1"})
    kwargs = get_provider_kwargs(GatewayConfig(providers={}), LLMProvider.OPENAI, workspace_id=uuid.uuid4())
    assert kwargs["api_key"] == _KEYLESS_PLACEHOLDER_API_KEY


def test_keyless_placeholder_helper() -> None:
    # Keyless custom endpoint: placeholder.
    assert keyless_placeholder_api_key("http://x/v1", None) == _KEYLESS_PLACEHOLDER_API_KEY
    # No api_base (hosted default): no placeholder.
    assert keyless_placeholder_api_key(None, None) is None
    # Key already present: no placeholder.
    assert keyless_placeholder_api_key("http://x/v1", "sk-real") is None


# ---------------------------------------------------------------------------
# normalize_pricing_key
# ---------------------------------------------------------------------------


def test_normalize_pricing_key_instance() -> None:
    config = GatewayConfig(providers={"home_lab": {"provider_type": "openai"}})
    assert normalize_pricing_key(config, "home_lab:deepseek") == "home_lab:deepseek"


def test_normalize_pricing_key_provider_slash_to_colon() -> None:
    assert normalize_pricing_key(GatewayConfig(), "openai/gpt-4o") == "openai:gpt-4o"


def test_normalize_pricing_key_accepts_string_provider_from_any_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AnyLLM, "split_model_provider", lambda _selector: ("registry-only", "model"))

    assert normalize_pricing_key(GatewayConfig(), "registry-only/model") == "registry-only:model"


def test_normalize_pricing_key_unparseable_returned_unchanged() -> None:
    assert normalize_pricing_key(GatewayConfig(), "bare-model") == "bare-model"


def test_normalize_pricing_key_orphaned_instance_does_not_raise() -> None:
    # Regression: a pricing row keyed on an instance that is no longer configured
    # (e.g. a stored provider that could not be decrypted and was skipped) must be
    # returned unchanged, not raise AnyLLMError and 500 the models listing.
    assert normalize_pricing_key(GatewayConfig(), "home-lab:qwen3") == "home-lab:qwen3"


# ---------------------------------------------------------------------------
# provider_key
# ---------------------------------------------------------------------------


def test_provider_key_from_enum_member() -> None:
    assert provider_key(LLMProvider.OPENAI) == "openai"


def test_provider_key_from_registry_only_name() -> None:
    # any-llm >= 1.24 returns a bare name for a gateway that has a registry entry
    # but no LLMProvider member, so pricing/budget keys must accept that form.
    assert provider_key("some-registry-gateway") == "some-registry-gateway"


# ---------------------------------------------------------------------------
# capabilities key on the instance name
# ---------------------------------------------------------------------------


def test_capabilities_match_instance_scoped_key() -> None:
    config = GatewayConfig(
        providers={"home_lab": {"provider_type": "openai"}},
        model_capabilities={"home_lab:qwen2-vl": ModelCapabilityConfig(supports_image=True)},
    )
    caps = resolve_capabilities(config, LLMProvider.OPENAI, "qwen2-vl", instance="home_lab")
    assert caps.source == "config"
    assert caps.image is True


# ---------------------------------------------------------------------------
# pricing read endpoints tolerate instance-scoped keys (no 500)
# ---------------------------------------------------------------------------


def test_candidate_model_keys_handles_instance_key_without_raising() -> None:
    # Regression: an instance name is not an any-llm provider, so the underlying
    # split raises AnyLLMError (not ValueError). _candidate_model_keys must catch
    # it and still return the stored key, rather than 500 on pricing reads.
    assert _candidate_model_keys("home_lab:deepseek") == ["home_lab:deepseek"]


def test_candidate_model_keys_normalizes_instance_slash_to_colon() -> None:
    assert _candidate_model_keys("home_lab/deepseek") == ["home_lab/deepseek", "home_lab:deepseek"]


def test_candidate_model_keys_real_provider_unchanged() -> None:
    assert _candidate_model_keys("openai:gpt-4o") == ["openai:gpt-4o", "openai/gpt-4o"]


def test_candidate_model_keys_accepts_string_provider_from_any_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AnyLLM, "split_model_provider", lambda _selector: ("registry-only", "model"))

    assert _candidate_model_keys("registry-only:model") == ["registry-only:model", "registry-only/model"]


# ---------------------------------------------------------------------------
# A provider nothing declares is not served on its own environment variable
# ---------------------------------------------------------------------------


def test_undeclared_provider_on_env_var_is_missing_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    config = GatewayConfig(providers={})
    kwargs = get_provider_kwargs(config, LLMProvider.ANTHROPIC)

    missing = missing_declared_credential(config, "anthropic", LLMProvider.ANTHROPIC, kwargs)

    assert missing is not None
    assert missing.provider == "anthropic"
    assert "sk-ant-secret-value" not in missing.detail
    with pytest.raises(MissingApiKeyError):
        require_declared_credential(config, "anthropic", LLMProvider.ANTHROPIC, kwargs)


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({"api_key": "sk-ant-test"}, id="interpolated-key"),
        pytest.param({"provider_type": "anthropic", "api_base": "http://x/v1"}, id="keyless-endpoint"),
    ],
)
def test_declared_provider_is_served(entry: dict[str, Any]) -> None:
    config = GatewayConfig(providers={"anthropic": entry})
    kwargs = get_provider_kwargs(config, LLMProvider.ANTHROPIC)

    assert missing_declared_credential(config, "anthropic", LLMProvider.ANTHROPIC, kwargs) is None


@pytest.mark.parametrize(
    "entry",
    [pytest.param({}, id="bare-entry"), pytest.param({"client_args": {"timeout": 60}}, id="client-args-only")],
)
def test_declared_entry_naming_no_key_is_refused_although_its_variable_is_set(
    entry: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    config = GatewayConfig(providers={"anthropic": entry})
    kwargs = get_provider_kwargs(config, LLMProvider.ANTHROPIC)

    missing = missing_declared_credential(config, "anthropic", LLMProvider.ANTHROPIC, kwargs)
    assert missing is not None
    assert missing.provider == "anthropic"


def test_organization_key_serves_an_undeclared_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(provider_kwargs, "cached_org_provider_kwargs", lambda _ws, _p: {"api_key": "sk-org"})
    config = GatewayConfig(providers={})
    kwargs = get_provider_kwargs(config, LLMProvider.ANTHROPIC, workspace_id=uuid.uuid4())

    assert missing_declared_credential(config, "anthropic", LLMProvider.ANTHROPIC, kwargs) is None


_NO_KEY_PROVIDERS = [
    LLMProvider.OLLAMA,
    LLMProvider.LLAMACPP,
    LLMProvider.LMSTUDIO,
    LLMProvider.VLLM,
    LLMProvider.BEDROCK,
    LLMProvider.SAGEMAKER,
    LLMProvider.VERTEXAI,
]


@pytest.mark.parametrize("provider", _NO_KEY_PROVIDERS)
def test_a_declared_keyless_or_ambient_provider_needs_no_key(provider: LLMProvider) -> None:
    # Vertex AI needs a project to build its client, which is configuration, not a credential.
    entry = {"project": "test-project"} if provider == LLMProvider.VERTEXAI else {}
    config = GatewayConfig(providers={provider.value: entry})
    kwargs = get_provider_kwargs(config, provider)

    assert missing_declared_credential(config, provider.value, provider, kwargs) is None


@pytest.mark.parametrize("provider", _NO_KEY_PROVIDERS)
def test_an_undeclared_keyless_or_ambient_provider_is_refused(provider: LLMProvider) -> None:
    # The local server and the cloud role are the deployment's own, so they serve
    # only a provider the deployment declares.
    config = GatewayConfig(providers={})
    kwargs = get_provider_kwargs(config, provider)

    assert missing_declared_credential(config, provider.value, provider, kwargs) is not None


def test_an_organization_key_naming_no_credential_never_rides_the_deployments_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_kwargs, "cached_org_provider_kwargs", lambda _ws, _p: {})
    config = GatewayConfig(providers={})
    kwargs = get_provider_kwargs(config, LLMProvider.BEDROCK, workspace_id=uuid.uuid4())

    assert missing_declared_credential(config, "bedrock", LLMProvider.BEDROCK, kwargs) is not None


@pytest.fixture
def ignored_variable_warnings(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    _capture_gateway_logs(caplog)
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


def test_startup_warns_about_a_variable_no_entry_declares(
    monkeypatch: pytest.MonkeyPatch, ignored_variable_warnings: pytest.LogCaptureFixture
) -> None:
    """Names the variable and the provider, never the value."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret-value")

    # The value interpolation leaves behind, so this entry reads OPENAI_API_KEY.
    warn_about_ignored_provider_variables(GatewayConfig(providers={"openai": {"api_key": "sk-openai-secret-value"}}))

    records = [r for r in ignored_variable_warnings.records if "Ignoring" in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "ANTHROPIC_API_KEY" in message and "anthropic" in message
    assert "OPENAI_API_KEY" not in message
    assert "secret-value" not in message


def test_startup_warns_nothing_when_every_variable_is_read(
    monkeypatch: pytest.MonkeyPatch, ignored_variable_warnings: pytest.LogCaptureFixture
) -> None:
    for provider in LLMProvider:
        monkeypatch.delenv(f"{provider.value.upper()}_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    # What ``api_key: ${ANTHROPIC_API_KEY}`` interpolates to.
    warn_about_ignored_provider_variables(GatewayConfig(providers={"anthropic": {"api_key": "sk-ant-test"}}))

    assert "Ignoring" not in ignored_variable_warnings.text


def test_startup_warns_about_a_variable_a_bare_entry_does_not_read(
    monkeypatch: pytest.MonkeyPatch, ignored_variable_warnings: pytest.LogCaptureFixture
) -> None:
    for provider in LLMProvider:
        monkeypatch.delenv(f"{provider.value.upper()}_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    warn_about_ignored_provider_variables(GatewayConfig(providers={"anthropic": {}}))

    assert "Ignoring ANTHROPIC_API_KEY" in ignored_variable_warnings.text


def test_startup_warns_nothing_about_a_variable_a_tool_reads(
    monkeypatch: pytest.MonkeyPatch, ignored_variable_warnings: pytest.LogCaptureFixture
) -> None:
    """DeepSeek's search and fetch read the DEEPSEEK_API_KEY its models do, with no ``providers:`` entry."""
    for provider in LLMProvider:
        monkeypatch.delenv(f"{provider.value.upper()}_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    warn_about_ignored_provider_variables(GatewayConfig())
    assert "Ignoring DEEPSEEK_API_KEY" in ignored_variable_warnings.text
    ignored_variable_warnings.clear()

    entry = {"provider": "deepseek", "api_key": "sk-deepseek-test"}
    warn_about_ignored_provider_variables(GatewayConfig(search_tools={"deepseek": entry}))
    warn_about_ignored_provider_variables(GatewayConfig(fetch_tools={"deepseek-fetch": entry}))

    assert "Ignoring" not in ignored_variable_warnings.text


# ---------------------------------------------------------------------------
# A providers: entry that declares no credential is ignored, not shadowing
# ---------------------------------------------------------------------------


@pytest.fixture
def uncredentialed_warnings(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> Iterator[pytest.LogCaptureFixture]:
    """Capture gateway warnings with the once-per-instance memory cleared and the SDK variables unset."""
    monkeypatch.setattr(provider_kwargs, "_uncredentialed_warned", set())
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    _capture_gateway_logs(caplog)
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({"api_key": ""}, id="empty-string"),
        pytest.param({"api_key": None}, id="none"),
        pytest.param({}, id="no-settings"),
        pytest.param({"client_args": {"timeout": 60}}, id="transport-only"),
        pytest.param({"provider_type": "openai", "models": ["gpt-4o"]}, id="meta-only"),
        pytest.param({"api_key": "", "temperature": 0.7}, id="call-options-are-not-credentials"),
        pytest.param({"api_key": "", "client_args": {"default_headers": {"X-Trace": "1"}}}, id="non-auth-headers"),
    ],
)
def test_entry_without_a_credential_names_the_variables_it_lacks(
    entry: dict[str, object], uncredentialed_warnings: pytest.LogCaptureFixture
) -> None:
    config = GatewayConfig(providers={"openai": entry})
    assert provider_kwargs.uncredentialed_env_names(config, "openai", entry) == ("OPENAI_API_KEY",)


def test_entry_naming_no_key_lacks_one_although_the_variable_is_set(
    monkeypatch: pytest.MonkeyPatch, uncredentialed_warnings: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    config = GatewayConfig(providers={"openai": {}})
    assert provider_kwargs.uncredentialed_env_names(config, "openai", {}) == ("OPENAI_API_KEY",)


@pytest.mark.parametrize(
    ("instance", "entry", "env"),
    [
        pytest.param("openai", {"api_key": "sk-config"}, {}, id="key-in-config"),
        pytest.param(
            "home_lab", {"provider_type": "openai", "api_base": "http://models.local/v1"}, {}, id="keyless-base"
        ),
        pytest.param("ollama", {}, {}, id="keyless-local-backend"),
        pytest.param("bedrock", {}, {}, id="ambient-credential"),
        pytest.param("vertexai", {}, {}, id="no-credential-variable"),
        pytest.param("mystery", {"provider_type": "not-a-provider"}, {}, id="unknown-type"),
        pytest.param(
            "openai",
            {"api_key": "", "client_args": {"default_headers": {"Authorization": "Bearer t"}}},
            {},
            id="auth-header-in-client-args",
        ),
        pytest.param("openai", {"azure_ad_token": "t"}, {}, id="other-credential-field"),
    ],
)
def test_entry_that_can_be_called_is_not_uncredentialed(
    instance: str,
    entry: dict[str, object],
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    uncredentialed_warnings: pytest.LogCaptureFixture,
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config = GatewayConfig.model_construct(providers={instance: entry})
    assert provider_kwargs.uncredentialed_env_names(config, instance, entry) is None


def test_prune_drops_the_hollow_entries_and_warns_once_per_instance(
    uncredentialed_warnings: pytest.LogCaptureFixture,
) -> None:
    config = GatewayConfig(
        providers={"openai": {"api_key": ""}, "anthropic": {"api_key": "sk-ant-config"}, "ollama": {}}
    )

    first = provider_kwargs.prune_uncredentialed_providers(config)
    second = provider_kwargs.prune_uncredentialed_providers(config)

    assert first == {"openai": ("OPENAI_API_KEY",)}
    assert second == {}
    assert set(config.providers) == {"anthropic", "ollama"}
    records = [r for r in uncredentialed_warnings.records if "declares no credential" in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "providers.openai" in message and "OPENAI_API_KEY" in message
    assert "sk-ant-config" not in message
