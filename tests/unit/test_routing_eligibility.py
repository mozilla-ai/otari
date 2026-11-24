"""Eligibility constraints for explicitly classified inference deployments.

Locality is a closed world: a target the operator has not classified is never
eligible. Operations are an open world: a declaration only narrows them.
"""

import contextlib
from collections.abc import AsyncIterator, Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest
from any_llm import LLMProvider
from pydantic import ValidationError

from gateway import main
from gateway.core.config import GatewayConfig, ModelCapabilityConfig
from gateway.models.routing import InferenceOperation, PolicySpec
from gateway.services.catalog import OfferingRow, build_selector_index, reset_selector_index, set_selector_index
from gateway.services.provider_kwargs import ResolvedProvider, resolve_provider_selector
from gateway.services.providers._hosted_provider_service import ConfiguredProviderImports
from gateway.services.routing import (
    NoEligibleCandidatesError,
    RoutingRequirements,
    RoutingService,
    RoutingSignal,
    compile_catalog_plan,
    compile_policy,
    decide_ordering,
    explain_router_ordering,
)
from gateway.services.routing import compiler as compiler_module

LOCAL = "edge:chat-model"
CLOUD = "cloud:chat-model"
IMAGE = "edge:image-model"


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compiler_module, "_warned_eligibility", set())


@pytest.fixture
def config() -> GatewayConfig:
    return GatewayConfig.model_validate(
        {
            "model_discovery": False,
            "providers": {
                "edge": {
                    "provider_type": "openai-compatible",
                    "api_base": "http://edge.internal/v1",
                    "routing_locality": "local",
                },
                "cloud": {"provider_type": "openai", "api_key": "test-key", "routing_locality": "cloud"},
            },
            "model_capabilities": {
                LOCAL: {"supports_image": True},
                IMAGE: {"operations": {"chat": {"enabled": False}}},
            },
        }
    )


def _policy(default: str, *backups: str, **eligibility: Any) -> PolicySpec:
    return PolicySpec.model_validate(
        {"select": [{"default": default}], "on_failure": list(backups), "eligibility": eligibility}
    )


def _drops(plan_or_error: Any) -> list[tuple[str, str]]:
    return [(item.selector, item.reason) for item in plan_or_error.dropped]


# ---------------------------------------------------------------------------
# Locality: a closed world
# ---------------------------------------------------------------------------


def test_locality_applies_to_the_default_and_its_backups(config: GatewayConfig) -> None:
    plan = compile_policy(config, "edge-only", _policy(CLOUD, IMAGE, LOCAL, allowed_localities=["local"]))

    assert plan.head.instance == "edge"
    assert plan.head.model == "chat-model"
    assert plan.head.selection_reason == "on_failure"
    assert plan.head.kwargs["api_base"] == "http://edge.internal/v1"
    assert "routing_locality" not in plan.head.kwargs
    assert plan.head.allowed_localities == frozenset({"local"})
    assert _drops(plan) == [(CLOUD, "locality_not_allowed"), (IMAGE, "target_unavailable")]


def test_a_credentialless_or_unclassified_target_cannot_be_rebound(config: GatewayConfig) -> None:
    config.providers["cloud"].pop("api_key")
    config.providers["edge"].pop("routing_locality")
    spec = _policy(CLOUD, LOCAL, allowed_localities=["local", "cloud"])
    with pytest.raises(NoEligibleCandidatesError) as caught:
        compile_policy(config, "preferred", spec)

    assert caught.value.status_code == 403
    assert "chat-model" not in caught.value.caller_detail
    assert [item.reason for item in caught.value.dropped] == ["target_unavailable", "unknown_locality"]


def test_a_hosted_target_takes_the_locality_its_hosted_provider_declares(
    config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    declared: dict[str, str | None] = {"anthropic": "cloud"}
    monkeypatch.setattr(compiler_module, "hosted_routing_locality", lambda name: declared.get(name))
    spec = _policy("anthropic:claude-x", LOCAL, allowed_localities=["cloud", "local"])

    plan = compile_policy(config, "hosted", spec)
    assert plan.head.instance == "anthropic"
    assert plan.head.allowed_localities == frozenset({"cloud", "local"})

    declared["anthropic"] = None
    plan = compile_policy(config, "hosted", spec)
    assert plan.head.instance == "edge"
    assert _drops(plan) == [("anthropic:claude-x", "unknown_locality")]
    assert "hosted provider" in plan.dropped[0].detail

    declared["anthropic"] = "private"
    plan = compile_policy(config, "hosted", spec)
    assert _drops(plan) == [("anthropic:claude-x", "locality_not_allowed")]


def test_a_tenant_credential_is_never_eligible(config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    tenant = ResolvedProvider(
        instance="anthropic",
        provider=LLMProvider.ANTHROPIC,
        model="claude-x",
        kwargs={"api_key": "org-key", "api_base": "https://org.example/v1"},
    )
    real = resolve_provider_selector

    def resolve(cfg: GatewayConfig, selector: str, *args: Any, **kwargs: Any) -> ResolvedProvider:
        return tenant if selector == "anthropic:claude-x" else real(cfg, selector, *args, **kwargs)

    monkeypatch.setattr(compiler_module, "resolve_provider_selector", resolve)
    monkeypatch.setattr(compiler_module, "hosted_routing_locality", lambda _name: "cloud")
    plan = compile_policy(config, "tenant", _policy("anthropic:claude-x", LOCAL, allowed_localities=["cloud", "local"]))

    assert plan.head.instance == "edge"
    assert _drops(plan) == [("anthropic:claude-x", "unknown_locality")]
    assert "did not declare" in plan.dropped[0].detail


# ---------------------------------------------------------------------------
# Operations: an open world a declaration narrows
# ---------------------------------------------------------------------------


def test_an_undeclared_operation_is_served_streamed_or_not(config: GatewayConfig) -> None:
    spec = _policy(CLOUD, allowed_localities=["cloud"])
    for operation in (InferenceOperation.CHAT, InferenceOperation.MESSAGES, InferenceOperation.RESPONSES):
        for streaming in (False, True):
            plan = compile_policy(config, "open", spec, requirements=RoutingRequirements(operation, streaming))
            assert plan.head.instance == "cloud"
            assert plan.dropped == []


def test_a_declaration_narrows_operations_and_streaming(config: GatewayConfig) -> None:
    config.model_capabilities[CLOUD] = ModelCapabilityConfig.model_validate(
        {"operations": {"messages": {"enabled": False}, "chat": {"streaming": False}}}
    )
    spec = _policy(CLOUD, LOCAL, allowed_localities=["local", "cloud"])

    messages = compile_policy(config, "multi", spec, requirements=RoutingRequirements(InferenceOperation.MESSAGES))
    assert messages.head.instance == "edge"
    assert _drops(messages) == [(CLOUD, "target_unavailable")]

    streamed = compile_policy(config, "multi", spec, requirements=RoutingRequirements(streaming=True))
    assert streamed.head.instance == "edge"
    assert _drops(streamed) == [(CLOUD, "streaming_not_supported")]

    unstreamed = compile_policy(config, "multi", spec)
    assert unstreamed.head.instance == "cloud"


def test_a_provider_default_applies_until_a_model_overrides_it(config: GatewayConfig) -> None:
    config.model_capabilities["edge:*"] = ModelCapabilityConfig.model_validate(
        {"operations": {"chat": {"streaming": False}}}
    )
    spec = _policy("edge:other-model", LOCAL, allowed_localities=["local"])
    streamed = RoutingRequirements(streaming=True)

    with pytest.raises(NoEligibleCandidatesError) as caught:
        compile_policy(config, "edge", spec, requirements=streamed)
    assert [item.reason for item in caught.value.dropped] == ["streaming_not_supported", "streaming_not_supported"]

    config.model_capabilities[LOCAL] = ModelCapabilityConfig.model_validate(
        {"supports_image": True, "operations": {"chat": {}}}
    )
    plan = compile_policy(config, "edge", spec, requirements=streamed)
    assert plan.head.model == "chat-model"
    assert _drops(plan) == [("edge:other-model", "streaming_not_supported")]


def test_required_native_input_takes_the_normalizer_verdict(config: GatewayConfig) -> None:
    spec = _policy("edge:other-model", LOCAL, allowed_localities=["local"], required_capabilities=["image"])
    plan = compile_policy(config, "vision", spec)

    assert plan.head.model == "chat-model"
    assert _drops(plan) == [("edge:other-model", "capability_not_supported")]

    pdf = _policy(LOCAL, allowed_localities=["local"], required_capabilities=["pdf"])
    with pytest.raises(NoEligibleCandidatesError) as caught:
        compile_policy(config, "pdf", pdf)
    assert [item.reason for item in caught.value.dropped] == ["capability_not_supported"]

    # Provider metadata counts on the provider's own endpoint, and not behind another.
    cloud = _policy(CLOUD, allowed_localities=["cloud"], required_capabilities=["image"])
    assert compile_policy(config, "cloud-vision", cloud).head.instance == "cloud"
    config.providers["cloud"]["api_base"] = "https://proxy.example/v1"
    with pytest.raises(NoEligibleCandidatesError):
        compile_policy(config, "cloud-vision", cloud)


# ---------------------------------------------------------------------------
# Ranking and explain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["priority", "weighted"])
@pytest.mark.asyncio
async def test_router_ranks_the_eligible_pool_and_retains_drop_reasons(config: GatewayConfig, backend: str) -> None:
    router: dict[str, object] = {"router": backend, "candidates": [IMAGE, CLOUD, LOCAL]}
    if backend == "weighted":
        router["weights"] = {IMAGE: 100}
    spec = PolicySpec.model_validate(
        {"select": [router, {"default": IMAGE}], "eligibility": {"allowed_localities": ["local", "cloud"]}}
    )
    ordering = await decide_ordering(
        config, spec, policy_name="preferred", user_id=None, allowlist=[LOCAL, IMAGE], signal=RoutingSignal()
    )
    assert ordering is not None
    assert ordering.selectors == [LOCAL]
    plan = compile_policy(config, "preferred", spec, allowlist=[LOCAL, IMAGE], router_ordering=ordering)
    assert _drops(plan) == [(IMAGE, "target_unavailable"), (CLOUD, "not_allowed")]
    explained, shares = explain_router_ordering(config, spec, allowlist=[LOCAL, IMAGE])
    assert explained is not None
    assert explained.selectors == [LOCAL]
    assert explained.dropped == tuple(plan.dropped)
    if backend == "weighted":
        assert [(share.selector, share.share_pct) for share in shares] == [(LOCAL, 100.0)]


# ---------------------------------------------------------------------------
# Drops are logged
# ---------------------------------------------------------------------------


def test_each_drop_is_warned_once_per_process(config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(compiler_module, "logger", logger)
    spec = _policy(CLOUD, LOCAL, allowed_localities=["local"])

    compile_policy(config, "edge-only", spec)
    compile_policy(config, "edge-only", spec)
    assert logger.warning.call_count == 1
    assert CLOUD in logger.warning.call_args.args

    with pytest.raises(NoEligibleCandidatesError):
        compile_policy(config, "cloud-only", _policy(LOCAL, allowed_localities=["cloud"]))
    assert logger.warning.call_count == 2


# ---------------------------------------------------------------------------
# Names: a constrained policy never resolves synchronously
# ---------------------------------------------------------------------------


def test_a_constrained_policy_cannot_fall_through_to_a_catalog_spelling(config: GatewayConfig) -> None:
    config.routing.policies["preferred"] = _policy(LOCAL, allowed_localities=["local"])
    set_selector_index(
        build_selector_index(
            [OfferingRow(CLOUD, "cloud", "openai", 0.0), OfferingRow(LOCAL, "edge", "openai", 1.0)],
            {"vendor/preferred": ("vendor/preferred", (CLOUD, LOCAL))},
        )
    )
    try:
        assert resolve_provider_selector(config, "vendor/preferred").instance == "cloud"
        config.routing.policies["unconstrained"] = config.routing.policies["preferred"].model_copy(
            update={"eligibility": None}
        )
        assert resolve_provider_selector(config, "unconstrained").instance == "edge"
        assert compile_catalog_plan(config, "preferred") is None
        with pytest.raises(ValueError, match="operation-aware"):
            resolve_provider_selector(config, "preferred")
    finally:
        reset_selector_index()


def test_a_constrained_policy_wins_over_a_stored_alias_of_its_name(
    config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.routing.policies["preferred"] = _policy(LOCAL, allowed_localities=["local"])
    monkeypatch.setattr(
        "gateway.services.provider_kwargs.resolve_effective_alias",
        lambda _config, name, *_args, **_kwargs: CLOUD if name in {"preferred", "unconstrained"} else None,
    )
    with pytest.raises(ValueError, match="operation-aware"):
        resolve_provider_selector(config, "preferred")

    config.routing.policies["unconstrained"] = config.routing.policies["preferred"].model_copy(
        update={"eligibility": None}
    )
    assert resolve_provider_selector(config, "unconstrained").instance == "cloud"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("locality", ["unknown", [], None])
def test_provider_locality_must_be_recognized(config: GatewayConfig, locality: object) -> None:
    config.providers["edge"]["routing_locality"] = locality
    with pytest.raises(ValueError, match="routing_locality"):
        config.validate_provider_instances()


def test_locality_requires_an_explicit_upstream_endpoint(config: GatewayConfig) -> None:
    config.providers["edge"].pop("api_base")
    with pytest.raises(ValueError, match="explicit HTTP"):
        config.validate_provider_instances()


@pytest.mark.parametrize("eligibility", [{}, {"allowed_localities": []}, {"allowed_localities": ["elsewhere"]}])
def test_policy_must_explicitly_admit_a_recognized_locality(eligibility: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PolicySpec.model_validate({"select": [{"default": LOCAL}], "eligibility": eligibility})


@pytest.mark.asyncio
async def test_a_private_locality_on_a_public_address_is_warned(
    config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.providers["lab"] = {
        "provider_type": "openai-compatible",
        "api_base": "https://lab.example/v1",
        "routing_locality": "private",
    }
    public = {"lab.example": ["203.0.113.7"], "edge.internal": []}

    async def resolve(host: str) -> list[str]:
        return public[host]

    monkeypatch.setattr("gateway.services.routing._locality_check.public_addresses_of", resolve)
    assert await RoutingService.warn_about_public_endpoints_declared_private(config) == ["lab"]


@pytest.mark.asyncio
async def test_an_entry_declaring_locality_is_never_copied_onto_a_hosted_provider(
    config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.providers["plain"] = {"provider_type": "openai", "api_key": "plain-key"}
    offered: list[Mapping[str, Any]] = []

    class _Service:
        async def import_configured_providers(self, entries: Mapping[str, Any]) -> ConfiguredProviderImports:
            offered.append(dict(entries))
            # As if an earlier start had copied every entry it is offered.
            return ConfiguredProviderImports(imported=[], already=list(entries), kept={})

    @contextlib.asynccontextmanager
    async def _uow() -> AsyncIterator[object]:
        yield object()

    monkeypatch.setattr(main, "create_unit_of_work", _uow)
    monkeypatch.setattr(main, "build_hosted_provider_service", lambda _uow, **_kwargs: _Service())
    await main._import_configured_providers(config)

    assert offered == [{"plain": {"provider_type": "openai", "api_key": "plain-key"}}]
    assert set(config.providers) == {"edge", "cloud"}
