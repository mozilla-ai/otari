"""Eligibility constraints for explicitly classified inference deployments."""

import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.models.routing import InferenceOperation, PolicySpec
from gateway.services.catalog import OfferingRow, build_selector_index, reset_selector_index, set_selector_index
from gateway.services.provider_kwargs import resolve_provider_selector
from gateway.services.routing import (
    NoEligibleCandidatesError,
    RoutingRequirements,
    RoutingSignal,
    compile_catalog_plan,
    compile_policy,
    decide_ordering,
    explain_router_ordering,
)

LOCAL = "edge:chat-model"
CLOUD = "cloud:chat-model"
IMAGE = "edge:image-model"


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
                LOCAL: {"operations": {"chat": {"streaming": True}}, "supports_image": True},
                CLOUD: {"operations": {"chat": {}, "messages": {}}},
                IMAGE: {"operations": {"image_generation": {}}},
                "undeclared-model": {"operations": {"chat": {}}},
            },
        }
    )


def test_capabilities_and_locality_apply_to_the_default_and_backups(config: GatewayConfig) -> None:
    spec = PolicySpec.model_validate(
        {
            "select": [{"default": IMAGE}],
            "on_failure": [CLOUD, "edge:undeclared-model", LOCAL],
            "eligibility": {"allowed_localities": ["local"], "required_capabilities": ["image"]},
        }
    )
    plan = compile_policy(config, "vision", spec, requirements=RoutingRequirements(streaming=True))

    assert spec.is_dynamic
    assert plan.head.instance == "edge"
    assert plan.head.model == "chat-model"
    assert plan.head.selection_reason == "on_failure"
    assert plan.head.kwargs["api_base"] == "http://edge.internal/v1"
    assert "routing_locality" not in plan.head.kwargs
    assert [(item.selector, item.reason) for item in plan.dropped] == [
        (IMAGE, "operation_not_supported"),
        (CLOUD, "locality_not_allowed"),
        ("edge:undeclared-model", "operation_not_supported"),
    ]


def test_an_operation_is_checked_independently_of_streaming_and_native_inputs(config: GatewayConfig) -> None:
    spec = PolicySpec.model_validate(
        {
            "select": [{"default": CLOUD}],
            "on_failure": [LOCAL],
            "eligibility": {"allowed_localities": ["local", "cloud"]},
        }
    )
    messages = compile_policy(config, "multi", spec, requirements=RoutingRequirements(InferenceOperation.MESSAGES))
    assert messages.head.instance == "cloud"
    assert [item.reason for item in messages.dropped] == ["operation_not_supported"]

    streamed = compile_policy(config, "multi", spec, requirements=RoutingRequirements(streaming=True))
    assert streamed.head.instance == "edge"
    assert [item.reason for item in streamed.dropped] == ["streaming_not_supported"]

    assert spec.eligibility is not None
    spec.eligibility.required_capabilities = ["image"]
    vision = compile_policy(config, "multi", spec)
    assert vision.head.instance == "edge"
    assert [item.reason for item in vision.dropped] == ["capability_not_supported"]


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
    assert [(item.selector, item.reason) for item in plan.dropped] == [
        (IMAGE, "operation_not_supported"),
        (CLOUD, "not_allowed"),
    ]
    explained, shares = explain_router_ordering(config, spec, allowlist=[LOCAL, IMAGE])
    assert explained is not None
    assert explained.selectors == [LOCAL]
    assert explained.dropped == tuple(plan.dropped)
    if backend == "weighted":
        assert [(share.selector, share.share_pct) for share in shares] == [(LOCAL, 100.0)]


def test_a_credentialless_or_unclassified_target_cannot_be_rebound(config: GatewayConfig) -> None:
    config.providers["cloud"].pop("api_key")
    config.providers["edge"].pop("routing_locality")
    spec = PolicySpec.model_validate(
        {
            "select": [{"default": CLOUD}],
            "on_failure": [LOCAL],
            "eligibility": {"allowed_localities": ["local", "cloud"]},
        }
    )
    with pytest.raises(NoEligibleCandidatesError) as caught:
        compile_policy(config, "preferred", spec)

    assert caught.value.status_code == 403
    assert "chat-model" not in caught.value.caller_detail
    assert [item.reason for item in caught.value.dropped] == ["target_unavailable", "unknown_locality"]


def test_a_disabled_operation_does_not_become_eligible_through_priority(config: GatewayConfig) -> None:
    config.model_capabilities[LOCAL].operations[InferenceOperation.CHAT].enabled = False
    spec = PolicySpec.model_validate(
        {
            "select": [{"default": LOCAL}],
            "on_failure": [CLOUD],
            "eligibility": {"allowed_localities": ["local", "cloud"]},
        }
    )
    plan = compile_policy(config, "preferred", spec)
    assert plan.head.instance == "cloud"
    assert [(item.selector, item.reason) for item in plan.dropped] == [(LOCAL, "target_unavailable")]


def test_a_constrained_policy_cannot_fall_through_to_a_catalog_spelling(config: GatewayConfig) -> None:
    config.routing.policies["preferred"] = PolicySpec.model_validate(
        {"select": [{"default": LOCAL}], "eligibility": {"allowed_localities": ["local"]}}
    )
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
