from gateway.services.routing_policy_service import (
    ROUTING_STRATEGIES,
    classify_request_tier,
    estimate_output_tokens,
    estimate_prompt_tokens,
)


def test_classify_request_tier_uses_complexity_hints() -> None:
    request = {
        "messages": [
            {
                "role": "user",
                "content": "Design an architecture and migration plan for this gateway.",
            }
        ]
    }

    tier = classify_request_tier(request, prompt_tokens=12, config={})

    assert tier == "complex"


def test_classify_request_tier_respects_configured_thresholds() -> None:
    request = {"messages": [{"role": "user", "content": "short prompt"}]}

    tier = classify_request_tier(
        request,
        prompt_tokens=120,
        config={"tier_thresholds": {"medium": 100, "complex": 500, "reasoning": 1000}},
    )

    assert tier == "medium"


def test_estimates_prompt_and_output_tokens_from_request() -> None:
    request = {
        "messages": [{"role": "user", "content": "abcd" * 20}],
        "max_completion_tokens": 123,
    }

    assert estimate_prompt_tokens(request) >= 20
    assert estimate_output_tokens(request) == 123


def test_supported_strategies_include_merge_style_names() -> None:
    assert {
        "single",
        "priority",
        "lowest_cost",
        "least_latency",
        "cost_tier",
        "intelligent",
        "weighted_score",
    } <= ROUTING_STRATEGIES
