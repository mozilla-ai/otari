"""Unit tests for threshold-aware budget reservation estimates.

Compared exactly: the estimate is a ``Decimal`` quantized to the micro-dollar
(mozilla-ai/otari#691), so the expected amount is spelled as the arithmetic that
produces it rather than approximated. ``pytest.approx`` here would pass on an
estimate that had gone back to binary floating point.
"""

from decimal import Decimal
from typing import Any

from gateway.models.entities import MAX_COUNT_LIMIT, Budget, ModelPricing
from gateway.services.budget_service import _blocked_axis, estimate_cost, estimate_tokens


def _micro_dollars(*terms: Decimal | int) -> Decimal:
    """The USD total of per-million-token charges, exactly.

    Each term is ``tokens * rate``, which the cost core divides by a million
    once. Written this way so the expected value stays the arithmetic under
    test rather than a constant somebody would have to re-derive.
    """
    return sum((Decimal(term) for term in terms), Decimal(0)) / 1_000_000


def _pricing(**overrides: Any) -> ModelPricing:
    defaults = {
        "model_key": "anthropic:claude-sonnet-4",
        "input_price_per_million": 3.0,
        "output_price_per_million": 15.0,
        "cache_write_price_per_million": 3.75,
        "cache_write_1h_price_per_million": 6.0,
        "pricing_tiers": [
            {
                "min_input_tokens": 200_000,
                "input_price_per_million": 6.0,
                "output_price_per_million": 22.5,
                "cache_write_price_per_million": 7.5,
                "cache_write_1h_price_per_million": 12.0,
            }
        ],
    }
    defaults.update(overrides)
    return ModelPricing(**defaults)


def test_estimate_cost_uses_base_rates_below_context_threshold() -> None:
    estimate = estimate_cost(
        _pricing(),
        prompt_chars=199_999 * 4,
        max_output_tokens=100,
        default_output_tokens=1_024,
    )

    assert estimate == _micro_dollars(199_999 * 3, 100 * 15)


def test_estimate_cost_uses_context_tier_for_all_meters_at_threshold() -> None:
    estimate = estimate_cost(
        _pricing(),
        prompt_chars=200_000 * 4,
        max_output_tokens=100,
        default_output_tokens=1_024,
    )

    assert estimate == _micro_dollars(200_000 * 6, Decimal(100) * Decimal("22.5"))


def test_estimate_cost_reserves_explicit_cache_write_at_the_write_rate() -> None:
    estimate = estimate_cost(
        _pricing(),
        prompt_chars=1_000 * 4,
        max_output_tokens=100,
        default_output_tokens=1_024,
        cache_write_ttl="5m",
    )

    # Any prompt token could become a 5m cache write, so the input side is
    # reserved at the (dearer) cache-write rate rather than stacked on top of it.
    assert estimate == _micro_dollars(Decimal(1_000) * Decimal("3.75"), 100 * 15)


def test_estimate_cost_reserves_cache_write_using_the_context_tier() -> None:
    estimate = estimate_cost(
        _pricing(),
        prompt_chars=200_000 * 4,
        max_output_tokens=100,
        default_output_tokens=1_024,
        cache_write_ttl="5m",
    )

    # The estimated input alone crosses the tier, so the tier's write and output
    # rates apply; the tier is selected from the real billable total, not double.
    assert estimate == _micro_dollars(Decimal(200_000) * Decimal("7.5"), Decimal(100) * Decimal("22.5"))


def test_estimate_cost_uses_requested_one_hour_cache_write_rate() -> None:
    estimate = estimate_cost(
        _pricing(),
        prompt_chars=1_000 * 4,
        max_output_tokens=0,
        default_output_tokens=1_024,
        cache_write_ttl="1h",
    )

    assert estimate == _micro_dollars(1_000 * 6)


def test_estimate_cost_falls_back_to_input_when_cache_write_is_unpriced() -> None:
    estimate = estimate_cost(
        _pricing(cache_write_price_per_million=None, cache_write_1h_price_per_million=None),
        prompt_chars=1_000 * 4,
        max_output_tokens=0,
        default_output_tokens=1_024,
        cache_write_ttl="1h",
    )

    assert estimate == _micro_dollars(1_000 * 3)


def test_estimate_cost_preserves_a_free_cache_write_rate() -> None:
    estimate = estimate_cost(
        _pricing(cache_write_1h_price_per_million=0.0),
        prompt_chars=1_000 * 4,
        max_output_tokens=0,
        default_output_tokens=1_024,
        cache_write_ttl="1h",
    )

    assert estimate == _micro_dollars(1_000 * 3)


def test_estimate_tokens_sums_the_two_figures_the_cost_is_priced_from() -> None:
    """Prompt chars at four to the token, rounded up, plus the declared output bound."""
    assert estimate_tokens(prompt_chars=401, max_output_tokens=1_000, default_output_tokens=4_096) == 101 + 1_000


def test_estimate_tokens_falls_back_to_the_default_output_bound() -> None:
    """An unbounded request reserves the deployment's cap, as the cost estimate does."""
    assert estimate_tokens(prompt_chars=0, max_output_tokens=None, default_output_tokens=4_096) == 4_096


def test_estimate_tokens_treats_a_zero_output_bound_as_a_bound() -> None:
    """``max_output_tokens=0`` is an explicit "no output", not an omitted field."""
    assert estimate_tokens(prompt_chars=8, max_output_tokens=0, default_output_tokens=4_096) == 2


def test_estimate_tokens_clamps_hostile_inputs() -> None:
    """Negatives cannot shrink a hold: both figures floor at zero."""
    assert estimate_tokens(prompt_chars=-100, max_output_tokens=-50, default_output_tokens=4_096) == 0


def test_estimate_tokens_is_clamped_by_the_reserve_path_not_here() -> None:
    """The estimator reports what a request asks for; the gate is what bounds it.

    ``max_output_tokens`` is client-supplied and nothing on the wire limits it, so
    an enormous figure reaches here honestly and ``reserve_budget`` clamps it to
    ``MAX_COUNT_LIMIT`` before it can be added to a BIGINT counter.
    """
    asked = estimate_tokens(
        prompt_chars=0, max_output_tokens=10**18, default_output_tokens=4_096
    )

    assert asked == 10**18
    assert min(asked, MAX_COUNT_LIMIT) == MAX_COUNT_LIMIT


def test_the_count_limit_survives_the_round_trip_through_a_json_number() -> None:
    """The published schema carries this as a double, so it has to be exact as one.

    The BIGINT maximum is not: it renders as one *above* itself, so a client
    sending the maximum the spec advertises would send a value the column refuses.
    """
    assert int(float(MAX_COUNT_LIMIT)) == MAX_COUNT_LIMIT
    # And three of them still fit the column the gate sums into.
    assert 3 * MAX_COUNT_LIMIT < 2**63 - 1


def _axis(budget: Budget, *, new_request: bool = True, **counters: object) -> str:
    """``_blocked_axis`` with every counter at zero unless the case sets it."""
    base: dict[str, object] = {
        "spend": Decimal(0),
        "reserved": Decimal(0),
        "tokens": 0,
        "reserved_tokens": 0,
        "requests": 0,
        "reserved_requests": 0,
        # A dollar amount of zero, not an absent one: `None` says the request has
        # no dollar axis at all, which is the free-model case tested separately.
        "amount": Decimal(0),
        "held_tokens": 0,
        "held_requests": 0,
    }
    base.update(counters)
    return _blocked_axis(budget, new_request=new_request, **base)  # type: ignore[arg-type]


def test_the_refusal_names_an_axis_already_at_its_cap() -> None:
    """The commonest refusal of all, and the one a "would exceed" test alone misses.

    A user at their cap is refused for a request that holds nothing, so the axis
    has to be found by the gate's strict clause rather than by arithmetic on the
    hold. The dollar axis stays "budget", so the message a dollar refusal has
    always sent is unchanged.
    """
    assert _axis(Budget(budget_id="b", max_budget=10.0), spend=10.0) == "budget"
    assert _axis(Budget(budget_id="b", token_limit=1_000), tokens=1_000) == "token"
    assert _axis(Budget(budget_id="b", request_limit=2), requests=2, held_requests=1) == "request"


def test_the_refusal_names_the_axis_a_hold_would_push_past() -> None:
    budget = Budget(budget_id="b", max_budget=10.0, token_limit=1_000)

    assert _axis(budget, spend=9.0, amount=Decimal("2")) == "budget"
    assert _axis(budget, tokens=900, held_tokens=200) == "token"


def test_the_refusal_names_the_spent_axis_not_the_one_with_room() -> None:
    """The point of naming it: a token cutover cannot debug "budget limit"."""
    budget = Budget(budget_id="b", max_budget=10.0, token_limit=5)

    assert _axis(budget, spend=1.0, tokens=5) == "token"


def test_a_money_counter_read_as_a_float_does_not_raise() -> None:
    """A session can still hold the float a caller assigned, and mixing raises.

    This is what made the refusal path a 500 instead of a 403 for a user exactly
    at their cap.
    """
    assert _axis(Budget(budget_id="b", max_budget=10.0), spend=10.0, reserved=Decimal(0)) == "budget"


def test_a_top_up_is_not_asked_the_arrival_question() -> None:
    """Its own hold is what filled the axis, so "already at the cap" is not its test.

    "budget" is also the no-axis-found answer, so this asserts the token axis
    instead, where the two are distinguishable.
    """
    budget = Budget(budget_id="b", max_budget=None, token_limit=1_000)

    assert _axis(budget, tokens=1_000, new_request=True) == "token"
    assert _axis(budget, tokens=1_000, new_request=False) == "budget"


def test_an_absent_dollar_amount_removes_that_axis_from_the_answer() -> None:
    """A free request is never gated on dollars, so dollars cannot have refused it.

    Both caps are spent here, so naming either would look plausible; only the
    token one was actually asked.
    """
    budget = Budget(budget_id="b", max_budget=10.0, token_limit=1_000)

    # Spelled out rather than unpacked from a dict: `_axis` takes a keyword-only
    # `new_request`, so a `**kwargs` of floats is a type error against it.
    assert _axis(budget, amount=Decimal(0), spend=10.0, tokens=1_000) == "budget"
    assert _axis(budget, amount=None, spend=10.0, tokens=1_000) == "token"


def test_an_absent_dollar_amount_with_no_other_cap_names_nothing() -> None:
    """Rather than falling back to the axis it was not gated on."""
    assert _axis(Budget(budget_id="b", max_budget=10.0), amount=None, spend=10.0) == "budget"
