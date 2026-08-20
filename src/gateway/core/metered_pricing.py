"""The single cost-math core: provider-neutral token meters and threshold rates.

Every cost the gateway derives from a resolved rate row goes through this
module: settlement, the reserve-time upper bound, repricing, and imported
usage. It takes no database access and knows nothing about the pricing lookup
chain (``services/pricing_service.py``), so one implementation of the
arithmetic serves a stored rate, an organization override, a genai-prices
default, and a transient rate built in a test, and a second one cannot drift
from it.

The arithmetic is ``Decimal`` throughout. Binary floating point cannot
represent a rate like ``0.075`` or a cost like ``0.000001`` exactly, and a
settled row is accounting truth (mozilla-ai/otari-ai#1751), so the error has
nowhere to be absorbed. ``Decimal`` also makes the rounding a decision rather
than an artifact: see :func:`quantize_cost`.

Two cached-token conventions are represented, because providers disagree about
which one they speak:

``cache_tokens_included=True``
    ``input_tokens`` already counts the cached buckets. This is the OpenAI
    shape, where a cached slice of the prompt is a re-priced discount.
``cache_tokens_included=False``
    The cached buckets sit alongside ``input_tokens`` and add to the billable
    input total. This is the Anthropic shape.

Callers state which one they speak; there is no default. A wrong guess prices
a cached prompt twice or not at all, and the two shapes are indistinguishable
from the numbers alone. :func:`billable_usage_of` reads the convention off a
:class:`~gateway.core.usage.GatewayUsage` carrier, which is how the request
path answers the question.

Rates are USD per million tokens. A rate of ``None`` means that meter is not
priced separately, so its tokens stay in the fresh-input bucket and bill at the
input rate: an absent cache rate is never a discount.

``pricing`` is read structurally (``getattr``) rather than through a declared
type, so a stored ``ModelPricing``, a transient rate row, and a test double all
work without this module importing the ORM.

Adopted from the platform's reconciled core (``app/core/metered_pricing.py`` in
mozilla-ai/otari-ai), which is where the two conventions were reconciled;
mozilla-ai/otari#661.
"""

import typing
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from any_llm.types.completion import CompletionUsage

from gateway.core.usage import (
    cache_read_tokens_of,
    cache_tokens_in_prompt_of,
    cache_write_1h_tokens_of,
    cache_write_tokens_of,
)
from gateway.log_config import logger

RATE_FIELDS: tuple[str, ...] = (
    "input_price_per_million",
    "output_price_per_million",
    "cache_read_price_per_million",
    "cache_write_price_per_million",
    "cache_write_1h_price_per_million",
)

# Key naming the lower bound of a threshold tier in a ``pricing_tiers`` entry.
TIER_THRESHOLD_FIELD = "min_input_tokens"

TOKENS_PER_PRICING_UNIT = Decimal(1_000_000)

# The rounding this module defines, and the two column scales it matches.
#
# A settled cost is rounded once, half-up, to the micro-dollar: that is the
# scale of ``usage_logs.cost``, the accounting truth. Half-up rather than
# Python's default half-even because it is what money is conventionally rounded
# with and what an operator reproducing a bill by hand will do.
#
# A rate keeps 1e-8 USD per million tokens, four orders of magnitude finer than
# any published rate, so converting one from the float column it used to live
# in never moves it: a float's shortest decimal representation is what the
# migration casts, and every rate anyone has entered is a short decimal.
COST_SCALE = 6
COST_QUANTUM = Decimal(1).scaleb(-COST_SCALE)
RATE_SCALE = 8
RATE_QUANTUM = Decimal(1).scaleb(-RATE_SCALE)

ChargeLine = dict[str, typing.Any]
"""One auditable line of a row's ``pricing_breakdown``.

The numeric fields are ``float``: this is a JSON column rendered by the
dashboard, and JSON has no exact decimal. The line is the explanation, not the
amount; the amount is the row's ``cost`` column, which is exact.
"""


@dataclass(frozen=True)
class BillableUsage:
    """Canonical token meters, independent of how a provider reported them."""

    total_input_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cache_write_1h_tokens: int

    @property
    def cache_write_base_tokens(self) -> int:
        """Cache writes on the default TTL, i.e. every write that is not a 1h write."""
        return self.cache_write_tokens - self.cache_write_1h_tokens


@dataclass(frozen=True)
class Rates:
    """The per-million rates that apply to one usage event, tiers resolved."""

    input_price_per_million: Decimal
    output_price_per_million: Decimal
    cache_read_price_per_million: Decimal | None
    cache_write_price_per_million: Decimal | None
    cache_write_1h_price_per_million: Decimal | None


def to_decimal(value: typing.Any) -> Decimal | None:
    """Coerce a stored rate or amount to ``Decimal``, or ``None`` when unusable.

    Values reach the cost core from a Numeric column (already ``Decimal``), a
    JSON tier override (``int``, ``float``, or numeric string), and callers that
    still hold a ``float``. Going through ``str`` keeps a float from
    contributing its binary-representation error: ``Decimal(str(0.1))`` is
    ``0.1``, where ``Decimal(0.1)`` is ``0.1000000000000000055511151231257827``.

    A non-finite value (``NaN`` or an infinity, both of which ``Decimal`` parses
    happily) is rejected rather than carried: it would poison every total it
    touched without raising anywhere. So is a negative one, which would credit
    the payer for using the model. The rate columns have check constraints
    against that, but a tier override lives in a JSON column with no constraint
    to lean on, so the guard belongs at this one coercion point.

    Rejecting rather than clamping keeps the two cases distinct: a corrupt base
    rate leaves :func:`effective_rates` with nothing to price from and raises,
    while a corrupt cache or tier rate reads as absent and falls back to the
    base rate.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        candidate = value
    elif isinstance(value, int | float | str):
        try:
            candidate = Decimal(str(value))
        except ArithmeticError:
            return None
    else:
        return None
    return candidate if candidate.is_finite() and candidate >= 0 else None


def quantize_cost(value: Decimal) -> Decimal:
    """Round a computed amount to the micro-dollar, half-up.

    This is the only rounding in the cost path. The arithmetic before it is
    exact, and it is applied where an amount becomes a settled total: here, in
    :func:`price_billable_usage` and :func:`estimate_metered_cost`, in
    ``price_tool_calls``, and again by ``models.money.UsdCost`` on the way into
    the column, which is the backstop for a writer that reached the column
    another way. So the stored value is decided here rather than by whichever
    engine happens to be underneath, which round differently from each other.
    """
    return value.quantize(COST_QUANTUM, rounding=ROUND_HALF_UP)


def quantize_rate(value: Decimal) -> Decimal:
    """Round a stored rate to the rate column's scale, half-up."""
    return value.quantize(RATE_QUANTUM, rounding=ROUND_HALF_UP)


def billable_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_tokens_included: bool,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
) -> BillableUsage:
    """Normalize reported token counts into the billable meters.

    ``cache_tokens_included`` has no default on purpose; the module docstring
    says why.

    1h cache writes are a subset of cache writes, so a count larger than the
    write total is clamped to it rather than double-charged.

    Under the inclusive convention, cache buckets that together exceed
    ``input_tokens`` are contradictory: the cached tokens cannot outnumber the
    prompt that contains them. Rather than produce a negative fresh-input
    bucket, the cache attribution is dropped and the whole prompt bills at the
    input rate, which over-charges relative to any cache discount and never
    under-charges.
    """
    prompt_tokens = max(input_tokens, 0)
    completion_tokens = max(output_tokens, 0)
    cache_read = max(cache_read_tokens, 0)
    cache_write = max(cache_write_tokens, 0)
    cache_write_1h = min(max(cache_write_1h_tokens, 0), cache_write)

    if not cache_tokens_included:
        return BillableUsage(
            total_input_tokens=prompt_tokens + cache_read + cache_write,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cache_write_1h_tokens=cache_write_1h,
        )

    if cache_read + cache_write > prompt_tokens:
        logger.warning(
            "Cache token count exceeds input tokens (input=%s, cache_read=%s, cache_write=%s); "
            "falling back to non-cache input pricing",
            prompt_tokens,
            cache_read,
            cache_write,
        )
        cache_read = 0
        cache_write = 0
        cache_write_1h = 0

    return BillableUsage(
        total_input_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cache_write_1h_tokens=cache_write_1h,
    )


def billable_usage_of(usage: CompletionUsage) -> BillableUsage:
    """Normalize a provider usage carrier, reading the convention off the carrier.

    ``GatewayUsage.cache_tokens_in_prompt`` is where the request path records
    which convention the provider spoke, and a plain ``CompletionUsage`` reads
    as inclusive.
    """
    return billable_usage(
        input_tokens=max(int(usage.prompt_tokens or 0), 0),
        output_tokens=max(int(usage.completion_tokens or 0), 0),
        cache_read_tokens=cache_read_tokens_of(usage),
        cache_write_tokens=cache_write_tokens_of(usage),
        cache_write_1h_tokens=cache_write_1h_tokens_of(usage),
        cache_tokens_included=cache_tokens_in_prompt_of(usage),
    )


def _tier_threshold(tier: typing.Mapping[str, typing.Any]) -> int | None:
    """A tier's lower bound, or ``None`` when it cannot be read as one.

    An absent bound means the tier starts at zero. An unreadable or negative one
    is not treated as zero: that would turn a malformed entry into a tier every
    request reaches, repricing everything.
    """
    raw = tier.get(TIER_THRESHOLD_FIELD, 0)
    if raw is None:
        return 0
    try:
        threshold = int(raw)
    except (TypeError, ValueError):
        return None
    return threshold if threshold >= 0 else None


def _applicable_tier(
    tiers: typing.Any,
    total_input_tokens: int,
) -> typing.Mapping[str, typing.Any] | None:
    """The highest threshold tier this request reaches, or ``None`` for the base rates.

    Thresholds are compared against the whole request's billable input, which is
    how providers publish "cliff" pricing: crossing the boundary reprices the
    entire request rather than only the tokens past it.
    """
    if not isinstance(tiers, list):
        return None
    reached: list[tuple[int, typing.Mapping[str, typing.Any]]] = []
    for tier in tiers:
        if not isinstance(tier, dict):
            continue
        threshold = _tier_threshold(tier)
        if threshold is not None and threshold <= total_input_tokens:
            reached.append((threshold, tier))
    if not reached:
        return None
    return max(reached, key=lambda item: item[0])[1]


def effective_rates(pricing: typing.Any, total_input_tokens: int) -> Rates:
    """Resolve the rates for a request, applying any threshold tier it reaches.

    A tier overrides only the rate fields it names, so a tier that reprices
    input alone leaves the cache and output rates on their base values.
    """
    values: dict[str, Decimal | None] = {field: to_decimal(getattr(pricing, field, None)) for field in RATE_FIELDS}

    tier = _applicable_tier(getattr(pricing, "pricing_tiers", None), total_input_tokens)
    if tier is not None:
        for field in RATE_FIELDS:
            override = to_decimal(tier.get(field))
            if override is not None:
                values[field] = override

    input_rate = values["input_price_per_million"]
    output_rate = values["output_price_per_million"]
    if input_rate is None or output_rate is None:
        # Both columns are non-nullable everywhere rates are stored, so this is
        # a broken rate object rather than a priceable one. Defaulting to zero
        # would bill the request at nothing and say so nowhere.
        raise ValueError("Pricing carries no usable input or output rate")
    return Rates(
        input_price_per_million=input_rate,
        output_price_per_million=output_rate,
        cache_read_price_per_million=values["cache_read_price_per_million"],
        cache_write_price_per_million=values["cache_write_price_per_million"],
        cache_write_1h_price_per_million=values["cache_write_1h_price_per_million"],
    )


def meter_cost(tokens: int, rate: Decimal) -> Decimal:
    """USD for ``tokens`` tokens at a per-million rate, exactly.

    The multiplication happens before the division so the only inexact step
    would be a product wider than the decimal context, and dividing by a power
    of ten is an exponent shift rather than a rounding.
    """
    return (Decimal(tokens) * rate) / TOKENS_PER_PRICING_UNIT


def _charge_line(meter: str, units: int, rate: Decimal, cost: Decimal) -> ChargeLine:
    return {"meter": meter, "units": units, "rate_per_million": float(rate), "cost": float(cost)}


def _price_meters(
    pricing: typing.Any,
    usage: BillableUsage,
) -> tuple[Decimal, dict[str, int], list[ChargeLine]]:
    """The exact total, the meters, and the charge lines, before any rounding.

    Split out from :func:`price_billable_usage` so a caller that prices several
    sub-amounts into one row can sum them exactly and round the row once.

    Each input token bills exactly once, under whichever meter it belongs to: a
    cache read, a cache write on either TTL, or fresh input. A meter with no
    configured rate leaves its tokens in the fresh-input bucket, so an unpriced
    cache meter costs the input rate instead of nothing.
    """
    rates = effective_rates(pricing, usage.total_input_tokens)

    # A 1h write with no dedicated rate bills as an ordinary cache write.
    write_1h_rate = (
        rates.cache_write_1h_price_per_million
        if rates.cache_write_1h_price_per_million is not None
        else rates.cache_write_price_per_million
    )

    fresh_input_tokens = usage.total_input_tokens
    if rates.cache_read_price_per_million is not None:
        fresh_input_tokens -= usage.cache_read_tokens
    if rates.cache_write_price_per_million is not None:
        fresh_input_tokens -= usage.cache_write_base_tokens
    if write_1h_rate is not None:
        fresh_input_tokens -= usage.cache_write_1h_tokens

    meters = {
        "total_input_tokens": usage.total_input_tokens,
        "fresh_input_tokens": fresh_input_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cache_write_1h_tokens": usage.cache_write_1h_tokens,
        "completion_tokens": usage.completion_tokens,
    }
    lines: list[ChargeLine] = []
    cost = Decimal(0)

    def charge(meter: str, units: int, rate: Decimal) -> None:
        nonlocal cost
        amount = meter_cost(units, rate)
        cost += amount
        if units:
            lines.append(_charge_line(meter, units, rate, amount))

    charge("input", fresh_input_tokens, rates.input_price_per_million)
    charge("output", usage.completion_tokens, rates.output_price_per_million)
    if rates.cache_read_price_per_million is not None:
        charge("cache_read", usage.cache_read_tokens, rates.cache_read_price_per_million)
    if rates.cache_write_price_per_million is not None:
        charge("cache_write_5m", usage.cache_write_base_tokens, rates.cache_write_price_per_million)
    if write_1h_rate is not None:
        charge("cache_write_1h", usage.cache_write_1h_tokens, write_1h_rate)
    return cost, meters, lines


def price_billable_usage(
    pricing: typing.Any,
    usage: BillableUsage,
) -> tuple[Decimal, dict[str, int], list[ChargeLine]]:
    """Settle normalized token meters and return auditable charge lines.

    The returned cost is rounded once, by :func:`quantize_cost`, after the
    meters have been summed exactly. Charge lines carry the unrounded per-meter
    amount as a float, so a breakdown can differ from the row's total by less
    than half a micro-dollar; the column is the amount, the lines explain it.
    """
    cost, meters, lines = _price_meters(pricing, usage)
    return quantize_cost(cost), meters, lines


def calculate_metered_cost(
    pricing: typing.Any,
    usage: CompletionUsage,
) -> tuple[Decimal, dict[str, int], list[ChargeLine]]:
    """Price a provider usage carrier, taking its cached-token convention from it.

    The request path's entry point into the core. Callers holding loose token
    counts rather than a carrier state the convention themselves through
    :func:`billable_usage` and price with :func:`price_billable_usage`.
    """
    return price_billable_usage(pricing, billable_usage_of(usage))


def calculate_token_cost(
    pricing: typing.Any,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_tokens_included: bool,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
    quantize: bool = True,
) -> Decimal:
    """Price loose token counts, for a caller with no usage carrier and no breakdown.

    ``quantize=False`` returns the exact amount instead of a settled one, for a
    caller that prices several sub-amounts into a single row: rounding each of
    them would round the row once per sub-amount, which is what a batch of a
    thousand identical short requests turns into a visible shortfall. Such a
    caller sums the exact amounts and applies :func:`quantize_cost` once.
    """
    cost, _, _ = _price_meters(
        pricing,
        billable_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_write_1h_tokens=cache_write_1h_tokens,
            cache_tokens_included=cache_tokens_included,
        ),
    )
    return quantize_cost(cost) if quantize else cost


def estimate_metered_cost(
    pricing: typing.Any,
    *,
    estimated_input_tokens: int,
    estimated_output_tokens: int,
    cache_write_ttl: typing.Literal["5m", "1h"] | None = None,
) -> Decimal:
    """Conservatively price a request before provider usage is available.

    Every estimated prompt token is billed as exactly one of fresh input, a
    cache read, or a cache write, so the upper bound for the input side is the
    token count times the dearest rate it could attract. When a cache write is
    requested that worst case is the cache-write rate; otherwise it is the input
    rate, since a cache read is never dearer than fresh input. Threshold rates
    are selected from the estimated input, which approximates the request's
    billable total. The estimate is reconciled to actual usage on completion.
    """
    input_tokens = max(estimated_input_tokens, 0)
    output_tokens = max(estimated_output_tokens, 0)
    rates = effective_rates(pricing, input_tokens)

    if cache_write_ttl == "1h":
        cache_write_rate = rates.cache_write_1h_price_per_million
        if cache_write_rate is None:
            cache_write_rate = rates.cache_write_price_per_million
    elif cache_write_ttl == "5m":
        cache_write_rate = rates.cache_write_price_per_million
    else:
        cache_write_rate = None

    # An unpriced cache write bills at the input rate (it stays in the fresh
    # bucket), so the input rate is the floor either way.
    per_input_token_rate = rates.input_price_per_million
    if cache_write_rate is not None:
        per_input_token_rate = max(per_input_token_rate, cache_write_rate)

    estimate = meter_cost(input_tokens, per_input_token_rate) + meter_cost(
        output_tokens, rates.output_price_per_million
    )
    return quantize_cost(estimate)
