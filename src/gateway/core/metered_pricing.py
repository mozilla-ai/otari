"""The single cost-math core: provider-neutral token meters and threshold rates.

Every cost the gateway derives from a resolved rate row goes through this
module: settlement, the reserve-time upper bound, repricing, and imported
usage. It takes no database access and knows nothing about the pricing lookup
chain (``services/pricing_service.py``), so one implementation of the
arithmetic serves a stored rate, an organization override, a models.dev
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

A rate row whose ``unit`` is ``request`` is the exception: its
``input_price_per_million`` is USD per million requests, and a request costs
that divided by a million whatever tokens it reported (:func:`price_request`).

``pricing`` is read structurally (``getattr``) rather than through a declared
type, so a stored ``ModelPricing``, a transient rate row, and a test double all
work without this module importing the ORM.

Adopted from the platform's reconciled core (``app/core/metered_pricing.py`` in
mozilla-ai/otari-ai), which is where the two conventions were reconciled;
mozilla-ai/otari#661.
"""

import re
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
from gateway.types.models_dev import ModelsDevPricing, ModelsDevUnitPrice
from gateway.types.price_unit import PRICE_PER, PriceUnit, current_unit_name, models_dev_unit

RATE_FIELDS: tuple[str, ...] = (
    "input_price_per_million",
    "output_price_per_million",
    "cache_read_price_per_million",
    "cache_write_price_per_million",
)

# Key naming the lower bound of a threshold tier in a ``pricing_tiers`` entry.
TIER_THRESHOLD_FIELD = "min_input_tokens"

TOKENS_PER_PRICING_UNIT = Decimal(1_000_000)

# Anthropic bills a 1h cache write at twice base input, against 1.25x for a 5m
# one. No price source carries a 1h rate (models.dev has no field for it), so
# a Claude model's is derived from its input rate.
CLAUDE_1H_CACHE_WRITE_INPUT_MULTIPLE = Decimal(2)

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


@dataclass(frozen=True)
class Rates:
    """The per-million rates that apply to one usage event, tiers resolved."""

    input_price_per_million: Decimal
    output_price_per_million: Decimal
    cache_read_price_per_million: Decimal | None
    cache_write_price_per_million: Decimal | None
    cache_write_1h_price_per_million: Decimal | None
    """``None`` when a 1h write bills at ``cache_write_price_per_million`` like any other write."""


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


_BEDROCK_VERSION_SUFFIX = re.compile(r"[0-9]+(?::|$)")


def _is_claude(pricing: typing.Any) -> bool:
    """Whether ``pricing`` prices an Anthropic Claude model, on whichever provider serves it.

    Read from the model half of ``model_key`` (the whole key when it has no
    provider, see below), because the provider half names
    an instance rather than a vendor, and an Anthropic-compatible endpoint can
    serve another vendor's model. ``anthropic:claude-haiku-5-5``,
    ``bedrock:us.anthropic.claude-haiku-5-5`` and
    ``vertexaianthropic:claude-haiku-5-5@default`` all qualify.

    A key without a provider is the bare model id, and some model ids carry a
    colon of their own: a Bedrock id ends in a ``:<n>`` version and
    can carry a context-window suffix after it, as in
    ``us.anthropic.claude-haiku-5-5-v1:0`` or ``anthropic.claude-3-sonnet-20240229-v1:0:200k``.
    When what follows the first colon starts with ASCII digits and then ends or
    meets another colon, it is that version, not a model, so the whole key is the model.
    """
    model_key = getattr(pricing, "model_key", None)
    if not isinstance(model_key, str):
        return False
    _, separator, model = model_key.partition(":")
    if not separator or _BEDROCK_VERSION_SUFFIX.match(model):
        return "claude" in model_key.lower()
    return "claude" in model.lower()


def effective_rates(pricing: typing.Any, total_input_tokens: int) -> Rates:
    """Resolve the rates for a request, applying any threshold tier it reaches.

    A tier overrides only the rate fields it names, so a tier that reprices
    input alone leaves the cache and output rates on their base values.

    A Claude model whose cache writes are priced above zero bills a 1h write at
    twice the input rate the request reaches. Any other rate bills a 1h write as
    an ordinary cache write.
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
    cache_write_rate = values["cache_write_price_per_million"]
    return Rates(
        input_price_per_million=input_rate,
        output_price_per_million=output_rate,
        cache_read_price_per_million=values["cache_read_price_per_million"],
        cache_write_price_per_million=cache_write_rate,
        cache_write_1h_price_per_million=input_rate * CLAUDE_1H_CACHE_WRITE_INPUT_MULTIPLE
        if cache_write_rate and _is_claude(pricing)
        else None,
    )


def meter_cost(tokens: int | Decimal, rate: Decimal) -> Decimal:
    """USD for ``tokens`` tokens, or units of whatever a column is per million of, at a per-million rate, exactly.

    The multiplication happens before the division so the only inexact step
    would be a product wider than the decimal context, and dividing by a power
    of ten is an exponent shift rather than a rounding.
    """
    return (Decimal(tokens) * rate) / TOKENS_PER_PRICING_UNIT


def _charge_line(meter: str, units: int, rate: Decimal, cost: Decimal) -> ChargeLine:
    return {"meter": meter, "units": units, "rate_per_million": float(rate), "cost": float(cost)}


REQUESTS_UNIT: PriceUnit = "request"


def priced_per_request(pricing: typing.Any) -> bool:
    """Whether ``pricing`` charges a flat amount per request rather than per token.

    A rate object without a ``unit`` (a transient row built for manual
    repricing, a test double) reads as token pricing. A retired spelling of
    the unit reads as the current one.
    """
    return current_unit_name(getattr(pricing, "unit", None)) == REQUESTS_UNIT


def request_cost(pricing: typing.Any) -> Decimal:
    """USD for one request at a per-request rate, exactly.

    ``input_price_per_million`` is USD per million requests, so one request
    costs a millionth of it.
    """
    rate = to_decimal(getattr(pricing, "input_price_per_million", None))
    if rate is None:
        raise ValueError("Pricing carries no usable per-request rate")
    return meter_cost(1, rate)


def request_charge_line(cost: Decimal) -> ChargeLine:
    """The charge line for one request billed at ``cost``.

    It carries ``unit_rate`` (USD per request) where a token line carries
    ``rate_per_million``, which is what tells a reader which unit applies.
    """
    return {"meter": "request", "units": 1, "unit_rate": float(cost), "cost": float(cost)}


def price_request(pricing: typing.Any) -> tuple[Decimal, dict[str, int], list[ChargeLine]]:
    """Settle one request at a per-request rate: the cost, the meters, the charge lines.

    A free request carries no charge line, so the breakdown never shows a
    billed meter explaining a charge that did not happen.
    """
    cost = quantize_cost(request_cost(pricing))
    return cost, {"requests": 1}, [request_charge_line(cost)] if cost else []


PriceKind = typing.Literal["price", "reasoning", "cache_read", "cache_write", "cache_storage"]
"""Which price of one slot a meter reads: the base price, or its reasoning, cache read, write or storage price."""

Direction = typing.Literal["input", "output"]


@dataclass(frozen=True)
class UnitRate:
    """One published price: USD per ``PRICE_PER[unit]`` of ``unit``."""

    unit: PriceUnit
    price: Decimal

    def cost(self, quantity: int | Decimal) -> Decimal:
        """USD for ``quantity`` of the unit, exactly."""
        return unit_cost(quantity, self.unit, self.price)

    def charge_line(self, meter: str, quantity: int | Decimal, cost: Decimal) -> ChargeLine:
        """The line explaining ``cost`` for ``quantity`` at this rate.

        A unit priced per million carries ``rate_per_million``, as a token line
        does; any other carries ``unit_rate``, USD per one of the unit, as a
        request line does. ``unit`` names which.
        """
        units = int(quantity) if Decimal(quantity) == int(quantity) else float(quantity)
        line: ChargeLine = {"meter": meter, "units": units, "unit": self.unit}
        if PRICE_PER[self.unit] == TOKENS_PER_PRICING_UNIT:
            line["rate_per_million"] = float(self.price)
        else:
            line["unit_rate"] = float(self.price / Decimal(PRICE_PER[self.unit]))
        line["cost"] = float(cost)
        return line


@dataclass(frozen=True)
class UnitCharge:
    """What a quantity costs at one price: the exact amount, and the line that explains it when there is one."""

    cost: Decimal
    lines: list[ChargeLine]


def _quantity(value: int | Decimal) -> Decimal:
    quantity = Decimal(value)
    if not quantity.is_finite() or quantity < 0:
        raise ValueError(f"a quantity to price must be a finite non-negative number, not {value}")
    return quantity


def _estimated(value: int | Decimal) -> Decimal:
    """An estimated quantity, clamped at zero as a token count is, so a hostile value cannot hold a credit."""
    quantity = Decimal(value)
    if not quantity.is_finite():
        raise ValueError(f"an estimated quantity must be a finite number, not {value}")
    return max(quantity, Decimal(0))


def unit_cost(quantity: int | Decimal, unit: PriceUnit, price: Decimal) -> Decimal:
    """USD for ``quantity`` of ``unit`` at a published price, exactly.

    A published price is per ``PRICE_PER[unit]`` of the unit: per million
    tokens or characters, per thousand searches, per one image, megapixel,
    second or minute. ``quantity`` may be fractional, as seconds and megapixels
    are, and must be finite and non-negative.
    """
    return (_quantity(quantity) * price) / Decimal(PRICE_PER[unit])


def slot_price(pricing: ModelsDevPricing, modality: str, direction: Direction) -> ModelsDevUnitPrice | None:
    """The price a cost block gives ``modality`` in ``direction``, or ``None`` where it gives none."""
    prices = pricing.modalities.get(modality)
    if prices is None:
        return None
    return prices.input if direction == "input" else prices.output


def unit_rate(
    price: ModelsDevUnitPrice, kind: PriceKind = "price", *, cache_write_ttl: str | None = None
) -> UnitRate | None:
    """The rate ``kind`` of one slot prices at, or ``None`` when it prices nothing there.

    A slot in a unit no meter knows prices nothing, whatever it says: a new unit
    upstream is named in the catalog's issues and never billed as tokens. A
    cache write on a TTL the slot does not price bills as a plain write. A value
    that is not a believable price reads as absent.
    """
    unit = models_dev_unit(price.unit)
    if unit is None:
        return None
    cache = price.cache
    value: object
    if kind == "price":
        value = price.price
    elif kind == "reasoning":
        value = price.reasoning
    elif cache is None:
        value = None
    elif kind == "cache_read":
        value = cache.read
    elif kind == "cache_storage":
        value = cache.storage
    else:
        # A TTL price that is no believable price reads as absent, so the TTL bills as a plain write.
        listed = next((write for ttl, write in cache.write_ttls if ttl == cache_write_ttl), None)
        value = listed if to_decimal(listed) is not None else cache.write
    rate = to_decimal(value)
    return None if rate is None else UnitRate(unit, rate)


def web_search_rate(pricing: ModelsDevPricing) -> UnitRate | None:
    """The rate a cost block prices a web search at, USD per thousand searches, or ``None``."""
    if pricing.web_search is None:
        return None
    rate = to_decimal(pricing.web_search.price)
    return None if rate is None else UnitRate("web_search", rate)


def charge_unit(rate: UnitRate, meter: str, quantity: int | Decimal) -> UnitCharge:
    """``quantity`` of the unit at ``rate``, exactly, with a line under ``meter`` when there is something to explain.

    Not rounded: a caller that prices several meters into one row sums the
    exact amounts and applies :func:`quantize_cost` once.
    """
    quantity = _quantity(quantity)
    cost = rate.cost(quantity)
    return UnitCharge(cost, [rate.charge_line(meter, quantity, cost)] if quantity else [])


def charge_unit_price(
    price: ModelsDevUnitPrice,
    quantity: int | Decimal,
    *,
    meter: str,
    kind: PriceKind = "price",
    cache_write_ttl: str | None = None,
) -> UnitCharge | None:
    """``quantity`` at the ``kind`` price of one slot, or ``None`` when the slot prices nothing there."""
    rate = unit_rate(price, kind, cache_write_ttl=cache_write_ttl)
    return None if rate is None else charge_unit(rate, meter, quantity)


def estimate_unit_cost(
    price: ModelsDevUnitPrice, quantity: int | Decimal, *, cache_write_ttl: str | None = None
) -> Decimal:
    """Conservatively price ``quantity`` of one slot's unit before usage is known.

    Every unit could bill at the base price, the reasoning price, a cache read
    or a cache write on the requested TTL, so the bound is the quantity at the
    dearest of them. A slot that prices nothing estimates nothing.
    """
    kinds: tuple[PriceKind, ...] = ("price", "reasoning", "cache_read")
    if cache_write_ttl is not None:
        kinds = (*kinds, "cache_write")
    rates = [rate for kind in kinds if (rate := unit_rate(price, kind, cache_write_ttl=cache_write_ttl)) is not None]
    if not rates:
        return Decimal(0)
    dearest = max(rates, key=lambda rate: rate.price)
    return quantize_cost(dearest.cost(_estimated(quantity)))


def _price_meters(
    pricing: typing.Any,
    usage: BillableUsage,
) -> tuple[Decimal, dict[str, int], list[ChargeLine]]:
    """The exact total, the meters, and the charge lines, before any rounding.

    Split out from :func:`price_billable_usage` so a caller that prices several
    sub-amounts into one row can sum them exactly and round the row once.

    Each input token bills exactly once, under whichever meter it belongs to: a
    cache read, a cache write, a 1h cache write where it has a rate of its own,
    or fresh input. A meter with no configured rate leaves its tokens in the
    fresh-input bucket, so an unpriced cache meter costs the input rate instead
    of nothing.
    """
    rates = effective_rates(pricing, usage.total_input_tokens)
    one_hour_write_tokens = usage.cache_write_1h_tokens if rates.cache_write_1h_price_per_million is not None else 0

    fresh_input_tokens = usage.total_input_tokens
    if rates.cache_read_price_per_million is not None:
        fresh_input_tokens -= usage.cache_read_tokens
    if rates.cache_write_price_per_million is not None:
        fresh_input_tokens -= usage.cache_write_tokens

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
        charge("cache_write", usage.cache_write_tokens - one_hour_write_tokens, rates.cache_write_price_per_million)
    if rates.cache_write_1h_price_per_million is not None:
        charge("cache_write_1h", one_hour_write_tokens, rates.cache_write_1h_price_per_million)
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
    estimated_units: int | Decimal | None = None,
) -> Decimal:
    """Conservatively price a request before provider usage is available.

    Every estimated prompt token is billed as exactly one of fresh input, a
    cache read, or a cache write, so the upper bound for the input side is the
    token count times the dearest rate any of them bills at. A ``1h`` TTL means
    at least one block asked for a 1h write, so the others may still write for
    5m, and both write rates are candidates. Every rate is considered,
    including the cache-read rate: it is normally a discount, but
    nothing in the schema requires that, and assuming it made the estimate an
    under-bound for a rate card that priced a cache read above fresh input. An
    under-bound is a budget hole rather than a rounding difference, because the
    ceiling admits a request against the estimate and the settlement can then
    exceed it. Threshold rates are selected from the estimated input, which
    approximates the request's billable total. The estimate is reconciled to
    actual usage on completion.

    A rate priced per request is estimated at exactly one request, which is
    what it settles at. A rate in any other unit than tokens is estimated at
    ``estimated_units`` of it, at the dearest rate the row holds for the unit,
    the output rate included since one quantity cannot say which side it is on,
    so a reservation is never zero for a priced model. A caller that gives no
    quantity gets the token estimate.
    """
    if priced_per_request(pricing):
        return quantize_cost(request_cost(pricing))
    input_tokens = max(estimated_input_tokens, 0)
    unit = current_unit_name(getattr(pricing, "unit", None)) or "token"
    if unit != "token" and estimated_units is not None:
        rates = effective_rates(pricing, input_tokens)
        per_unit = [
            rates.input_price_per_million,
            rates.output_price_per_million,
            rates.cache_read_price_per_million,
        ]
        if cache_write_ttl is not None:
            per_unit.append(rates.cache_write_price_per_million)
        if cache_write_ttl == "1h":
            per_unit.append(rates.cache_write_1h_price_per_million)
        return quantize_cost(
            meter_cost(_estimated(estimated_units), max(rate for rate in per_unit if rate is not None))
        )
    output_tokens = max(estimated_output_tokens, 0)
    rates = effective_rates(pricing, input_tokens)

    # An unpriced meter bills at the input rate (its tokens stay in the fresh
    # bucket), so the input rate is the floor and every configured rate a prompt
    # token could attract raises it.
    candidate_rates = [rates.input_price_per_million]
    if rates.cache_read_price_per_million is not None:
        candidate_rates.append(rates.cache_read_price_per_million)
    if cache_write_ttl is not None and rates.cache_write_price_per_million is not None:
        candidate_rates.append(rates.cache_write_price_per_million)
    if cache_write_ttl == "1h" and rates.cache_write_1h_price_per_million is not None:
        candidate_rates.append(rates.cache_write_1h_price_per_million)
    per_input_token_rate = max(candidate_rates)

    estimate = meter_cost(input_tokens, per_input_token_rate) + meter_cost(
        output_tokens, rates.output_price_per_million
    )
    return quantize_cost(estimate)
