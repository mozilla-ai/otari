"""The prices one request pays: a rate's block with the overrides and multipliers the request meets applied.

A models.dev price is a base price plus rules. An override on an input, an
output or the web search price replaces the prices it names ``when`` its
condition holds, and a multiplier beside the modalities scales the prices it
names ``when`` its condition holds. models.dev's schema states the order:
matching overrides merge field by field, nested objects included, in order;
then every matching multiplier applies. A rule whose condition names a key no
fact here answers is skipped, as the schema says, and so is one whose
condition or factor cannot be read. A key a fact answers but the request gives
no value for is a condition that does not hold.

The block is resolved as the document writes it, so every modality, the web
search price and every price in them (reasoning, cache read, each cache write
TTL, storage, explicit caching) resolve alike, and the result reads back
through the one reader every rate goes through.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from gateway.models.pricing import ModelPricing
from gateway.repositories.pricing import transient_pricing
from gateway.types.models_dev import parse_pricing
from gateway.types.price_rates import columns_from_pricing
from gateway.types.price_unit import PRICE_UNITS

_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_DIRECTIONS = ("input", "output")
_COMPARISONS = ("gt", "gte", "lt", "lte")

# One rule's condition, as the document writes it.
Condition = Mapping[str, Any]


@dataclass(frozen=True)
class PriceFacts:
    """What one request says about itself that a price's conditions ask.

    ``None`` is a request with no value for that key. ``context`` is the
    request's billable input in tokens, ``duration`` a length in seconds, and
    ``at`` when the request started; the clock keys (``date``, ``utc_days``,
    ``utc_hours``) are read from it, in UTC, and from now when it is ``None``.
    """

    context: int | None = None
    service_tier: str | None = None
    resolution: str | None = None
    quality: str | None = None
    style: str | None = None
    aspect_ratio: str | None = None
    duration: int | Decimal | None = None
    at: datetime | None = None

    def values(self) -> dict[str, object]:
        """The value of every condition key a fact answers, ``None`` where the request gives none."""
        at = self.at or datetime.now(UTC)
        at = at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)
        return {
            "context": self.context,
            "service_tier": self.service_tier,
            "resolution": self.resolution,
            "quality": self.quality,
            "style": self.style,
            "aspect_ratio": self.aspect_ratio,
            "duration": self.duration,
            "date": at.date().isoformat(),
            "utc_days": _DAYS[at.weekday()],
            "utc_hours": at.strftime("%H:%M"),
            # Nothing Otari sends through any-llm carries an input video or runs over a websocket.
            "input_video": False,
            "websocket": False,
        }


@dataclass(frozen=True)
class ResolvedRates:
    """A block with each price final and no rule left, and the conditions that changed each price.

    ``rules`` is keyed by the price's path (``text.input``, ``image.output``,
    ``web_search``), each condition in the order its rule applied.
    """

    document: dict[str, Any]
    rules: Mapping[str, tuple[Condition, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class AppliedRate:
    """A rate with the rules its request meets applied, and the conditions that changed each price."""

    rate: ModelPricing
    rules: Mapping[str, tuple[Condition, ...]] = field(default_factory=dict)


def resolve_rates(document: Mapping[str, Any], facts: PriceFacts) -> ResolvedRates:
    """``document`` with the overrides, then the multipliers, that ``facts`` meet applied, as a new block."""
    values = facts.values()
    rules: dict[str, list[Condition]] = {}
    resolved: dict[str, Any] = {}
    for key, entry in document.items():
        if key == "multipliers":
            continue
        if key == "web_search":
            resolved[key] = _resolved_price(entry, values, rules, key)
        elif isinstance(entry, Mapping):
            resolved[key] = {
                direction: _resolved_price(price, values, rules, f"{key}.{direction}")
                if direction in _DIRECTIONS
                else copy.deepcopy(price)
                for direction, price in entry.items()
            }
        else:
            resolved[key] = copy.deepcopy(entry)
    multipliers = document.get("multipliers")
    for multiplier in multipliers if isinstance(multipliers, (list, tuple)) else ():
        if not isinstance(multiplier, Mapping) or not _a_factor(factor := multiplier.get("factor")):
            continue
        when = multiplier.get("when")
        if _holds(when, values) is not True:
            continue
        for path in _price_paths(resolved):
            if _reaches(factor, path):
                rules.setdefault(".".join(path), []).append(_plain(when))
        resolved = _scaled(resolved, factor)
    return ResolvedRates(resolved, {path: tuple(found) for path, found in rules.items()})


def apply_rules(rate: ModelPricing, facts: PriceFacts) -> AppliedRate:
    """``rate`` priced as the rules ``facts`` meet say: a transient row, never added to a session.

    Its ``rates`` is the resolved block, and its rate columns are read from
    it the way every writer derives them, so the token meters bill the
    resolved text prices and no ``pricing_tiers`` are left to reapply. A row
    with no block is priced by its columns as stored, and one whose block
    reads back in another unit than the row's keeps its columns.
    """
    if not rate.rates:
        return AppliedRate(rate)
    resolved = resolve_rates(rate.rates, facts)
    unit = rate.unit or "token"
    try:
        pricing, _ = parse_pricing(resolved.document, units=PRICE_UNITS, path="rates")
        columns = columns_from_pricing(pricing)
    except (TypeError, ValueError):
        columns = None
    if columns is None or columns.unit != unit:
        applied = transient_pricing(rate, rates=resolved.document)
    else:
        applied = transient_pricing(
            rate,
            rates=resolved.document,
            input_price_per_million=columns.input_price_per_million,
            output_price_per_million=columns.output_price_per_million,
            cache_read_price_per_million=columns.cache_read_price_per_million,
            cache_write_price_per_million=columns.cache_write_price_per_million,
            pricing_tiers=[dict(tier) for tier in columns.pricing_tiers],
        )
    return AppliedRate(applied, resolved.rules)


def _resolved_price(price: object, values: Mapping[str, object], rules: dict[str, list[Condition]], path: str) -> Any:
    """One price with each matching override merged over it, in order."""
    if not isinstance(price, Mapping):
        return copy.deepcopy(price)
    resolved = {key: copy.deepcopy(value) for key, value in price.items() if key != "overrides"}
    overrides = price.get("overrides")
    for override in overrides if isinstance(overrides, (list, tuple)) else ():
        if not isinstance(override, Mapping):
            continue
        when = override.get("when")
        if _holds(when, values) is not True:
            continue
        _merge(resolved, {key: value for key, value in override.items() if key not in ("when", "unit", "overrides")})
        rules.setdefault(path, []).append(_plain(when))
    return resolved


def _merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Merge ``source`` over ``target`` field by field: an object into an object, anything else replacing."""
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            inner = dict(target[key])
            _merge(inner, value)
            target[key] = inner
        else:
            target[key] = copy.deepcopy(value)


def _holds(when: object, values: Mapping[str, object]) -> bool | None:
    """Whether ``when`` holds for ``values``, or ``None`` when it names a key no fact answers or cannot be read.

    Every key is read before the answer is given, so a rule naming an unknown
    key anywhere is skipped rather than decided by the keys before it.
    """
    if not isinstance(when, Mapping):
        return None
    held = True
    for key, condition in when.items():
        found: bool | None
        if key in ("all", "any"):
            if not isinstance(condition, list):
                return None
            each = [_holds(item, values) for item in condition]
            if any(item is None for item in each):
                return None
            found = all(each) if key == "all" else any(each)
        elif key == "not":
            inner = _holds(condition, values)
            found = None if inner is None else not inner
        elif key in values:
            found = _meets(values[key], condition)
        else:
            found = None
        if found is None:
            return None
        held = held and found
    return held


def _meets(value: object, condition: object) -> bool | None:
    """Whether ``value`` meets one condition, or ``None`` when the condition is not one operator and its operand."""
    if not isinstance(condition, Mapping) or len(condition) != 1:
        return None
    ((operator, operand),) = condition.items()
    if operator == "equals":
        return None if not _scalar(operand) else value is not None and _same(value, operand)
    if operator == "in":
        if not isinstance(operand, list) or not all(_scalar(item) for item in operand):
            return None
        return value is not None and any(_same(value, item) for item in operand)
    if operator in _COMPARISONS:
        if isinstance(operand, bool) or not isinstance(operand, (str, int, float, Decimal)):
            return None
        return _compares(value, operator, operand)
    if operator == "min_items":
        if isinstance(operand, bool) or not isinstance(operand, int) or operand < 0:
            return None
        return isinstance(value, (list, tuple)) and len(value) >= operand
    return None


def _scalar(value: object) -> bool:
    return isinstance(value, (str, bool, int, float, Decimal))


def _same(value: object, operand: object) -> bool:
    """Equal as the schema compares: a bool to a bool, a number to a number, and text without regard to case."""
    if isinstance(value, bool) or isinstance(operand, bool):
        return isinstance(value, bool) and isinstance(operand, bool) and value == operand
    if (left := _decimal(value)) is not None and (right := _decimal(operand)) is not None:
        return left == right
    return isinstance(value, str) and isinstance(operand, str) and value.casefold() == operand.casefold()


def _compares(value: object, operator: str, operand: object) -> bool:
    """``value`` against ``operand`` by ``operator``: numbers as numbers, text lexicographically, else not."""
    left: Any
    right: Any
    if isinstance(value, str) and isinstance(operand, str):
        left, right = value, operand
    elif (left := _decimal(value)) is None or (right := _decimal(operand)) is None:
        return False
    if operator == "gt":
        return bool(left > right)
    if operator == "gte":
        return bool(left >= right)
    if operator == "lt":
        return bool(left < right)
    return bool(left <= right)


def _decimal(value: object) -> Decimal | None:
    """A number as a ``Decimal``, through its shortest text for a float, or ``None`` for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _a_factor(factor: object) -> bool:
    """Whether ``factor`` is a multiplier's factor: a number, zero or more, or an object of such factors."""
    if isinstance(factor, Mapping):
        return bool(factor) and all(_a_factor(item) for item in factor.values())
    number = _decimal(factor)
    return number is not None and number >= 0


def _price_paths(document: Mapping[str, Any]) -> list[tuple[str, ...]]:
    """The path of every price a resolved block holds: each modality's input and output, and the web search price."""
    paths: list[tuple[str, ...]] = []
    for key, entry in document.items():
        if key == "web_search":
            paths.append((key,))
        elif isinstance(entry, Mapping):
            paths.extend((key, direction) for direction in _DIRECTIONS if direction in entry)
    return paths


def _reaches(factor: object, path: tuple[str, ...]) -> bool:
    """Whether ``factor`` scales some price at ``path``."""
    for step in path:
        if not isinstance(factor, Mapping):
            return True
        if step not in factor:
            return False
        factor = factor[step]
    return True


def _scaled(node: Any, factor: object) -> Any:
    """``node`` with the prices ``factor`` names multiplied: a number scales every price under it."""
    if isinstance(factor, Mapping):
        if not isinstance(node, Mapping):
            return node
        return {key: _scaled(value, factor[key]) if key in factor else value for key, value in node.items()}
    multiple = _decimal(factor)
    return node if multiple is None else _times(node, multiple)


def _times(node: Any, multiple: Decimal) -> Any:
    if isinstance(node, Mapping):
        return {key: value if key == "unit" else _times(value, multiple) for key, value in node.items()}
    number = _decimal(node)
    return node if number is None else number * multiple


def _plain(value: object) -> Any:
    """A condition as JSON holds it: each ``Decimal`` a stored rate read back as the number it was written as."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return value
