"""Response shapes for the billing detail carried on a usage row.

``UsageLog.billing_meters`` and ``UsageLog.pricing_breakdown`` are JSON columns,
and until now the response models described them only as "some object" and "a
list of some objects". That is what the wire contract said, so every generated
client (the dashboard's, and every SDK core, which come from the same spec) had
to hand back an untyped blob and let the caller guess. The shapes are not
actually open: six producers write charge lines and all six write one of the two
below.

Each field pairs its precise shapes with a permissive fallback arm, and that
fallback is load-bearing rather than defensive clutter. These models validate
rows read back out of a database that a self-hosted gateway has been
accumulating across upgrades, so a row written by an older version whose shape
nobody can inspect now must still render. A strict union would turn one such row
into a 500 for the whole usage page. The precise arms are what a client codegens
against; the fallback is what keeps an old row on the page.

For the same reason every model here allows extra keys: validating a stored line
into a model that silently dropped an unrecognised key would delete data on the
way out.
"""

from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class _OpenShape(BaseModel):
    """Base for a model describing JSON already written to a usage row.

    Extra keys are kept: validating a stored line into a model that dropped an
    unrecognised key would delete billing detail on the way out.
    """

    model_config = ConfigDict(extra="allow")


class TokenChargeLine(_OpenShape):
    """A charge line billed per million tokens.

    ``rate_per_million`` is the discriminator: its presence (rather than
    ``unit_rate``) is what tells a reader, and a renderer, which unit convention
    applies. See ``gateway.services.pricing_service``.
    """

    meter: str = Field(description="What was metered, e.g. 'input' or 'cache_read'.")
    units: int | float = Field(description="Tokens billed on this meter.")
    rate_per_million: float = Field(description="USD per million units.")
    cost: float = Field(description="USD charged for this line.")


class UnitChargeLine(_OpenShape):
    """A charge line billed per call, for gateway-run tools and per-request routes.

    ``unit_rate`` is the discriminator, in the same sense as
    :class:`TokenChargeLine`'s ``rate_per_million``.
    """

    meter: str = Field(description="What was metered, e.g. 'request' or 'web_search_calls'.")
    units: int | float = Field(description="Calls billed on this meter.")
    unit_rate: float = Field(description="USD per call.")
    cost: float = Field(description="USD charged for this line.")


class ToolMeter(TypedDict, total=False):
    """One gateway-run tool's call counts on a request.

    A TypedDict rather than a model, and every key optional, so a stored entry
    serializes back as it was written: a model with defaults would put
    ``"unit_rate": null`` on every tool that has never been priced.
    """

    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore[misc]

    #: Successful calls, which are charged.
    billed: int
    #: Failed calls, which are counted but never charged.
    errors: int
    #: USD per call at the time of the request.
    unit_rate: float | None


class BillingMeters(TypedDict, total=False):
    """The meters a request was billed on.

    Token meters sit flat (``{"input": 1200}``), which is why extra keys are
    allowed: the names come from the provider and the endpoint, so the map is
    genuinely open. Gateway-run tool counts are the one reserved key, nested
    under ``tools`` because an MCP server can advertise a tool named after a
    token meter and a flat collision would corrupt the billed-token aggregates
    for the whole window (see ``gateway.services.tool_usage``).

    A TypedDict for the reason above, and for one that cost a round trip to
    learn: a model that suppressed its defaults with a ``model_serializer``
    published an *empty* schema, because that serializer replaces the
    serialization JSON schema and that is the one FastAPI puts in the spec. The
    properties below are the whole point, so they have to survive into
    docs/public/openapi.json; ``tests/unit/test_billing_schemas.py`` asserts they
    do.
    """

    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore[misc]

    #: Per-tool call counts, keyed by tool name.
    tools: dict[str, ToolMeter]


# The permissive arms. Named rather than inlined so the fields that use them read
# as one decision, and so the reason lives in one place.
#
# `left_to_right` is what makes the precise arms real. Pydantic's default smart
# union prefers the arm needing no coercion, and a stored line is already a dict,
# so the fallback would win every time: the shapes would be published in the spec
# but never once used to validate. Trying them in order means a line that matches
# is parsed as what it is, and only a line that matches neither falls through.
ChargeLine = Annotated[
    # `dict[str, Any]`, not a narrower value type: the arm exists to accept a line
    # whose shape nobody can inspect now, so constraining what its values may hold
    # defeats it. A stored line carrying a null, a list or a nested object was
    # rejected by every arm and took the whole usage page down with a 500, which is
    # the exact failure this is here to prevent.
    TokenChargeLine | UnitChargeLine | dict[str, Any],
    Field(union_mode="left_to_right"),
]
MeterMap = Annotated[BillingMeters | dict[str, Any], Field(union_mode="left_to_right")]
