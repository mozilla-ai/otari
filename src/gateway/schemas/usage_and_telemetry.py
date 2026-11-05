"""Response shapes for usage rows: the billing detail a row carries, and the reads over the rows.

``UsageLog.billing_meters`` and ``UsageLog.pricing_breakdown`` are JSON columns,
and until now the response models described them only as "some object" and "a
list of some objects". That is what the wire contract said, so every generated
client (the dashboard's, and every SDK core, which come from the same spec) had
to hand back an untyped blob and let the caller guess. The shapes are not
actually open: six producers write charge lines and all six write one of the
shapes below.

Each field pairs its precise shapes with a permissive fallback arm, and that
fallback is load-bearing rather than defensive clutter. These models validate
rows read back out of a database that a self-hosted gateway has been
accumulating across upgrades, so a row written by an older version whose shape
nobody can inspect now must still render. A strict union would turn one such row
into a 500 for the whole usage page. The precise arms are what a client codegens
against; the fallback is what keeps an old row on the page.

For the same reason every billing-detail model here allows extra keys: validating a stored line
into a model that silently dropped an unrecognised key would delete data on the
way out.
"""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypedDict, get_args

from pydantic import BaseModel, ConfigDict, Field

from gateway.core.sql import utc_iso
from gateway.models.money import as_float
from gateway.models.usage import SERVED_HERE_SLUG, UsageLog

# The meter namespace inside ``UsageLog.billing_meters``. Tool meters are nested
# under one reserved key rather than sitting flat next to the token meters:
# MCP tool names come from a caller-supplied server, and a tool named
# ``completion_tokens`` sitting flat would be picked up by the billed-token SQL
# of the usage aggregates and corrupt them for the whole window.
TOOL_METER_NAMESPACE = "tools"


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


class ToolChargeLine(UnitChargeLine):
    """A gateway-run tool's charge line, one per tool, label and payer.

    A subclass with ``label`` required, rather than a defaulted field on
    :class:`UnitChargeLine`, so the per-request, rerank and image lines, and a
    stored tool line without a label, serialize back without a
    ``"label": null`` they never had.

    A line written since tools are charged by whose account ran them also
    carries ``deployment_paid``: true where the deployment's account ran the
    calls, which charges the organization's funds where the deployment bills
    them, and false where the organization's own key did. It is kept as an extra
    key rather than declared, so an older line serializes back without it.
    """

    label: str = Field(
        description=(
            "What served the calls: '<provider>:<instance>' for a search or fetch instance, "
            "'builtin:builtin_fetch' for the built-in fetcher, 'org:<provider>' for an organization's "
            "own key, or 'otari:<tool>' for any other gateway-run tool."
        )
    )


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
    #: USD per call at the time of the request: the tool's cost over its billed
    #: calls, so billed calls times this rate is its spend across every label.
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
    #
    # `ToolChargeLine` before `UnitChargeLine`, whose every field it has, or no
    # line would ever be parsed as a tool line.
    ToolChargeLine | TokenChargeLine | UnitChargeLine | dict[str, Any],
    Field(union_mode="left_to_right"),
]
MeterMap = Annotated[BillingMeters | dict[str, Any], Field(union_mode="left_to_right")]


Bucket = Literal["hour", "day"]
SeriesGroupBy = Literal["model", "user_id", "api_key_id", "source"]

# Coarse display buckets for a failure's status code. A closed Literal rather than
# a bare str so the set lands in the OpenAPI schema as an enum and a consumer can
# switch on it exhaustively instead of string-matching whatever the server sent.
ErrorClass = Literal["pricing", "rate_limit", "auth", "provider_error", "client_error", "unknown"]

# The gateway-run tools that can be enumerated for a filter or a breakdown. MCP
# tool names come from a caller-supplied server, so they are unbounded and appear
# only in a row's own detail, never as a dimension of their own. The ``any``
# selector still matches them, because it tests the meter namespace itself.
ToolFilter = Literal["any", "web_search", "web_fetch", "code_execution"]
ANY_TOOL = "any"
GATEWAY_TOOL_NAMES: tuple[str, ...] = tuple(tool for tool in get_args(ToolFilter) if tool != ANY_TOOL)

# Every breakdown a summary can compute; the extra ``none`` is the explicit empty
# selection (a repeated query param cannot express an empty list on the wire).
SummaryDimension = Literal[
    "model", "user", "api_key", "source", "source_label", "endpoint", "provider", "status_code", "tool", "none"
]


class UsageEntry(BaseModel):
    """A single usage log entry."""

    id: str
    user_id: str | None
    # Display labels resolved server-side, so a client rendering a page of rows
    # does not have to hold the whole users/api_keys tables to name them. Null
    # when the row has no owner, when the referenced row is gone (both foreign
    # keys are ON DELETE SET NULL), or when the entity simply has no label set;
    # a client falls back to the id in every one of those cases.
    user_alias: str | None = None
    api_key_id: str | None
    api_key_name: str | None = None
    timestamp: str
    model: str
    provider: str | None
    endpoint: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    cache_write_1h_tokens: int | None
    # A subset of completion_tokens; null on rows written before it was recorded.
    reasoning_tokens: int | None = None
    # Precise shapes with a permissive fallback arm; see schemas/usage_and_telemetry.py for why
    # the fallback is what keeps a row written by an older gateway renderable.
    billing_meters: MeterMap | None
    pricing_breakdown: Sequence[ChargeLine] | None
    cost: float | None
    status: str
    error_message: str | None
    status_code: int | None
    latency_ms: int | None
    provider_latency_ms: int | None
    source: str
    source_label: str | None
    # The request's tags (from its ``metadata``); null when it sent none.
    tags: dict[str, str] | None = None
    counts_toward_budget: bool
    # Whether the bulk operator mutations can reach this row: the fixed scope
    # ``_selection_conditions`` pins them to, which is provenance *and* budget
    # participation. Derived here rather than left to the client, because a client
    # composing it from the two fields above is a second copy of the rule, and the
    # copy is what let the dashboard offer a checkbox the delete then refused (#781).
    bulk_editable: bool
    # Routing attribution. All null for a request that named a plain model, except
    # `request_group_id`, which is the request's `Otari-Request-ID` on every row.
    # `status == "absorbed"` marks an attempt a policy recovered from; those rows
    # are excluded from `error_count` and from `request_count`, since the request
    # they belong to is counted once by the attempt that served it.
    policy_name: str | None = None
    selection_reason: str | None = None
    attempt_position: int | None = None
    attempt_count: int | None = None
    request_group_id: str | None = None

    @classmethod
    def from_model(
        cls,
        log: UsageLog,
        *,
        user_alias: str | None = None,
        api_key_name: str | None = None,
    ) -> "UsageEntry":
        return cls(
            id=log.id,
            user_id=log.user_id,
            user_alias=user_alias,
            api_key_id=log.api_key_id,
            api_key_name=api_key_name,
            timestamp=utc_iso(log.timestamp),
            model=log.model,
            provider=log.provider,
            endpoint=log.endpoint,
            source=log.source,
            source_label=log.source_label,
            tags=log.tags,
            counts_toward_budget=log.counts_toward_budget,
            bulk_editable=log.source != SERVED_HERE_SLUG and not log.counts_toward_budget,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            cache_read_tokens=log.cache_read_tokens,
            cache_write_tokens=log.cache_write_tokens,
            cache_write_1h_tokens=log.cache_write_1h_tokens,
            reasoning_tokens=log.reasoning_tokens,
            billing_meters=log.billing_meters,
            pricing_breakdown=log.pricing_breakdown,
            cost=as_float(log.cost),
            status=log.status,
            error_message=log.error_message,
            status_code=log.status_code,
            latency_ms=log.latency_ms,
            provider_latency_ms=log.provider_latency_ms,
            policy_name=log.policy_name,
            selection_reason=log.selection_reason,
            attempt_position=log.attempt_position,
            attempt_count=log.attempt_count,
            request_group_id=log.request_group_id,
        )


class UsageCount(BaseModel):
    """Total number of usage logs matching a set of filters."""

    total: int


class RequestSettlement(BaseModel):
    """What one request settled at, summed over every usage row it wrote.

    A routed request writes a row per attempt and a vision-normalized one a row for
    the describe call, all sharing the ``Otari-Request-ID`` the caller was sent as
    their ``request_group_id``, so this is the request's whole bill rather than one
    attempt's. ``cost_usd`` uses the inline ``usage.cost_usd`` format and is null
    when no row was priced.
    """

    request_id: str
    status: Literal["success", "error"]
    cost_usd: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    row_count: int


class UsageTotals(BaseModel):
    """Grand totals over the filtered window."""

    cost: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cache_write_1h_tokens: int
    reasoning_tokens: int = 0
    request_count: int
    error_count: int
    avg_latency_ms: float | None
    # Served rows with no configured price (cost is NULL), e.g. imported usage for
    # an unpriced model. Surfaced so a $0 cost is not mistaken for free usage.
    # Scoped to status="success": a gateway-side rejection also carries cost=NULL
    # (nothing was spent), so counting error rows here would make a budget or
    # allow-list incident read as a pricing misconfiguration.
    unpriced_requests: int = 0
    # Billed input tokens (fresh input plus both cache buckets), normalized via
    # each row's billing meters where present (see ``_billed_expr``). Unlike
    # ``prompt_tokens`` this is convention-independent, so cache hit rate
    # (cache_read_tokens / billed_input_tokens) is meaningful across providers.
    billed_input_tokens: int = 0
    # Billed output tokens, normalized the same way, so the breakdown fold row
    # reconciles against the same quantity the per-group rows sum (a raw
    # ``completion_tokens`` residual would drift, even negative, whenever a
    # row's meter and column disagree).
    billed_output_tokens: int = 0


class UsageGroupRow(BaseModel):
    """One breakdown row (a model, a user, an API key, a session, ...).

    ``key`` is None both for the synthesized fold row (``is_other=True``) and for a
    real group whose column was NULL (e.g. usage from a since-deleted user, with
    ``is_other=False``). ``is_other`` disambiguates the two so the UI does not
    mislabel deleted-user usage as the fold.
    """

    key: str | None
    # Display name for an opaque key (a user's alias, an API key's name), resolved
    # in the same GROUP BY. Only ever set for the ``user`` and ``api_key``
    # dimensions, and null there too when the entity has no label or is gone; a
    # client falls back to ``key``. Its purpose is to let a client build a user or
    # key filter from this breakdown alone, rather than reading both whole tables.
    label: str | None = None
    cost: float
    tokens: int
    requests: int
    is_other: bool = False


class UsageErrorCodeRow(BaseModel):
    """One error-taxonomy row: the failures in the window sharing a status code.

    ``status_code`` is None for failures recorded without one (rows written
    before the column existed, and failures no HTTP status describes, e.g. a
    stream that finished without usage data under the ``fail`` policy).
    ``error_class`` is the coarse display bucket derived from the code, so a UI
    can group "provider fault" against "my own misconfiguration" without
    re-deriving a status ladder; the raw code stays alongside it for precision.
    """

    status_code: int | None
    error_class: ErrorClass
    requests: int


class UsageSeriesPoint(BaseModel):
    """One time bucket. ``bucket_start`` is canonical ISO-8601 UTC (``...Z``),
    identical across SQLite and PostgreSQL for the same underlying instant.

    ``tokens`` stays the raw provider-reported total (the field predates the
    composition split and external consumers may read it). The billed
    composition fields are normalized via billing meters (see ``_billed_expr``):
    ``input_tokens`` includes both cache buckets, so a chart derives fresh input
    as ``max(0, input_tokens - cache_read_tokens - cache_write_tokens)`` and the
    billed total as ``fresh + cache_read + cache_write + output``.
    """

    bucket_start: str
    cost: float
    tokens: int
    requests: int
    errors: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0


class UsageToolRow(BaseModel):
    """Spend and volume for one gateway-run tool inside the window.

    ``calls`` counts billable calls, not requests: one request can run a tool
    several times, which is the whole reason a per-tool view exists. ``errors``
    counts calls that failed and were therefore never billed. ``requests`` is how
    many requests ran the tool at least once, so the number reconciles with the
    Activity list when the same filter is applied there.
    """

    tool: str
    calls: int
    errors: int
    requests: int
    cost: float


class UsageSummary(BaseModel):
    """Aggregate spend/volume for the Usage & analytics page.

    Every breakdown field is always present. One the caller excluded through
    ``dimensions`` comes back as an empty list, the same shape a window with no
    matching rows produces, so narrowing the selector never changes the schema.
    """

    start_date: str
    end_date: str
    bucket: Bucket
    totals: UsageTotals
    by_model: list[UsageGroupRow]
    by_user: list[UsageGroupRow]
    by_api_key: list[UsageGroupRow]
    by_source: list[UsageGroupRow]
    # Session/project attribution for agent traffic: a handful of long-running
    # sessions routinely account for most of a workload's tokens, so this is the
    # dimension that turns "spend went up" into "this task went wrong". Gateway
    # rows carry no label, so they group under a single null key.
    by_source_label: list[UsageGroupRow]
    # API surface (/api/v1/chat/completions vs /api/v1/messages vs /api/v1/responses) and
    # upstream provider: the two splits a gateway operator needs and that no
    # other endpoint reports.
    by_endpoint: list[UsageGroupRow]
    by_provider: list[UsageGroupRow]
    # Gateway-run tool spend. Empty when the window has none, and MCP tools are
    # excluded by design (their names are unbounded, see GATEWAY_TOOL_NAMES).
    by_tool: list[UsageToolRow] = []
    # Spend by the values of the tag ``group_by_tag`` named; empty when it named none.
    by_tag: list[UsageGroupRow] = []
    # Failures only, so the taxonomy is not swamped by the successes that carry
    # no status code. Counts sum to ``totals.error_count``, unless a window
    # somehow held more than ``_BREAKDOWN_TOP_N`` distinct codes, in which case
    # the tail is omitted rather than folded (there is no synthesized "other"
    # row: a null key would collide with the real "no code recorded" group).
    errors_by_status_code: list[UsageErrorCodeRow]
    series: list[UsageSeriesPoint]


class UsageGroupedSeriesPoint(BaseModel):
    """One (time bucket, group) cell of a grouped series.

    ``key``/``is_other`` follow the ``UsageGroupRow`` convention: ``key=None``
    with ``is_other=True`` is the fold of groups outside the top N, ``key=None``
    with ``is_other=False`` is a real NULL group (e.g. a deleted user).
    ``tokens`` is the *billed* total (input including cache, plus output), the
    same quantity the ungrouped series' composition fields sum to.
    """

    bucket_start: str
    key: str | None
    is_other: bool = False
    cost: float
    tokens: int
    requests: int


class UsageGroupedSeries(BaseModel):
    """A per-group time series for the dashboard's stacked charts.

    ``groups`` ranks the window's top groups by spend (plus the reconciling
    ``other`` fold), in the order a chart should stack and color them; ``points``
    is sparse (only populated cells), keyed by canonical UTC ``bucket_start``.
    """

    start_date: str
    end_date: str
    bucket: Bucket
    group_by: SeriesGroupBy
    groups: list[UsageGroupRow]
    points: list[UsageGroupedSeriesPoint]
