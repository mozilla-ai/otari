"""Runtime-toggleable settings, persisted so a dashboard change survives restart.

A tiny layer over the ``runtime_settings`` table for the config fields the
dashboard can change at runtime. A stored override wins over the
config-file/env value; it is loaded and applied on startup and updated in place
when the dashboard changes it.

Only fields that are genuinely safe to hot-change belong here: each key in
``_SPECS`` is read fresh from the in-memory config on the request path (or on
the next model discovery / catalog lookup), so mutating ``config`` takes effect
immediately in this worker. Startup-only fields (``host``, ``port``,
``database_url``, ``mode``, the ``db_pool_*`` sizing, ``rate_limit_rpm``,
``cors_allow_origins``, ``budget_strategy``, docs/metrics toggles, the log
writer) are deliberately excluded: they are consumed once at startup, so
mutating ``config`` alone would not change running behavior. The outbound
network-safety gates (``mcp_allow_*``, ``web_search_allow_private_hosts``, the
``*_url`` fields) are also excluded on purpose: opening an SSRF gate is a
security decision that should not sit behind a dashboard toggle. Those stay
display-only in the dashboard.

Note on ``require_pricing`` / ``reject_user_mismatch``: these are metering and
tenant-isolation policy controls, yet they *are* settable, unlike the SSRF
gates above. The distinction is deliberate. Both keep fail-closed defaults and
are master-key gated, and toggling them changes only how the gateway meters or
attributes a request; it does not widen the trust boundary for untrusted input
or open outbound network access the way flipping an SSRF gate would. They are
operator policy knobs, so a dashboard toggle is appropriate; an SSRF gate is a
security posture change, so it is not.

Scope note: applying an override mutates this worker's in-memory config and the
process-wide pricing flag, so it takes effect immediately here. Other workers or
replicas pick up the change on their next startup (the stored value is the source
of truth), not live; that is acceptable for these operator settings.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import STREAM_MISSING_USAGE_POLICIES, VISION_STRATEGIES, GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import RuntimeSetting
from gateway.services.pricing_service import configure_default_pricing

MODEL_DISCOVERY = "model_discovery"
DEFAULT_PRICING = "default_pricing"
REQUIRE_PRICING = "require_pricing"
REJECT_USER_MISMATCH = "reject_user_mismatch"
MODEL_CACHE_TTL_SECONDS = "model_cache_ttl_seconds"
STREAM_MISSING_USAGE_POLICY = "stream_missing_usage_policy"
MODEL_DISCOVERY_TIMEOUT_SECONDS = "model_discovery_timeout_seconds"
MODEL_DISCOVERY_NEGATIVE_TTL_SECONDS = "model_discovery_negative_ttl_seconds"
MODELS_DEV_METADATA = "models_dev_metadata"
MODELS_DEV_CACHE_TTL_SECONDS = "models_dev_cache_ttl_seconds"
FILE_UNDERSTANDING_ENABLED = "file_understanding_enabled"
VISION_STRATEGY = "vision_strategy"
VISION_DESCRIBE_MODEL = "vision_describe_model"
VISION_DESCRIBE_MAX_TOKENS = "vision_describe_max_tokens"
BUDGET_ESTIMATE_DEFAULT_OUTPUT_TOKENS = "budget_estimate_default_output_tokens"

# A settable value is a plain scalar. Stored as a string in ``runtime_settings``
# (the table is intentionally schema-light) and parsed back per the key's spec.
# ``None`` is only valid for a nullable string field and is stored as "".
SettingValue = bool | int | float | str | None


@dataclass(frozen=True)
class _Spec:
    """How one settable key is typed, bounded, and validated.

    Mirrors the corresponding ``GatewayConfig`` field so a dashboard write cannot
    push the running gateway into a state a fresh config load would reject.
    """

    type: str  # "bool" | "int" | "float" | "str"
    ge: float | None = None  # inclusive lower bound (matches Field(ge=...))
    gt: float | None = None  # exclusive lower bound (matches Field(gt=...))
    options: tuple[str, ...] | None = None  # allowed values for an enum string
    nullable: bool = False  # a string field that may be unset (stored as "")


# The config fields the dashboard may override, each with the constraints its
# GatewayConfig field declares. ``SETTABLE_KEYS`` is derived from this so the two
# never drift.
_SPECS: dict[str, _Spec] = {
    MODEL_DISCOVERY: _Spec("bool"),
    DEFAULT_PRICING: _Spec("bool"),
    REQUIRE_PRICING: _Spec("bool"),
    REJECT_USER_MISMATCH: _Spec("bool"),
    MODELS_DEV_METADATA: _Spec("bool"),
    FILE_UNDERSTANDING_ENABLED: _Spec("bool"),
    MODEL_CACHE_TTL_SECONDS: _Spec("int", ge=0),
    MODELS_DEV_CACHE_TTL_SECONDS: _Spec("int", ge=0),
    VISION_DESCRIBE_MAX_TOKENS: _Spec("int", gt=0),
    BUDGET_ESTIMATE_DEFAULT_OUTPUT_TOKENS: _Spec("int", ge=0),
    MODEL_DISCOVERY_TIMEOUT_SECONDS: _Spec("float", gt=0),
    MODEL_DISCOVERY_NEGATIVE_TTL_SECONDS: _Spec("float", ge=0),
    STREAM_MISSING_USAGE_POLICY: _Spec("str", options=STREAM_MISSING_USAGE_POLICIES),
    VISION_STRATEGY: _Spec("str", options=VISION_STRATEGIES),
    VISION_DESCRIBE_MODEL: _Spec("str", nullable=True),
}

# The config fields the dashboard may override; everything else is config/env only.
SETTABLE_KEYS = tuple(_SPECS)


def settable_options(key: str) -> tuple[str, ...] | None:
    """Return the allowed values for a settable enum field, or ``None``."""
    spec = _SPECS.get(key)
    return spec.options if spec is not None else None


def settable_bounds(key: str) -> tuple[float | None, float | None]:
    """Return ``(ge, gt)`` numeric bounds for a settable field (``None`` if unset).

    Lets the dashboard gate a numeric input against the same lower bound the
    field declares, so a ``gt=0`` field disables Save at ``0`` rather than
    round-tripping to a 422.
    """
    spec = _SPECS.get(key)
    if spec is None:
        return (None, None)
    return (spec.ge, spec.gt)


def _serialize(value: SettingValue) -> str:
    if value is None:
        return ""
    # bool is a subclass of int, so check it first.
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse(key: str, raw: str) -> SettingValue:
    spec = _SPECS[key]
    if spec.type == "bool":
        return raw.strip().lower() == "true"
    if spec.type == "int":
        return int(raw)
    if spec.type == "float":
        return float(raw)
    if spec.nullable and raw == "":
        return None
    return raw


def validate_value(key: str, value: SettingValue) -> SettingValue:
    """Return the normalized value for a settable key, or raise ``ValueError``."""
    spec = _SPECS.get(key)
    if spec is None:
        msg = f"Unknown runtime setting: {key!r}"
        raise ValueError(msg)

    if spec.type == "bool":
        if not isinstance(value, bool):
            msg = f"{key} must be a boolean."
            raise ValueError(msg)
        return value

    if spec.type in ("int", "float"):
        # bool is a subclass of int; reject it so a stray ``true`` is not stored as 1.
        if isinstance(value, bool):
            msg = f"{key} must be a number."
            raise ValueError(msg)
        if spec.type == "int":
            if not isinstance(value, int):
                msg = f"{key} must be an integer."
                raise ValueError(msg)
            number: float = value
        else:
            if not isinstance(value, (int, float)):
                msg = f"{key} must be a number."
                raise ValueError(msg)
            number = float(value)
        if spec.ge is not None and number < spec.ge:
            msg = f"{key} must be >= {spec.ge}."
            raise ValueError(msg)
        if spec.gt is not None and number <= spec.gt:
            msg = f"{key} must be > {spec.gt}."
            raise ValueError(msg)
        return number if spec.type == "float" else value

    # String field (possibly an enum, possibly nullable).
    if spec.nullable and (value is None or (isinstance(value, str) and value.strip() == "")):
        return None
    if not isinstance(value, str):
        msg = f"{key} must be a string."
        raise ValueError(msg)
    if spec.options is not None:
        normalized = value.strip().lower()
        if normalized not in spec.options:
            msg = f"{key} must be one of {sorted(spec.options)}."
            raise ValueError(msg)
        return normalized
    return value


async def load_overrides(session: AsyncSession) -> dict[str, SettingValue]:
    """Return every stored, still-valid override as a ``{key: value}`` map.

    A row whose key is no longer settable, or whose stored value no longer
    validates, is skipped (and logged) rather than crashing startup: the config
    value stands in its place.
    """
    rows = (await session.execute(select(RuntimeSetting))).scalars().all()
    overrides: dict[str, SettingValue] = {}
    for row in rows:
        if row.key not in _SPECS:
            continue
        try:
            overrides[row.key] = validate_value(row.key, _parse(row.key, row.value))
        except (ValueError, TypeError):
            logger.warning("Ignoring invalid stored runtime setting %s=%r", row.key, row.value)
    return overrides


def apply_override(config: GatewayConfig, key: str, value: SettingValue) -> None:
    """Apply one override to the running gateway (mutating config + globals)."""
    if key == DEFAULT_PRICING:
        config.default_pricing = bool(value)
        configure_default_pricing(bool(value))
        return
    setattr(config, key, value)


async def apply_overrides_from_db(config: GatewayConfig, session: AsyncSession) -> None:
    """Load persisted overrides and apply them to ``config`` at startup."""
    overrides = await load_overrides(session)
    for key, value in overrides.items():
        apply_override(config, key, value)
        logger.info("Applied runtime setting override %s=%s", key, value)


async def stage_override(session: AsyncSession, key: str, value: SettingValue) -> SettingValue:
    """Stage an override for persistence, without applying it yet.

    Persistence and application are deliberately separate: the caller commits
    first and only applies the override (mutating in-memory config and the
    process-wide pricing flag, which affects the billing path) once the write
    has succeeded. Applying before the commit would leave this worker metering
    or listing against a value that was never persisted if the commit failed.
    Caller is responsible for committing the session, then calling
    ``apply_override`` with the returned (normalized) value.

    Raises ``ValueError`` if the key is not settable or the value is invalid.
    """
    normalized = validate_value(key, value)
    row = await session.get(RuntimeSetting, key)
    if row is None:
        session.add(RuntimeSetting(key=key, value=_serialize(normalized)))
    else:
        row.value = _serialize(normalized)
    return normalized
