"""Runtime-editable built-in tool & guardrail settings for the admin dashboard.

The service-endpoint and web-search fields that :mod:`runtime_settings_service`
deliberately keeps display-only (the ``*_url`` fields and the web-search knobs)
are made editable here, on their own dashboard page, with structural URL
validation and per-service reachability tests. Values are stored in the shared
``runtime_settings`` table under their config-field names; a stored override wins
over the config-file/env value and is applied to the running worker immediately,
because the request-path read sites resolve ``config.<attr>`` (with an env
fallback) rather than reading the environment directly.

Every field is nullable: clearing one removes the override, and the read site
falls back to the configured env/YAML value (not to "nothing"). So a deployment
that set ``OTARI_WEB_SEARCH_URL`` reverts to that env URL when the override is
cleared; it does not disable web search.

Security posture: these settings are master-key-gated and standalone-only. The
operator is already fully trusted (they can point the running gateway at any URL
via config/env), so URL validation is deliberately **structural** (scheme must be
http/https, a host must be present) rather than an SSRF deny-list. A deny-private
gate here would reject the bundled docker-compose sidecars (``http://searxng:8080``,
``http://guardrails:8000``), which is the primary thing this page configures. The
SSRF gates (``mcp_allow_*``, ``web_search_allow_private_hosts``) stay display-only
and are not widened from the dashboard.
"""

from dataclasses import dataclass
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.env import otari_env
from gateway.log_config import logger
from gateway.models.entities import RuntimeSetting
from gateway.services.runtime_settings_service import SettingValue

WEB_SEARCH_URL = "web_search_url"
WEB_SEARCH_ENGINES = "web_search_engines"
WEB_SEARCH_MAX_RESULTS = "web_search_max_results"
WEB_SEARCH_EXTRACT = "web_search_extract"
WEB_SEARCH_PURPOSE_HINT = "web_search_purpose_hint"
SANDBOX_URL = "sandbox_url"
SANDBOX_PURPOSE_HINT = "sandbox_purpose_hint"
GUARDRAILS_URL = "guardrails_url"


@dataclass(frozen=True)
class _ToolSpec:
    """How one editable tool/guardrail field is typed and validated.

    ``type`` is one of ``"url" | "str" | "int" | "bool"``. Every field is
    nullable (an empty value clears the override); ``ge`` is an inclusive lower
    bound for ``int`` fields, mirroring the ``GatewayConfig`` field's constraint.
    """

    type: str
    ge: int | None = None


# The tool/guardrail config fields the dashboard may edit. These are the ``*_url``
# and web-search fields that runtime_settings_service excludes on purpose; keeping
# them in a separate registry guarantees they can never be reached through the
# generic ``/v1/settings`` PATCH.
_TOOL_SPECS: dict[str, _ToolSpec] = {
    WEB_SEARCH_URL: _ToolSpec("url"),
    SANDBOX_URL: _ToolSpec("url"),
    GUARDRAILS_URL: _ToolSpec("url"),
    WEB_SEARCH_ENGINES: _ToolSpec("str"),
    WEB_SEARCH_PURPOSE_HINT: _ToolSpec("str"),
    SANDBOX_PURPOSE_HINT: _ToolSpec("str"),
    WEB_SEARCH_MAX_RESULTS: _ToolSpec("int", ge=1),
    WEB_SEARCH_EXTRACT: _ToolSpec("bool"),
}

TOOL_SETTABLE_KEYS = tuple(_TOOL_SPECS)

# Which service each field belongs to, for grouping in the dashboard.
_FIELD_SERVICE: dict[str, str] = {
    WEB_SEARCH_URL: "web_search",
    WEB_SEARCH_ENGINES: "web_search",
    WEB_SEARCH_MAX_RESULTS: "web_search",
    WEB_SEARCH_EXTRACT: "web_search",
    WEB_SEARCH_PURPOSE_HINT: "web_search",
    SANDBOX_URL: "sandbox",
    SANDBOX_PURPOSE_HINT: "sandbox",
    GUARDRAILS_URL: "guardrails",
}

# The URL field each service is tested/probed against.
SERVICE_URL_FIELD: dict[str, str] = {
    "web_search": WEB_SEARCH_URL,
    "sandbox": SANDBOX_URL,
    "guardrails": GUARDRAILS_URL,
}


def _is_blank(value: SettingValue) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def validate_url(value: str) -> str:
    """Structural URL check: http/https scheme + a host. Raises ``ValueError``.

    This is a typo guard for a trusted operator, not an SSRF gate: it does not
    resolve DNS or reject private/loopback hosts (the bundled sidecars live
    there). It does reject non-web schemes like ``file://`` / ``gopher://``.
    """
    candidate = value.strip()
    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        msg = f"Not a valid URL: {value!r}"
        raise ValueError(msg) from exc
    if parts.scheme not in ("http", "https"):
        msg = f"URL must use http or https, got {parts.scheme or 'no'} scheme."
        raise ValueError(msg)
    if not parts.hostname:
        msg = "URL must include a host."
        raise ValueError(msg)
    return candidate


def validate_value(key: str, value: SettingValue) -> SettingValue:
    """Return the normalized value for an editable tool field, or raise ``ValueError``.

    Every field is nullable: a blank value (``None`` or empty/whitespace string)
    clears the override and returns ``None``.
    """
    spec = _TOOL_SPECS.get(key)
    if spec is None:
        msg = f"Unknown tool setting: {key!r}"
        raise ValueError(msg)

    # Every field is nullable: a blank value (None, or an empty/whitespace string)
    # clears the override regardless of the field's type.
    if _is_blank(value):
        return None

    if spec.type == "bool":
        if not isinstance(value, bool):
            msg = f"{key} must be a boolean."
            raise ValueError(msg)
        return value

    if spec.type == "int":
        # bool is an int subclass; reject it so a stray ``true`` is not stored as 1.
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"{key} must be an integer."
            raise ValueError(msg)
        if spec.ge is not None and value < spec.ge:
            msg = f"{key} must be >= {spec.ge}."
            raise ValueError(msg)
        return value

    # String-valued (plain str or url).
    if not isinstance(value, str):
        msg = f"{key} must be a string."
        raise ValueError(msg)
    if spec.type == "url":
        return validate_url(value)
    return value


def _serialize(value: SettingValue) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse(key: str, raw: str) -> SettingValue:
    spec = _TOOL_SPECS[key]
    if raw == "":
        return None
    if spec.type == "bool":
        return raw.strip().lower() == "true"
    if spec.type == "int":
        return int(raw)
    return raw


async def load_overrides(session: AsyncSession) -> dict[str, SettingValue]:
    """Return every stored, still-valid tool override as a ``{key: value}`` map.

    A row whose key is no longer editable, or whose stored value no longer
    validates, is skipped (and logged) rather than crashing startup.
    """
    # Filter to the tool keys in the query: the runtime_settings table also holds
    # unrelated settings, so there is no need to fetch and scan those in Python.
    rows = (
        await session.execute(select(RuntimeSetting).where(RuntimeSetting.key.in_(TOOL_SETTABLE_KEYS)))
    ).scalars().all()
    overrides: dict[str, SettingValue] = {}
    for row in rows:
        try:
            overrides[row.key] = validate_value(row.key, _parse(row.key, row.value))
        except (ValueError, TypeError):
            # Log the key only, never the value: a *_url field may embed credentials
            # (e.g. https://user:secret@host), and stored values must not reach logs.
            logger.warning("Ignoring invalid stored tool setting %s", row.key)
    return overrides


def apply_override(config: GatewayConfig, key: str, value: SettingValue) -> None:
    """Apply one tool override to the running gateway by mutating ``config``.

    The request-path read sites resolve ``config.<attr>`` (with an env fallback),
    so the change takes effect on the next request in this worker.
    """
    setattr(config, key, value)


async def apply_overrides_from_db(config: GatewayConfig, session: AsyncSession) -> None:
    """Load persisted tool overrides and apply them to ``config`` at startup."""
    overrides = await load_overrides(session)
    for key, value in overrides.items():
        apply_override(config, key, value)
        # Key only: a *_url override value may embed credentials; keep it out of logs.
        logger.info("Applied tool setting override %s", key)


async def stage_override(session: AsyncSession, key: str, value: SettingValue) -> SettingValue:
    """Validate + stage a tool override for persistence, without applying it yet.

    The caller commits first, then applies the returned (normalized) value, so a
    failed commit never leaves this worker running against an unpersisted value.
    Raises ``ValueError`` if the key is not editable or the value is invalid.
    """
    normalized = validate_value(key, value)
    row = await session.get(RuntimeSetting, key)
    if row is None:
        session.add(RuntimeSetting(key=key, value=_serialize(normalized)))
    else:
        row.value = _serialize(normalized)
    return normalized


def effective_value(config: GatewayConfig, key: str) -> SettingValue:
    """Resolve the value a request would actually use for one tool field.

    Mirrors the request-path read sites: ``config.<attr>`` (which already has the
    ``OTARI_<FIELD>`` env / YAML value layered in at load, and any applied
    override on top), with an ``otari_env`` fallback for robustness. Used by the
    GET endpoint so the dashboard shows / places-as-placeholder exactly what a
    request resolves to.
    """
    value: SettingValue = getattr(config, key, None)
    if value is not None:
        return value
    raw_env = otari_env(key.upper())
    if raw_env is None or raw_env == "":
        return None
    spec = _TOOL_SPECS[key]
    if spec.type == "int":
        try:
            return int(raw_env)
        except ValueError:
            return None
    if spec.type == "bool":
        return raw_env.strip().lower() not in {"0", "false", "no", "off"}
    return raw_env


def effective_values(config: GatewayConfig) -> dict[str, SettingValue]:
    """The effective value of every editable tool field, keyed by field name."""
    return {key: effective_value(config, key) for key in _TOOL_SPECS}


def field_service(key: str) -> str:
    """Which service ('web_search' | 'sandbox' | 'guardrails') a field belongs to."""
    return _FIELD_SERVICE[key]


def field_type(key: str) -> str:
    """The display/validation type of a field ('url' | 'str' | 'int' | 'bool')."""
    return _TOOL_SPECS[key].type
