"""Resolve natural-language time expressions for the ``resolve_time`` tool.

A backend the tool-use loop in :mod:`gateway.services.mcp_loop` dispatches to
whenever the model emits a ``resolve_time(expression=…)`` call. Unlike the
sandbox / web_search backends, this one is pure local computation — it shells
out to no service and holds no HTTP client — so its async-context-manager
methods are no-ops; it implements them only so the loop can drive it as a
``pool`` uniformly.

The heavy lifting is the `dateparser` library, with the parsing policy
(timezone handling, preferred direction, date order, week start, languages,
and an opaque settings passthrough) supplied per request by the platform's
``/gateway/resolve-time/resolve`` response. This mirrors how web_search reads
its workspace policy: the gateway runs the tool, the platform owns the config.

``timezone_mode`` reconciles the workspace ``timezone`` with the ``timezone_name``
the model may pass per call:

* ``request`` — honour the model's per-call timezone; fall back to UTC.
* ``default`` — honour the model's per-call timezone, else the workspace
  ``timezone``, else UTC.
* ``forced`` — ignore the model entirely; always use the workspace ``timezone``.

This satisfies the same duck-typed protocol the MCP loop uses for tool dispatch
(``openai_tools``, ``owns_tool``, ``purpose_hints``, ``call_tool``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

import dateparser

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

RESOLVE_TIME_TOOL_NAME = "resolve_time"

_VALID_TIMEZONE_MODES = frozenset({"request", "default", "forced"})
_VALID_PREFER_DATES_FROM = frozenset({"past", "future", "current_period"})
_VALID_DATE_ORDERS = frozenset({"MDY", "DMY", "YMD"})
_VALID_WEEK_STARTS = frozenset({"monday", "sunday"})

_DEFAULT_TIMEZONE_MODE = "default"
_DEFAULT_PREFER_DATES_FROM = "past"
_DEFAULT_DATE_ORDER = "MDY"
_DEFAULT_WEEK_START = "monday"

# Gateway-controlled dateparser settings that an opaque parser_options bag must
# never override — these are derived from the typed policy fields instead.
_RESERVED_PARSER_SETTINGS = frozenset({"RELATIVE_BASE", "RETURN_AS_TIMEZONE_AWARE", "PREFER_DATES_FROM", "DATE_ORDER"})

# Defensive bounds mirroring the platform's stored-config caps. In standalone
# mode the config comes straight from the request with no platform validation,
# so the backend clamps it itself: an oversized expression would block the parse
# thread, and an unbounded parser_options/languages payload is pointless input.
_MAX_EXPRESSION_LEN = 512
_MAX_LANGUAGES = 20
_MAX_PARSER_OPTION_KEYS = 30
_MAX_PARSER_OPTIONS_BYTES = 4096

_DEFAULT_PURPOSE_HINT = (
    "Call `resolve_time` to turn any relative or natural-language date "
    '("last 2 weeks", "yesterday", "since Monday") into exact ISO 8601 '
    "timestamps before passing them to other tools. Do not compute dates "
    "yourself. With no argument it returns the current time."
)

_TOOL_DESCRIPTION = (
    "Resolve a natural-language time expression to deterministic ISO 8601 "
    "timestamps. Call with no argument to get the current time. Single points "
    '("9 days ago", "March 1") return one timestamp; periods and ranges '
    '("yesterday", "last week", "9 days ago to today") return a JSON '
    "object with `start` and `end`."
)


class ResolveTimeBackend:
    """Stateless, local ``resolve_time`` tool backend.

    Duck-types as the MCP loop's ``pool`` parameter. Construction takes the
    resolved workspace policy; defaults reproduce octonous's behaviour
    (UTC-fallback, prefer past dates, MDY, Monday week start).
    """

    def __init__(
        self,
        *,
        timezone_mode: str | None = None,
        timezone_name: str | None = None,
        prefer_dates_from: str | None = None,
        date_order: str | None = None,
        week_start: str | None = None,
        languages: list[str] | None = None,
        parser_options: dict[str, Any] | None = None,
        purpose_hint: str | None = None,
    ) -> None:
        self._timezone_mode = timezone_mode if timezone_mode in _VALID_TIMEZONE_MODES else _DEFAULT_TIMEZONE_MODE
        self._workspace_tz = _safe_zoneinfo(timezone_name)
        self._prefer_dates_from = (
            prefer_dates_from if prefer_dates_from in _VALID_PREFER_DATES_FROM else _DEFAULT_PREFER_DATES_FROM
        )
        self._date_order = date_order if date_order in _VALID_DATE_ORDERS else _DEFAULT_DATE_ORDER
        self._week_start = week_start if week_start in _VALID_WEEK_STARTS else _DEFAULT_WEEK_START
        self._languages = [str(lang) for lang in languages[:_MAX_LANGUAGES]] if languages else None
        self._parser_options = _bounded_parser_options(parser_options)
        self._purpose_hint = purpose_hint or _DEFAULT_PURPOSE_HINT

    async def __aenter__(self) -> ResolveTimeBackend:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        return None

    # ----- duck-typed protocol the MCP loop uses on `pool` -----

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": RESOLVE_TIME_TOOL_NAME,
                    "description": _TOOL_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": (
                                    'A natural-language time expression, e.g. "last 2 weeks", '
                                    '"yesterday", "9 days ago", "January 1 to January 15". '
                                    "Omit to get the current time."
                                ),
                            },
                            "timezone_name": {
                                "type": "string",
                                "description": (
                                    'Optional IANA timezone (e.g. "America/New_York") to resolve '
                                    "the expression in. May be ignored by workspace policy."
                                ),
                            },
                        },
                        "required": [],
                    },
                },
            }
        ]

    def owns_tool(self, name: str) -> bool:
        return name == RESOLVE_TIME_TOOL_NAME

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [(RESOLVE_TIME_TOOL_NAME, self._purpose_hint)]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name != RESOLVE_TIME_TOOL_NAME:
            raise KeyError(f"ResolveTimeBackend does not own tool {name!r}")

        expression = arguments.get("expression")
        requested_tz = arguments.get("timezone_name")
        tz = self._effective_tz(requested_tz if isinstance(requested_tz, str) else None)

        if not isinstance(expression, str) or not expression.strip():
            return _fmt(datetime.now(tz), tz)

        cleaned = expression.strip()
        if len(cleaned) > _MAX_EXPRESSION_LEN:
            return f"Time expression too long (max {_MAX_EXPRESSION_LEN} characters)."

        # dateparser.parse is synchronous and CPU-bound; run it off the event
        # loop so a slow parse can't stall other requests sharing this worker.
        return await asyncio.to_thread(self._resolve, cleaned, tz)

    # ----- internals -----

    def _effective_tz(self, requested_tz: str | None) -> timezone | ZoneInfo:
        """Pick the timezone to resolve in, per ``timezone_mode``."""
        if self._timezone_mode == "forced":
            return self._workspace_tz or timezone.utc
        if self._timezone_mode == "request":
            return _safe_zoneinfo(requested_tz) or timezone.utc
        # default: model's per-call tz, then workspace tz, then UTC.
        return _safe_zoneinfo(requested_tz) or self._workspace_tz or timezone.utc

    def _resolve(self, expression: str, tz: timezone | ZoneInfo) -> str:
        now = datetime.now(tz)

        # Explicit range: "X to Y" -> {start, end}
        if " to " in expression:
            start_raw, end_raw = expression.split(" to ", maxsplit=1)
            start_dt = self._parse_expression(start_raw.strip(), now)
            end_dt = self._parse_expression(end_raw.strip(), now)
            if start_dt is not None and end_dt is not None:
                start = _floor(start_dt, tz)
                end = _floor(end_dt, tz) + timedelta(days=1)
                return _range_json(start, end, tz)

        # Named periods: "yesterday", "last week", etc. -> {start, end}
        window = self._resolve_common_period_window(expression, now, tz)
        if window is not None:
            return _range_json(window[0], window[1], tz)

        # Single point in time: "9 days ago", "March 1" -> bare ISO timestamp
        parsed = self._parse_expression(expression, now)
        if parsed is None:
            return f"Could not parse time expression: {expression!r}"
        return _fmt(_floor(parsed, tz), tz)

    def _parse_expression(self, expr: str, reference_date: datetime) -> datetime | None:
        settings: dict[str, Any] = {
            key: value for key, value in self._parser_options.items() if key not in _RESERVED_PARSER_SETTINGS
        }
        settings.update(
            {
                "RELATIVE_BASE": reference_date.replace(tzinfo=None),
                "RETURN_AS_TIMEZONE_AWARE": False,
                "PREFER_DATES_FROM": self._prefer_dates_from,
                "DATE_ORDER": self._date_order,
            }
        )
        try:
            result: datetime | None = dateparser.parse(expr, languages=self._languages, settings=cast(Any, settings))
        except Exception as exc:  # noqa: BLE001 — dateparser raises broad on bad settings
            logger.warning("resolve_time: dateparser failed for %r: %s", expr, exc)
            return None
        if result is None:
            return None
        return result.replace(tzinfo=reference_date.tzinfo or timezone.utc)

    def _resolve_common_period_window(
        self, expression: str, now: datetime, tz: timezone | ZoneInfo
    ) -> tuple[datetime, datetime] | None:
        """Return [start, end) windows for common relative phrases."""
        expr = expression.strip().lower()
        today = _floor(now, tz)
        week_offset = self._weekday_offset(today)

        if expr == "today":
            return today, today + timedelta(days=1)
        if expr == "yesterday":
            start = today - timedelta(days=1)
            return start, today
        if expr == "this week":
            start = today - timedelta(days=week_offset)
            return start, start + timedelta(days=7)
        if expr == "last week":
            this_week_start = today - timedelta(days=week_offset)
            start = this_week_start - timedelta(days=7)
            return start, this_week_start
        if expr == "this month":
            start = today.replace(day=1)
            return start, _first_day_next_month(start)
        if expr == "last month":
            this_month_start = today.replace(day=1)
            return _first_day_previous_month(this_month_start), this_month_start
        return None

    def _weekday_offset(self, day: datetime) -> int:
        """Days since the start of the week, honouring ``week_start``.

        Python's ``weekday()`` is Monday=0…Sunday=6. For a Sunday-start week,
        Sunday must map to 0 and Monday…Saturday to 1…6.
        """
        if self._week_start == "sunday":
            return (day.weekday() + 1) % 7
        return day.weekday()


def _safe_zoneinfo(name: str | None) -> ZoneInfo | None:
    if not name or not name.strip():
        return None
    try:
        return ZoneInfo(name.strip())
    except (KeyError, ValueError):
        logger.warning("resolve_time: ignoring invalid timezone %r", name)
        return None


def _bounded_parser_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Clamp a request-supplied parser_options bag to the backend's caps.

    Returns at most ``_MAX_PARSER_OPTION_KEYS`` keys, and drops the bag entirely
    if it can't be serialised or exceeds ``_MAX_PARSER_OPTIONS_BYTES`` — the
    platform applies the same caps when storing config, so this only bites in
    standalone mode where config comes straight from the request.
    """
    if not options:
        return {}
    bounded = dict(list(options.items())[:_MAX_PARSER_OPTION_KEYS])
    try:
        if len(json.dumps(bounded, default=str)) > _MAX_PARSER_OPTIONS_BYTES:
            logger.warning("resolve_time: parser_options exceeds size cap; ignoring")
            return {}
    except (TypeError, ValueError):
        logger.warning("resolve_time: parser_options not serialisable; ignoring")
        return {}
    return bounded


def _floor(dt: datetime, tz: timezone | ZoneInfo) -> datetime:
    """Floor a datetime to midnight (start of day) in the given timezone."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=tz)


def _is_utc(tz: timezone | ZoneInfo) -> bool:
    """True for UTC, however it was constructed (timezone.utc or ZoneInfo('UTC'))."""
    return tz == timezone.utc or getattr(tz, "key", "") in ("UTC", "Etc/UTC")


def _fmt(dt: datetime, tz: timezone | ZoneInfo) -> str:
    """ISO 8601 string — ``Z`` suffix for UTC, numeric offset otherwise."""
    if _is_utc(tz):
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.isoformat()


def _range_json(start: datetime, end: datetime, tz: timezone | ZoneInfo) -> str:
    return json.dumps({"start": _fmt(start, tz), "end": _fmt(end, tz)})


def _first_day_next_month(dt: datetime) -> datetime:
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1, day=1)
    return dt.replace(month=dt.month + 1, day=1)


def _first_day_previous_month(dt: datetime) -> datetime:
    if dt.month == 1:
        return dt.replace(year=dt.year - 1, month=12, day=1)
    return dt.replace(month=dt.month - 1, day=1)
