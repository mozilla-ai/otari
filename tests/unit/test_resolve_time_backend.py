"""Unit tests for the local ``resolve_time`` tool backend.

Time is frozen by monkeypatching the module's ``datetime`` with a subclass
whose ``now()`` returns a fixed instant (Sat 2026-02-14 15:30 UTC), so the
relative-expression assertions are deterministic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, tzinfo

import pytest

from gateway.services import resolve_time_backend as backend_module
from gateway.services.resolve_time_backend import RESOLVE_TIME_TOOL_NAME, ResolveTimeBackend

_FROZEN_NOW = datetime(2026, 2, 14, 15, 30, 0, tzinfo=timezone.utc)  # a Saturday


@pytest.fixture(autouse=True)
def _freeze_now(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:  # type: ignore[override]
            return _FROZEN_NOW.astimezone(tz) if tz is not None else _FROZEN_NOW.replace(tzinfo=None)

    monkeypatch.setattr(backend_module, "datetime", _FrozenDatetime)


async def _call(backend: ResolveTimeBackend, expression: str | None = None, **kwargs: str) -> str:
    args: dict[str, str] = dict(kwargs)
    if expression is not None:
        args["expression"] = expression
    async with backend:
        return await backend.call_tool(RESOLVE_TIME_TOOL_NAME, args)


# ───────────── basics ─────────────


@pytest.mark.asyncio
async def test_no_argument_returns_now_utc() -> None:
    assert await _call(ResolveTimeBackend()) == "2026-02-14T15:30:00Z"


@pytest.mark.asyncio
async def test_empty_expression_returns_now() -> None:
    assert await _call(ResolveTimeBackend(), "   ") == "2026-02-14T15:30:00Z"


@pytest.mark.asyncio
async def test_single_point_floors_to_midnight() -> None:
    assert await _call(ResolveTimeBackend(), "9 days ago") == "2026-02-05T00:00:00Z"


@pytest.mark.asyncio
async def test_yesterday_returns_one_day_window() -> None:
    out = json.loads(await _call(ResolveTimeBackend(), "yesterday"))
    assert out == {"start": "2026-02-13T00:00:00Z", "end": "2026-02-14T00:00:00Z"}


@pytest.mark.asyncio
async def test_explicit_range_inclusive_end() -> None:
    out = json.loads(await _call(ResolveTimeBackend(), "9 days ago to today"))
    # end is floored + 1 day so the window is inclusive of "today".
    assert out == {"start": "2026-02-05T00:00:00Z", "end": "2026-02-15T00:00:00Z"}


@pytest.mark.asyncio
async def test_unparseable_expression_returns_error_string() -> None:
    out = await _call(ResolveTimeBackend(), "zzzz not a date qwerty")
    assert out.startswith("Could not parse time expression")


# ───────────── week start ─────────────


@pytest.mark.asyncio
async def test_last_week_monday_start() -> None:
    out = json.loads(await _call(ResolveTimeBackend(week_start="monday"), "last week"))
    # Sat 2026-02-14 → this week starts Mon 2026-02-09 → last week is Feb 2..9.
    assert out == {"start": "2026-02-02T00:00:00Z", "end": "2026-02-09T00:00:00Z"}


@pytest.mark.asyncio
async def test_last_week_sunday_start() -> None:
    out = json.loads(await _call(ResolveTimeBackend(week_start="sunday"), "last week"))
    # Sunday-start: this week starts Sun 2026-02-08 → last week is Feb 1..8.
    assert out == {"start": "2026-02-01T00:00:00Z", "end": "2026-02-08T00:00:00Z"}


# ───────────── timezone modes ─────────────


@pytest.mark.asyncio
async def test_forced_mode_uses_workspace_tz_and_ignores_request() -> None:
    backend = ResolveTimeBackend(timezone_mode="forced", timezone_name="America/New_York")
    # Even though the model passes UTC, forced mode resolves in New York.
    out = json.loads(await _call(backend, "yesterday", timezone_name="UTC"))
    assert out == {"start": "2026-02-13T00:00:00-05:00", "end": "2026-02-14T00:00:00-05:00"}


@pytest.mark.asyncio
async def test_request_mode_honours_model_tz() -> None:
    backend = ResolveTimeBackend(timezone_mode="request")
    # No workspace tz configured; request mode resolves in the model-passed tz,
    # so the floored result carries New York's -05:00 offset.
    out = await _call(backend, "9 days ago", timezone_name="America/New_York")
    assert out == "2026-02-05T00:00:00-05:00"


@pytest.mark.asyncio
async def test_request_mode_without_tz_falls_back_to_utc() -> None:
    backend = ResolveTimeBackend(timezone_mode="request", timezone_name="America/New_York")
    # request mode ignores the workspace tz when the model passes none → UTC.
    assert await _call(backend, "9 days ago") == "2026-02-05T00:00:00Z"


@pytest.mark.asyncio
async def test_default_mode_prefers_request_then_workspace() -> None:
    backend = ResolveTimeBackend(timezone_mode="default", timezone_name="America/New_York")
    # No per-call tz → falls back to workspace New York.
    assert await _call(backend, "9 days ago") == "2026-02-05T00:00:00-05:00"


# ───────────── policy knobs ─────────────


@pytest.mark.asyncio
async def test_invalid_timezone_name_falls_back_to_utc() -> None:
    backend = ResolveTimeBackend(timezone_mode="forced", timezone_name="Mars/Olympus_Mons")
    assert await _call(backend, "9 days ago") == "2026-02-05T00:00:00Z"


@pytest.mark.asyncio
async def test_zoneinfo_utc_emits_z_suffix() -> None:
    # timezone="UTC" resolves to ZoneInfo("UTC") (not timezone.utc); the result
    # must still use the Z suffix, not +00:00, for output consistency.
    backend = ResolveTimeBackend(timezone_mode="forced", timezone_name="UTC")
    assert await _call(backend, "9 days ago") == "2026-02-05T00:00:00Z"
    out = json.loads(await _call(backend, "yesterday"))
    assert out == {"start": "2026-02-13T00:00:00Z", "end": "2026-02-14T00:00:00Z"}


@pytest.mark.asyncio
async def test_overlong_expression_is_rejected_without_parsing() -> None:
    out = await _call(ResolveTimeBackend(), "a" * 600)
    assert "too long" in out


@pytest.mark.asyncio
async def test_today_and_month_windows() -> None:
    backend = ResolveTimeBackend()
    assert json.loads(await _call(backend, "today")) == {
        "start": "2026-02-14T00:00:00Z",
        "end": "2026-02-15T00:00:00Z",
    }
    assert json.loads(await _call(backend, "this month")) == {
        "start": "2026-02-01T00:00:00Z",
        "end": "2026-03-01T00:00:00Z",
    }
    assert json.loads(await _call(backend, "last month")) == {
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-02-01T00:00:00Z",
    }


def test_parser_options_clamped_to_key_cap() -> None:
    backend = ResolveTimeBackend(parser_options={f"K{i}": i for i in range(50)})
    assert len(backend._parser_options) == 30


def test_oversized_parser_options_dropped() -> None:
    backend = ResolveTimeBackend(parser_options={"BIG": "x" * 5000})
    assert backend._parser_options == {}


def test_languages_clamped() -> None:
    backend = ResolveTimeBackend(languages=[f"l{i}" for i in range(40)])
    assert backend._languages is not None
    assert len(backend._languages) == 20


@pytest.mark.asyncio
async def test_reserved_parser_options_cannot_override_gateway_settings() -> None:
    # A caller trying to flip PREFER_DATES_FROM via parser_options must be ignored;
    # the typed prefer_dates_from policy wins. "9 days ago" is unambiguous, so we
    # just assert it still resolves deterministically rather than erroring.
    backend = ResolveTimeBackend(parser_options={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": "garbage"})
    assert await _call(backend, "9 days ago") == "2026-02-05T00:00:00Z"


# ───────────── protocol ─────────────


def test_openai_tools_schema_shape() -> None:
    tools = ResolveTimeBackend().openai_tools
    assert len(tools) == 1
    fn = tools[0]["function"]
    assert fn["name"] == RESOLVE_TIME_TOOL_NAME
    assert set(fn["parameters"]["properties"]) == {"expression", "timezone_name"}
    assert fn["parameters"]["required"] == []


def test_owns_only_its_own_tool() -> None:
    backend = ResolveTimeBackend()
    assert backend.owns_tool(RESOLVE_TIME_TOOL_NAME) is True
    assert backend.owns_tool("web_search") is False


def test_purpose_hint_override() -> None:
    backend = ResolveTimeBackend(purpose_hint="custom hint")
    assert backend.purpose_hints() == [(RESOLVE_TIME_TOOL_NAME, "custom hint")]


@pytest.mark.asyncio
async def test_call_tool_rejects_foreign_tool_name() -> None:
    backend = ResolveTimeBackend()
    async with backend:
        with pytest.raises(KeyError):
            await backend.call_tool("web_search", {})
