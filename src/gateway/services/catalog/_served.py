"""What the engine serving a model says about it, kept per provider in the database and in each worker.

vLLM and SGLang list each model with ``max_model_len``, the context length the
server runs it at, and ``root``, the model it loaded under whatever name it is
served by. SGLang also answers ``/model_info`` with the tool-call and reasoning
parsers it was started with and what it reads besides text. Discovery reports
each listing here; a listing is the same whichever way the provider was
connected, so a configured instance and a hosted provider are covered alike,
and an organization's own key reports its listings under its organization.

A write reaches its own worker at once and every other worker within
``SERVED_MODEL_REFRESH_SECONDS``.
"""

import asyncio
import dataclasses
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from any_llm.types.model import Model

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.models.catalog import SERVED_ENGINES, ServedModel
from gateway.repositories.catalog import ServedModelRepository

SERVED_MODEL_REFRESH_SECONDS = 60.0

# SGLang's own description of the server, newest path first.
_MODEL_INFO_PATHS = ("/model_info", "/get_model_info")
_MODEL_INFO_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class EngineSettings:
    """How an SGLang server was started, from its ``/model_info``."""

    reasoning_parser: str | None
    tool_call_parser: str | None
    input_modalities: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class ServedFacts:
    """One model as its engine described it. ``settings_reported`` makes a ``None`` parser mean off."""

    engine: str
    context_window: int | None = None
    root: str | None = None
    settings_reported: bool = False
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    input_modalities: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class KeyOwner:
    """The organization key a listing was made on, for a provider an organization brought."""

    organization_id: uuid.UUID
    key_id: uuid.UUID


# (organization, provider) -> model id -> facts; the organization is ``None`` for the deployment's own providers.
_served: dict[tuple[uuid.UUID | None, str], dict[str, ServedFacts]] = {}


def served_facts(provider: str, model_id: str, organization_id: uuid.UUID | None = None) -> ServedFacts | None:
    """What the engine behind ``provider`` last said about ``model_id``, or ``None``.

    ``organization_id`` reads that organization's own keys instead of the deployment's providers.
    """
    return _served.get((organization_id, provider), {}).get(model_id)


def _engine(model: Model) -> str | None:
    """The engine that listed ``model``, known by the ``owned_by`` each one stamps and the fields only they add."""
    extra = model.model_extra or {}
    if model.owned_by in SERVED_ENGINES and ("max_model_len" in extra or "root" in extra):
        return model.owned_by
    return None


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def facts_from_listing(model: Model, settings: EngineSettings | None) -> ServedFacts | None:
    """What a vLLM or SGLang listing entry says about its model, or ``None`` for any other listing."""
    engine = _engine(model)
    if engine is None:
        return None
    extra = model.model_extra or {}
    facts = ServedFacts(
        engine=engine, context_window=_positive_int(extra.get("max_model_len")), root=_text(extra.get("root"))
    )
    if engine != "sglang" or settings is None:
        return facts
    return dataclasses.replace(
        facts,
        settings_reported=True,
        reasoning_parser=settings.reasoning_parser,
        tool_call_parser=settings.tool_call_parser,
        input_modalities=settings.input_modalities,
    )


def _settings_from(info: Mapping[str, Any]) -> EngineSettings:
    modalities: tuple[str, ...] | None = None
    if "has_image_understanding" in info or "has_audio_understanding" in info:
        modalities = ("text",)
        if info.get("has_image_understanding") is True:
            modalities += ("image",)
        if info.get("has_audio_understanding") is True:
            modalities += ("audio",)
    return EngineSettings(
        reasoning_parser=_text(info.get("reasoning_parser")),
        tool_call_parser=_text(info.get("tool_call_parser")),
        input_modalities=modalities,
    )


def _server_root(api_base: str) -> str:
    """SGLang answers ``/model_info`` beside ``/v1``, not under it."""
    root = api_base.rstrip("/")
    return root.removesuffix("/v1")


async def read_engine_settings(
    models: Sequence[Model], *, api_base: str | None, api_key: str | None
) -> EngineSettings | None:
    """Ask the SGLang server behind ``models`` how it was started, or ``None`` when it is not one or will not say.

    Called only after the listing from the same ``api_base`` succeeded, so the
    address has passed the same checks. Redirects are not followed.
    """
    if not api_base or not any(_engine(model) == "sglang" for model in models):
        return None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    root = _server_root(api_base)
    try:
        async with httpx.AsyncClient(timeout=_MODEL_INFO_TIMEOUT_SECONDS, follow_redirects=False) as client:
            for path in _MODEL_INFO_PATHS:
                response = await client.get(root + path, headers=headers)
                if response.status_code == httpx.codes.NOT_FOUND:
                    continue
                response.raise_for_status()
                info = response.json()
                return _settings_from(info) if isinstance(info, dict) else None
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("Could not read the SGLang server's model info (%s)", type(exc).__name__)
    return None


def _columns(facts: ServedFacts) -> dict[str, Any]:
    columns = dataclasses.asdict(facts)
    if facts.input_modalities is not None:
        columns["input_modalities"] = list(facts.input_modalities)
    return columns


def _facts(row: ServedModel) -> ServedFacts:
    return ServedFacts(
        engine=row.engine,
        context_window=row.context_window,
        root=row.root,
        settings_reported=row.settings_reported,
        reasoning_parser=row.reasoning_parser,
        tool_call_parser=row.tool_call_parser,
        input_modalities=tuple(row.input_modalities) if row.input_modalities is not None else None,
    )


async def record_served_models(
    provider: str,
    models: Sequence[Model],
    settings: EngineSettings | None,
    *,
    owner: KeyOwner | None = None,
    uow: UnitOfWork | None = None,
) -> None:
    """Keep what a successful listing of ``provider`` says about each model, replacing what it said before.

    ``owner`` is the organization key the listing was made on, and ``provider``
    then the prefix its organization reaches it by; without one, ``provider``
    is the deployment's. ``uow`` is the caller's own, for a request; a worker
    passes none and the write gets one of its own.

    Writes only when the answer changed, so a provider that is no engine, or
    one whose answer holds, costs nothing per dial. A listing with no engine
    fields at all keeps what was stored, because it is the instance's declared
    ``models:`` standing in for a listing that failed. A failed write is
    logged: the worker keeps the answer, and the next listing that differs from
    the stored one tries again.
    """
    facts = {model.id: found for model in models if (found := facts_from_listing(model, settings)) is not None}
    held = (owner.organization_id if owner else None, provider)
    if _served.get(held, {}) == facts or (not facts and not any(model.model_extra for model in models)):
        return
    if facts:
        _served[held] = facts
    else:
        _served.pop(held, None)
    columns = {model_id: _columns(found) for model_id, found in facts.items()}
    try:
        if uow is not None:
            await _replace(uow, provider, columns, owner)
        else:
            async with create_unit_of_work() as own:
                await _replace(own, provider, columns, owner)
    except DATABASE_ERRORS as exc:
        logger.warning("Could not store what provider '%s' serves: %s", provider, describe_database_error(exc))
    except RuntimeError:
        # No database was initialized, as in a CLI run with no server.
        logger.debug("No database to store what provider '%s' serves", provider)


async def _replace(uow: UnitOfWork, provider: str, columns: dict[str, dict[str, Any]], owner: KeyOwner | None) -> None:
    async with uow:
        await ServedModelRepository(uow).replace(
            provider,
            columns,
            organization_id=owner.organization_id if owner else None,
            org_provider_key_id=owner.key_id if owner else None,
        )


async def refresh_served_models(uow: UnitOfWork) -> None:
    """Reload every row in one query and swap the copy in one step."""
    global _served  # noqa: PLW0603

    async with uow:
        rows = await ServedModelRepository(uow).list_all()
    served: dict[tuple[uuid.UUID | None, str], dict[str, ServedFacts]] = {}
    for row in rows:
        served.setdefault((row.organization_id, row.provider), {})[row.model] = _facts(row)
    _served = served


def reset_served_models() -> None:
    """Drop the copy so the next load starts clean (startup and tests)."""
    global _served  # noqa: PLW0603

    _served = {}


async def _refresh_on_a_session_of_its_own() -> None:
    async with create_unit_of_work() as uow:
        await refresh_served_models(uow)


async def load_served_models_at_startup() -> None:
    """Prime the copy before the first listing. A failure is logged, not raised."""
    reset_served_models()
    try:
        await _refresh_on_a_session_of_its_own()
    except Exception:
        logger.exception("Failed to load what model servers serve; continuing with none")


async def run_served_model_refresher(interval: float = SERVED_MODEL_REFRESH_SECONDS) -> None:
    """Reload the copy forever, so another worker's listings arrive. Every error is retried next tick."""
    while True:
        await asyncio.sleep(interval)
        try:
            await _refresh_on_a_session_of_its_own()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Served model refresh failed; retrying in %ss", interval, exc_info=True)
