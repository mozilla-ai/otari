"""Standalone-vs-platform mode seam for the LLM route handlers.

The gateway's operating mode is derived once from ``config.is_platform_mode``
(``core/config.py``). Historically each of the chat / messages / responses
handlers consumed that single boolean and then re-branched on it inline at many
sites, interleaving the two modes' credential-resolution and usage-reporting
logic inside one long function body. This module lifts those two concerns
behind a per-request strategy object so the handlers select the mode once and
never branch on it again for resolution or settlement.

Two seams live here:

* :class:`RequestModeStrategy` — credential resolution. ``resolve`` runs the
  platform resolve call (platform) or the auth + budget-reservation flow
  (standalone) and stashes the per-request state the handler needs.
* :class:`RequestSettlement` — usage reporting. The four async hooks match the
  callbacks ``gateway.streaming.streaming_generator`` already accepts, plus the
  standalone non-streaming success/error tail. Platform reports usage upstream;
  standalone writes a ``UsageLog`` row and reconciles/refunds the reservation.

Dispatch mechanics (the ``run_platform_attempts`` runner vs the standalone
single call, and the streaming fallback-vs-single split) deliberately stay in
the handlers — they are tool/stream plumbing, not mode-resolution.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import AnyLLMError
from any_llm.types.completion import CompletionUsage
from fastapi import HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes._platform import (
    ResolvedRoute,
    _extract_platform_user_token,
    _report_platform_usage,
    _resolve_platform_credentials,
    _resolve_platform_mcp_servers,
)
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.mcp import McpServerConfig
from gateway.rate_limit import RateLimitInfo, check_rate_limit
from gateway.services.budget_service import (
    ReservationHandle,
    estimate_cost,
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import find_model_pricing, pricing_required_but_missing

# ---------------------------------------------------------------------------
# Usage-reporting seam
# ---------------------------------------------------------------------------


class RequestSettlement(ABC):
    """Settles a request's usage once the provider call resolves.

    The four hooks mirror the callbacks ``streaming_generator`` accepts so a
    settlement can be wired straight into the SSE generator, and are reused on
    the standalone non-streaming success/error tail.
    """

    @abstractmethod
    async def on_success(self, usage: CompletionUsage | None) -> None:
        """A completed call (non-stream result, or stream that carried usage)."""

    @abstractmethod
    async def on_error(self, error: str) -> None:
        """A failed call that should be recorded as an error (and, standalone,
        release the reservation)."""

    @abstractmethod
    async def on_provider_error_precommit(self, error: str) -> None:
        """A provider error raised before any stream bytes committed.

        Standalone records the error without settling the reservation, matching
        the long-standing messages/responses pre-commit behavior (the
        non-streaming and chat paths refund here; messages/responses do not).
        Platform reports nothing — this path predates platform usage reporting
        and there is no local reservation to release.
        """

    @abstractmethod
    async def on_no_usage(self) -> None:
        """A stream that completed without any provider usage data."""

    @abstractmethod
    async def on_incomplete(self) -> None:
        """A stream the client abandoned before completion."""


class PlatformSettlement(RequestSettlement):
    """Reports usage back to the platform via ``_report_platform_usage``.

    Platform mode has no local budget reservation, so ``on_no_usage`` /
    ``on_incomplete`` are no-ops — there is nothing local to settle.
    """

    def __init__(self, *, config: GatewayConfig, correlation_id: str) -> None:
        self._config = config
        self._correlation_id = correlation_id

    async def on_success(self, usage: CompletionUsage | None) -> None:
        asyncio.create_task(
            _report_platform_usage(
                config=self._config,
                correlation_id=self._correlation_id,
                outcome="success",
                usage=usage,
            )
        )

    async def on_error(self, error: str) -> None:
        asyncio.create_task(
            _report_platform_usage(
                config=self._config,
                correlation_id=self._correlation_id,
                outcome="error",
                usage=None,
            )
        )

    async def on_provider_error_precommit(self, error: str) -> None:
        return

    async def on_no_usage(self) -> None:
        return

    async def on_incomplete(self) -> None:
        return


class StandaloneSettlement(RequestSettlement):
    """Writes a ``UsageLog`` row and reconciles/refunds the local reservation."""

    def __init__(
        self,
        *,
        db: AsyncSession | None,
        log_writer: LogWriter,
        api_key_id: str | None,
        user_id: str | None,
        provider: LLMProvider,
        model: str,
        endpoint: str,
        reservation: ReservationHandle | None,
        config: GatewayConfig,
    ) -> None:
        self._db = db
        self._log_writer = log_writer
        self._api_key_id = api_key_id
        self._user_id = user_id
        self._provider = provider
        self._model = model
        self._endpoint = endpoint
        self._reservation = reservation
        self._config = config

    async def on_success(self, usage: CompletionUsage | None) -> None:
        # Local import breaks the chat <-> _mode_strategy import cycle:
        # chat.py imports this module at top level, so this module cannot import
        # chat.log_usage at module scope (it is defined after chat's import of us).
        from gateway.api.routes.chat import log_usage

        if self._db is None:
            return
        actual_cost = await log_usage(
            db=self._db,
            log_writer=self._log_writer,
            api_key_id=self._api_key_id,
            model=self._model,
            provider=self._provider,
            endpoint=self._endpoint,
            user_id=self._user_id,
            usage_override=usage,
        )
        if self._reservation is not None:
            await reconcile_reservation(self._db, self._reservation, actual_cost or 0.0)

    async def on_error(self, error: str) -> None:
        from gateway.api.routes.chat import log_usage

        if self._db is None:
            return
        await log_usage(
            db=self._db,
            log_writer=self._log_writer,
            api_key_id=self._api_key_id,
            model=self._model,
            provider=self._provider,
            endpoint=self._endpoint,
            user_id=self._user_id,
            error=error,
        )
        if self._reservation is not None:
            await refund_reservation(self._db, self._reservation)

    async def on_provider_error_precommit(self, error: str) -> None:
        from gateway.api.routes.chat import log_usage

        # Log the error but leave the reservation untouched, matching the
        # existing messages/responses pre-commit streaming behavior.
        if self._db is None:
            return
        await log_usage(
            db=self._db,
            log_writer=self._log_writer,
            api_key_id=self._api_key_id,
            model=self._model,
            provider=self._provider,
            endpoint=self._endpoint,
            user_id=self._user_id,
            error=error,
        )

    async def on_no_usage(self) -> None:
        from gateway.api.routes.chat import log_usage

        # Stream completed but the provider sent no usage data. Settle the
        # reservation per stream_missing_usage_policy instead of billing $0.
        if self._db is None or self._log_writer is None or self._reservation is None:
            return
        policy = self._config.stream_missing_usage_policy
        if policy == "allow_free":
            await log_usage(
                db=self._db,
                log_writer=self._log_writer,
                api_key_id=self._api_key_id,
                model=self._model,
                provider=self._provider,
                endpoint=self._endpoint,
                user_id=self._user_id,
            )
            await refund_reservation(self._db, self._reservation)
            return
        # 'estimate' and 'fail' both charge the up-front estimate; 'fail' also
        # records the request as errored.
        await log_usage(
            db=self._db,
            log_writer=self._log_writer,
            api_key_id=self._api_key_id,
            model=self._model,
            provider=self._provider,
            endpoint=self._endpoint,
            user_id=self._user_id,
            error="stream completed without usage data" if policy == "fail" else None,
            cost_override=self._reservation.estimate,
        )
        await reconcile_reservation(self._db, self._reservation, self._reservation.estimate)

    async def on_incomplete(self) -> None:
        # Client disconnected mid-stream — release the reservation.
        if self._db is None or self._reservation is None:
            return
        await refund_reservation(self._db, self._reservation)


# ---------------------------------------------------------------------------
# Credential-resolution seam
# ---------------------------------------------------------------------------


@dataclass
class ResolveErrors:
    """Endpoint-specific HTTPExceptions the standalone resolve flow raises.

    Each handler builds these in its own error envelope (chat / responses use
    plain ``HTTPException`` detail strings; messages uses the Anthropic error
    envelope) so the shared resolve logic stays format-agnostic.
    """

    db_unavailable: HTTPException
    master_key_user_required: HTTPException
    api_key_validation_failed: HTTPException
    no_user: HTTPException
    forbidden_user: HTTPException
    no_pricing: HTTPException


@dataclass
class ResolveSpec:
    """Per-request inputs to :meth:`RequestModeStrategy.resolve`."""

    model_selector: str
    user_id_from_request: str | None
    prompt_chars: int
    max_output_tokens: int | None
    errors: ResolveErrors
    # Optional provider-support guard (responses' SUPPORTS_RESPONSES check).
    # Platform applies it to every resolved attempt; standalone applies it to
    # the split provider. Raises the endpoint's own HTTPException on rejection.
    validate_provider: Callable[[LLMProvider], None] | None = None


class RequestModeStrategy(ABC):
    """Per-request seam over standalone vs platform mode.

    Select once via :func:`select_request_mode_strategy`, call :meth:`resolve`,
    then read the populated attributes. The handler never branches on mode for
    credential resolution or usage settlement again.
    """

    is_platform: bool

    def __init__(self) -> None:
        self.route: ResolvedRoute | None = None
        self.user_token: str | None = None
        self.api_key_id: str | None = None
        self.user_id: str | None = None
        self.reservation: ReservationHandle | None = None
        self.rate_limit_info: RateLimitInfo | None = None

    @abstractmethod
    async def resolve(self, *, raw_request: Request, response: Response, spec: ResolveSpec) -> None:
        """Resolve credentials (platform) or authenticate + reserve budget
        (standalone), populating the per-request attributes above."""

    @abstractmethod
    async def resolve_mcp_servers(
        self, mcp_server_ids: list[uuid.UUID], *, reject_error: Exception
    ) -> list[McpServerConfig]:
        """Swap workspace-scoped MCP server ids for inline configs (platform),
        or reject the field (standalone raises ``reject_error``)."""

    @abstractmethod
    def make_settlement(
        self,
        *,
        provider: LLMProvider,
        model: str,
        endpoint: str,
        correlation_id: str | None,
    ) -> RequestSettlement:
        """Build the settlement object for this request's mode."""


class PlatformStrategy(RequestModeStrategy):
    is_platform = True

    def __init__(self, config: GatewayConfig) -> None:
        super().__init__()
        self._config = config

    async def resolve(self, *, raw_request: Request, response: Response, spec: ResolveSpec) -> None:
        self.user_token = _extract_platform_user_token(raw_request)
        start_time = time.perf_counter()
        route = await _resolve_platform_credentials(
            config=self._config,
            user_token=self.user_token,
            model_selector=spec.model_selector,
        )
        self.route = route
        resolve_latency_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Otari-Request-ID"] = route.request_id
        logger.info(
            "Platform resolve succeeded request_id=%s attempts=%d fallback_enabled=%s resolve_latency_ms=%.2f",
            route.request_id,
            len(route.attempts),
            route.fallback_enabled,
            resolve_latency_ms,
        )
        if spec.validate_provider is not None:
            # Validate every resolved attempt so a fallback that lands on an
            # unsupported provider fails fast here instead of crashing
            # mid-fallback inside the runner.
            for attempt in route.attempts:
                spec.validate_provider(LLMProvider(attempt.provider))

    async def resolve_mcp_servers(
        self, mcp_server_ids: list[uuid.UUID], *, reject_error: Exception
    ) -> list[McpServerConfig]:
        assert self.user_token is not None  # guaranteed by resolve()
        return await _resolve_platform_mcp_servers(
            config=self._config,
            user_token=self.user_token,
            mcp_server_ids=mcp_server_ids,
        )

    def make_settlement(
        self,
        *,
        provider: LLMProvider,
        model: str,
        endpoint: str,
        correlation_id: str | None,
    ) -> RequestSettlement:
        assert correlation_id is not None  # platform settlement always has a correlation id
        return PlatformSettlement(config=self._config, correlation_id=correlation_id)


class StandaloneStrategy(RequestModeStrategy):
    is_platform = False

    def __init__(self, config: GatewayConfig, db: AsyncSession | None, log_writer: LogWriter) -> None:
        super().__init__()
        self._config = config
        self._db = db
        self._log_writer = log_writer

    async def resolve(self, *, raw_request: Request, response: Response, spec: ResolveSpec) -> None:
        if self._db is None:
            raise spec.errors.db_unavailable
        api_key, is_master_key = await verify_api_key_or_master_key(raw_request, self._db, self._config)
        self.api_key_id = api_key.id if api_key else None
        self.user_id = resolve_user_id(
            user_id_from_request=spec.user_id_from_request,
            api_key=api_key,
            is_master_key=is_master_key,
            master_key_error=spec.errors.master_key_user_required,
            no_api_key_error=spec.errors.api_key_validation_failed,
            no_user_error=spec.errors.no_user,
            forbidden_user_error=spec.errors.forbidden_user,
            reject_mismatch=self._config.reject_user_mismatch,
        )
        self.rate_limit_info = check_rate_limit(raw_request, self.user_id)
        # Tolerate an unparseable selector here — the budget gate (404/403) and
        # the downstream provider call surface those; an unparseable model
        # simply has no pricing.
        try:
            gate_provider, gate_model = AnyLLM.split_model_provider(spec.model_selector)
        except (ValueError, AnyLLMError):
            gate_provider, gate_model = None, spec.model_selector
        gate_pricing = await find_model_pricing(self._db, gate_provider, gate_model)
        estimate = estimate_cost(
            gate_pricing,
            prompt_chars=spec.prompt_chars,
            max_output_tokens=spec.max_output_tokens,
            default_output_tokens=self._config.budget_estimate_default_output_tokens,
        )
        # Reserve first so user/blocked/budget rejections (404/403) take
        # precedence over the missing-pricing rejection (402); refund if we then
        # reject for missing pricing.
        self.reservation = await reserve_budget(
            self._db, self.user_id, estimate, model=spec.model_selector, strategy=self._config.budget_strategy
        )
        if pricing_required_but_missing(gate_pricing, require_pricing=self._config.require_pricing):
            await refund_reservation(self._db, self.reservation)
            raise spec.errors.no_pricing
        if spec.validate_provider is not None:
            # Standalone runs the provider-support guard after reservation,
            # matching the existing responses handler order.
            provider, _model = AnyLLM.split_model_provider(spec.model_selector)
            spec.validate_provider(provider)

    async def resolve_mcp_servers(
        self, mcp_server_ids: list[uuid.UUID], *, reject_error: Exception
    ) -> list[McpServerConfig]:
        raise reject_error

    def make_settlement(
        self,
        *,
        provider: LLMProvider,
        model: str,
        endpoint: str,
        correlation_id: str | None,
    ) -> RequestSettlement:
        return StandaloneSettlement(
            db=self._db,
            log_writer=self._log_writer,
            api_key_id=self.api_key_id,
            user_id=self.user_id,
            provider=provider,
            model=model,
            endpoint=endpoint,
            reservation=self.reservation,
            config=self._config,
        )


def select_request_mode_strategy(
    config: GatewayConfig,
    db: AsyncSession | None,
    log_writer: LogWriter,
) -> RequestModeStrategy:
    """Select the request mode strategy once per request from the gateway mode."""
    if config.is_platform_mode:
        return PlatformStrategy(config)
    return StandaloneStrategy(config, db, log_writer)
