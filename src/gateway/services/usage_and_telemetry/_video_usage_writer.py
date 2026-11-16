"""Recording what a settled video job used, in the transaction that claims it."""

from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.ports.usage_signal_port import RecordedUsage, UsageSignalPort
from gateway.repositories.usage_and_telemetry import UsageLogRepository, settled_job_row
from gateway.services.inference import CompletedVideoUsage
from gateway.services.log_writer import recorded_usage
from gateway.services.pricing_service import pricing_provenance


class VideoUsageWriter:
    """Records what a settled video job used, in the same transaction that claims the job.

    NOTE: callers own that transaction and must not record the usage through the shared usage log,
    which may drop what it has already accepted.
    """

    def __init__(self, uow: UnitOfWork, signal: UsageSignalPort) -> None:
        self._rows = UsageLogRepository(uow)
        self._signal = signal
        self._facts: dict[str, RecordedUsage] = {}

    async def record(self, usage: CompletedVideoUsage) -> None:
        charge = usage.charge
        pricing = usage.pricing if charge is not None else None
        source, reference, effective_at = pricing_provenance(pricing) if pricing is not None else (None, None, None)
        row = settled_job_row(
            workspace_id=usage.workspace_id,
            api_key_id=usage.api_key_id,
            user_id=usage.user_id,
            model=usage.model,
            provider=usage.provider,
            endpoint=usage.endpoint,
            counts_toward_budget=usage.counts_toward_budget,
            deployment_paid=usage.deployment_paid,
            cost=usage.cost,
            billing_meters=dict(charge.meters) if charge is not None and charge.meters else None,
            pricing_breakdown=charge.lines if charge is not None and charge.lines else None,
            pricing_source=source,
            pricing_reference=reference,
            pricing_effective_at=effective_at,
        )
        self._facts[usage.video_id] = recorded_usage(row)
        await self._rows.add(row)

    async def recorded(self, usage: CompletedVideoUsage) -> None:
        """Announces the usage once the caller has committed it."""
        facts = self._facts.pop(usage.video_id, None)
        if facts is None:
            return
        try:
            await self._signal.usage_recorded(facts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Usage row not signaled (%s)", type(exc).__name__)
