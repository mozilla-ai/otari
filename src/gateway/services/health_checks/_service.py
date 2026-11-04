"""The scheduled health checks: their settings, a run, and what the Health page shows."""

import asyncio
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.models.health_checks import HealthCheckResult, HealthCheckRun, HealthCheckSettings
from gateway.repositories.health_checks import (
    HealthCheckRecoveryRepository,
    HealthCheckResultRepository,
    HealthCheckRunRepository,
    HealthCheckSettingsRepository,
)
from gateway.schemas.health_checks import (
    HealthCheckRecoveryCreate,
    HealthCheckResultPublic,
    HealthCheckRunPublic,
    HealthCheckSettingsPublic,
    HealthCheckSettingsUpdate,
    HealthCheckStatus,
)
from gateway.services.health_checks._alerts import (
    deployment_name,
    diff_alerts,
    render_health_alert,
    render_unavailable_alert,
)
from gateway.services.health_checks._probe import HealthCheckOffering, ProbeOutcome, capped_error, probe_all
from gateway.services.health_checks._summary import FailureStreak, failure_streaks, summarize
from gateway.services.inference import is_model_unavailable_error
from gateway.services.mail import Mailer
from gateway.services.upstream_redaction import redact_upstream_message

# Results are kept this long, which is what the page's history shows.
RETENTION = timedelta(hours=48)
HISTORY = timedelta(hours=24)
# An ended outage outlives the results it was read from.
RECOVERY_RETENTION = timedelta(days=90)
# A run unfinished after this long belongs to a replica that died mid-run.
ABANDONED_AFTER = timedelta(hours=1)
# Where an alert links to, the dashboard route of the Health page.
HEALTH_PAGE = "/#/health-checks"
HOSTED_PROVIDERS_PAGE = "/#/admin/hosted-providers"


ListOfferings = Callable[[], Awaitable[Sequence[HealthCheckOffering]]]


class HostedRoster(Protocol):
    """What the checks need of the hosted providers: which models they serve, and switching models off."""

    async def serveable_models(self, *, organization_id: uuid.UUID | None) -> dict[str, frozenset[str]]: ...

    async def switch_off_models(self, models: Mapping[str, Collection[str]]) -> int: ...

    async def switch_off_unavailable(self, provider: str, model: str) -> bool: ...


def refusal_switches_off(exc: BaseException, *, deployment_paid: bool) -> bool:
    """Whether ``exc`` should switch its model off: its provider refusing the model on the deployment's own credential.

    An organization's own key never switches anything: what its account may
    reach says nothing about the deployment's.
    """
    return deployment_paid and is_model_unavailable_error(exc)


def _model_of(result: HealthCheckResult) -> str:
    return result.selector.removeprefix(f"{result.provider}:")


def _served(result: HealthCheckResult, served: Mapping[str, frozenset[str]]) -> bool:
    return _model_of(result) in served.get(result.provider, frozenset())


def _settings_public(row: HealthCheckSettings) -> HealthCheckSettingsPublic:
    return HealthCheckSettingsPublic(
        schedule_enabled=row.schedule_enabled,
        interval_minutes=row.interval_minutes,
        prompt=row.prompt,
        max_tokens=row.max_tokens,
        reasoning_max_tokens=row.reasoning_max_tokens,
        reasoning_every_runs=row.reasoning_every_runs,
        next_run_at=row.next_run_at,
        run_requested=row.run_requested,
        alert_emails=list(row.alert_emails),
    )


def _run_public(row: HealthCheckRun) -> HealthCheckRunPublic:
    return HealthCheckRunPublic(
        id=str(row.id),
        started_at=row.started_at,
        finished_at=row.finished_at,
        included_reasoning=row.included_reasoning,
        total=row.total,
        passed=row.passed,
        failed=row.failed,
    )


def _result_public(row: HealthCheckResult, streak: FailureStreak | None = None) -> HealthCheckResultPublic:
    return HealthCheckResultPublic(
        selector=row.selector,
        provider=row.provider,
        status=row.status,  # type: ignore[arg-type]  # the check constraint holds it to the Literal
        status_code=row.status_code,
        error=row.error,
        latency_ms=row.latency_ms,
        checked_at=row.checked_at,
        failing_since=streak.since if streak else None,
        failed_runs=streak.runs if streak else 0,
    )


class HealthCheckService:
    """Reads and changes the settings, starts a run when one is due, and reports the latest results."""

    def __init__(
        self,
        uow: UnitOfWork,
        settings: HealthCheckSettingsRepository,
        runs: HealthCheckRunRepository,
        results: HealthCheckResultRepository,
        recoveries: HealthCheckRecoveryRepository,
        hosted: HostedRoster,
        config: GatewayConfig,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._uow = uow
        self._settings = settings
        self._runs = runs
        self._results = results
        self._recoveries = recoveries
        self._hosted = hosted
        self._config = config
        self._clock = clock

    async def get_settings(self) -> HealthCheckSettingsPublic:
        async with self._uow:
            return _settings_public(await self._settings.get_or_create())

    async def update_settings(self, request: HealthCheckSettingsUpdate) -> HealthCheckSettingsPublic:
        """Change the settings. Turning the checks on runs them within a poll."""
        now = self._clock()
        changes = request.model_dump(exclude_unset=True, exclude_none=True)
        async with self._uow:
            row = await self._settings.get_or_create()
            if changes.get("schedule_enabled") and not row.schedule_enabled:
                changes["next_run_at"] = now
            elif "interval_minutes" in changes and row.next_run_at is not None:
                changes["next_run_at"] = min(row.next_run_at, now + timedelta(minutes=changes["interval_minutes"]))
            row = await self._settings.update(row, changes)
            return _settings_public(row)

    async def request_run(self) -> HealthCheckSettingsPublic:
        """Ask for a run now, whether or not the schedule is on. A worker picks it up within a poll."""
        async with self._uow:
            row = await self._settings.get_or_create()
            row = await self._settings.update(row, {"run_requested": True})
            return _settings_public(row)

    async def get_status(self) -> HealthCheckStatus:
        now = self._clock()
        async with self._uow:
            settings = await self._settings.get_or_create()
            running = await self._runs.latest_unfinished(now - ABANDONED_AFTER)
            last = await self._runs.latest_finished()
            history_runs = await self._runs.finished_since(now - HISTORY)
            served = await self._hosted.serveable_models(organization_id=None)
            # A model switched off since its check is no longer the operator's to fix.
            latest = [result for result in await self._latest_results(last, now) if _served(result, served)]
            tallies = await self._results.tallies([run.id for run in history_runs])
            streaks = await self._streaks({result.selector for result in latest if result.status == "fail"}, now)
            return HealthCheckStatus(
                settings=_settings_public(settings),
                running=_run_public(running) if running is not None else None,
                last_run=_run_public(last) if last is not None else None,
                providers=summarize(latest, history_runs, tallies),
                results=[_result_public(result, streaks.get(result.selector)) for result in latest],
            )

    async def disable_failing(self) -> int:
        """Stop serving every model the latest results show failing. Returns how many were switched off.

        Its rows and prices stay on the Hosted providers page, where it is switched back on.
        """
        now = self._clock()
        async with self._uow:
            failing: dict[str, set[str]] = defaultdict(set)
            for result in await self._latest_results(await self._runs.latest_finished(), now):
                if result.status == "fail":
                    failing[result.provider].add(_model_of(result))
            return await self._hosted.switch_off_models(failing) if failing else 0

    async def _latest_results(self, last: HealthCheckRun | None, now: datetime) -> list[HealthCheckResult]:
        """Each model's newest result: the last run's, plus what it deferred, from the last run that checked them.

        Without the second part a failing reasoning model would vanish from the
        page on every run that leaves reasoning models out, while the alert list
        still counts it as failing. Only what the last run deferred is carried,
        so a model the deployment stopped serving leaves the page with it.
        Called inside an open unit of work.
        """
        if last is None:
            return []
        latest = list(await self._results.for_runs([last.id]))
        deferred = set(last.deferred_selectors) - {result.selector for result in latest}
        if deferred:
            reasoning_run = await self._runs.latest_finished_with_reasoning(now - RETENTION)
            if reasoning_run is not None:
                latest.extend(await self._results.for_runs([reasoning_run.id], deferred))
        return sorted(latest, key=lambda result: result.selector)

    async def run_if_due(self, list_offerings: ListOfferings) -> HealthCheckRunPublic | None:
        """Claim a due run and carry it out, or return None when none is due or another replica claimed it.

        The probes run outside any transaction, because a run takes up to a minute per offering. A model its
        provider refuses to serve is switched off, as a request meeting that refusal would.
        """
        now = self._clock()
        async with self._uow:
            settings = await self._settings.get_or_create()
            # A run still going holds the next one back, so two never send the same alert. A due run waits, and
            # a requested one stays requested, until it finishes or is abandoned.
            if await self._runs.latest_unfinished(now - ABANDONED_AFTER) is not None:
                return None
            number = await self._settings.claim_due_run(now, now + timedelta(minutes=settings.interval_minutes))
            if number is None:
                return None
            every = settings.reasoning_every_runs
            include_reasoning = every > 0 and (number - 1) % every == 0
            prompt, max_tokens, reasoning_max_tokens = (
                settings.prompt,
                settings.max_tokens,
                settings.reasoning_max_tokens,
            )
            run = await self._runs.start(now, included_reasoning=include_reasoning)
            run_id = run.id

        try:
            offerings = await list_offerings()
            targets = [offering for offering in offerings if include_reasoning or not offering.reasoning]
            deferred = sorted(
                offering.selector for offering in offerings if offering.reasoning and not include_reasoning
            )
            outcomes = await probe_all(
                targets, prompt=prompt, max_tokens=max_tokens, reasoning_max_tokens=reasoning_max_tokens
            )
        except BaseException:
            # Left in place, the run would read as running, and hold the next one back, for ABANDONED_AFTER.
            # A shutdown cancels the run mid-probe too, so the delete is shielded from that cancellation.
            await asyncio.shield(self._forget_run(run_id))
            raise

        finished = self._clock()
        async with self._uow:
            await self._results.add_all(
                run_id,
                (
                    HealthCheckResultPublic(
                        selector=outcome.target.selector,
                        provider=outcome.target.provider,
                        status=outcome.status,
                        status_code=outcome.status_code,
                        error=outcome.error,
                        latency_ms=outcome.latency_ms,
                        checked_at=finished,
                    )
                    for outcome in outcomes
                ),
            )
            started = await self._runs.get(run_id)
            if started is None:
                return None
            run = await self._runs.update(
                started,
                {
                    "finished_at": finished,
                    "total": sum(outcome.status != "skip" for outcome in outcomes),
                    "passed": sum(outcome.status == "pass" for outcome in outcomes),
                    "failed": sum(outcome.status == "fail" for outcome in outcomes),
                    "deferred_selectors": deferred,
                },
            )
            await self._record_recoveries(run_id, outcomes, finished)
            await self._prune(finished)
            public = _run_public(run)

        refused = await self._switch_off_refused(outcomes)
        await self._alert(
            run_id,
            [o for o in outcomes if o.target.selector not in refused],
            {offering.selector for offering in offerings} - refused,
            finished,
        )
        return public

    async def _switch_off_refused(self, outcomes: Sequence[ProbeOutcome]) -> set[str]:
        """Switch off each model its provider refused this run, as a request meeting the refusal would.

        Every offering is reached on the deployment's own credential, so the refusal counts. Returns the selectors
        now off, which leave the run's alert with nothing to fix. One that could not be switched off stays in it,
        and the next run tries again.
        """
        off: set[str] = set()
        for outcome in outcomes:
            if outcome.refusal is None:
                continue
            selector = outcome.target.selector
            try:
                await self.switch_off_unavailable(outcome.target.provider, outcome.target.model, outcome.refusal)
            except DATABASE_ERRORS as exc:
                # codeql[py/clear-text-logging-sensitive-data]
                logger.warning("Could not switch off unavailable model %s (%s)", selector, describe_database_error(exc))
                continue
            off.add(selector)
        return off

    async def switch_off_unavailable(self, provider: str, model: str, refusal: str) -> bool:
        """Switch ``provider:model`` off because its provider refused to serve it, and email the alert list.

        Called only for a refusal :func:`refusal_switches_off` accepts. The
        alert goes out once, from the call that switched the model off, with the
        provider's ``refusal`` redacted and capped. Returns whether this call
        switched it off.
        """
        if not await self._hosted.switch_off_unavailable(provider, model):
            return False
        selector = f"{provider}:{model}"
        # codeql[py/clear-text-logging-sensitive-data]
        logger.warning("Switched off hosted model %s: its provider refused to serve it", selector)
        await self._alert_unavailable(selector, capped_error(redact_upstream_message(refusal)))
        return True

    async def _alert_unavailable(self, selector: str, refusal: str) -> None:
        async with self._uow:
            recipients = list((await self._settings.get_or_create()).alert_emails)
        if not recipients:
            return
        mailer = Mailer(self._config)
        if not mailer.can_send_links:
            logger.warning(
                "Unavailable model alert not sent: this deployment cannot send mail (missing %s)",
                ", ".join(mailer.missing_settings),
            )
            return
        message = render_unavailable_alert(
            selector,
            refusal,
            deployment=deployment_name(self._config.effective_ui_base_url),
            providers_link=mailer.link(HOSTED_PROVIDERS_PAGE),
            health_link=mailer.link(HEALTH_PAGE),
            asset_root=mailer.asset_root,
        )
        for recipient in recipients:
            delivery = await mailer.send(to=recipient, message=message)
            if not delivery.delivered:
                # codeql[py/clear-text-logging-sensitive-data]
                logger.warning("Unavailable model alert for %s not delivered via %s", selector, delivery.transport)

    async def _forget_run(self, run_id: uuid.UUID) -> None:
        async with self._uow:
            await self._runs.delete_ids([run_id])

    async def _streaks(
        self, selectors: set[str], now: datetime, *, before: uuid.UUID | None = None
    ) -> dict[str, FailureStreak]:
        """The failure streaks of ``selectors``, read only for them so a page poll stays small.

        ``before`` leaves that run out, which is how an alert reads the streak a recovery ended.
        Called inside an open unit of work.
        """
        if not selectors:
            return {}
        runs = [run for run in await self._runs.finished_since(now - RETENTION) if run.id != before]
        results = await self._results.for_runs([run.id for run in runs], selectors)
        return failure_streaks(runs, results)

    async def _alert(
        self, run_id: uuid.UUID, outcomes: Sequence[ProbeOutcome], offered: set[str], checked_at: datetime
    ) -> None:
        """Email the alert list what this run changed, then remember it as told.

        Remembered only once at least one address took the email, so a failed
        send is retried by the next run rather than lost. With nobody to tell,
        or no way to send, the change is remembered all the same: adding an
        address later should start from what is failing then, not replay every
        change since the checks began.
        """
        async with self._uow:
            settings = await self._settings.get_or_create()
            alerted, recipients = list(settings.alerted_selectors), list(settings.alert_emails)
            change = diff_alerts(alerted, outcomes, offered)
            down_for = (
                await self._streaks({o.target.selector for o in change.recovered}, checked_at, before=run_id)
                if change and recipients
                else {}
            )
        if change and recipients:
            mailer = Mailer(self._config)
            if mailer.can_send_links:
                message = render_health_alert(
                    change,
                    outcomes,
                    down_for=down_for,
                    checked_at=checked_at,
                    deployment=deployment_name(self._config.effective_ui_base_url),
                    health_link=mailer.link(HEALTH_PAGE),
                    asset_root=mailer.asset_root,
                )
                deliveries = [await mailer.send(to=recipient, message=message) for recipient in recipients]
                if not any(delivery.delivered for delivery in deliveries):
                    return
            else:
                logger.warning(
                    "Health check alert not sent: this deployment cannot send mail (missing %s)",
                    ", ".join(mailer.missing_settings),
                )
        if list(change.alerted) != alerted:
            async with self._uow:
                row = await self._settings.get_or_create()
                await self._settings.update(row, {"alerted_selectors": list(change.alerted)})

    async def _record_recoveries(
        self, run_id: uuid.UUID, outcomes: Sequence[ProbeOutcome], recovered_at: datetime
    ) -> None:
        """Keep a row for each model this run found answering after a streak of failures.

        Called inside an open unit of work, once the run's results are stored.
        """
        answered = {outcome.target.selector: outcome for outcome in outcomes if outcome.status == "pass"}
        ended = await self._streaks(set(answered), recovered_at, before=run_id)
        await self._recoveries.add_all(
            HealthCheckRecoveryCreate(
                selector=selector,
                provider=answered[selector].target.provider,
                failing_since=streak.since,
                recovered_at=recovered_at,
                failed_runs=streak.runs,
                status_code=streak.last.status_code if streak.last else None,
                error=streak.last.error if streak.last else None,
            )
            for selector, streak in ended.items()
        )

    async def _prune(self, now: datetime) -> None:
        await self._recoveries.delete_before(now - RECOVERY_RETENTION)
        expired: list[uuid.UUID] = await self._runs.ids_started_before(now - RETENTION)
        await self._results.delete_for_runs(expired)
        await self._runs.delete_ids(expired)
