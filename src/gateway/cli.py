import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import click
import uvicorn
from uvicorn.config import logger

from gateway.core.config import API_ROOT, load_config
from gateway.log_config import setup_logger
from gateway.main import create_app

_LOG_LEVEL_NAMES: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _parse_log_level(ctx: click.Context, param: click.Parameter, value: str | None) -> int:
    """Map a symbolic (DEBUG/INFO/...) or numeric log level to its numeric value."""
    if value is None:
        return logging.INFO
    normalized = value.strip().upper()
    if normalized in _LOG_LEVEL_NAMES:
        return _LOG_LEVEL_NAMES[normalized]
    if normalized.isdigit():
        return int(normalized)
    choices = ", ".join(_LOG_LEVEL_NAMES)
    raise click.BadParameter(
        f"{value!r} is not a valid log level. Choose one of {choices} (case-insensitive) "
        "or a numeric level such as 20."
    )


@click.group()
def cli() -> None:
    """Otari CLI."""


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config YAML file",
    default=None,
)
@click.option("--host", default=None, help="Host to bind the server to")
@click.option("--port", default=None, type=int, help="Port to bind the server to")
@click.option("--database-url", envvar="DATABASE_URL", help="Database connection URL")
@click.option(
    "--master-key",
    envvar="OTARI_MASTER_KEY",
    help="Master key for management endpoints",
)
@click.option(
    "--auto-migrate/--no-auto-migrate",
    default=None,
    help="Automatically run database migrations on startup",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    help="Number of worker processes. Only 1 is supported today; values greater than 1 are rejected.",
)
@click.option(
    "--log-level",
    default="INFO",
    callback=_parse_log_level,
    help="Logging level (case-insensitive): DEBUG, INFO, WARNING, ERROR, CRITICAL. Numeric levels are also accepted.",
)
def serve(
    config: str | None,
    host: str | None,
    port: int | None,
    database_url: str | None,
    master_key: str | None,
    auto_migrate: bool | None,
    workers: int,
    log_level: int,
) -> None:
    """Start the Otari server."""
    if workers > 1:
        raise click.ClickException(
            "Otari does not support running more than one worker process yet. "
            "uvicorn only honors workers greater than 1 when it is given an import string, "
            "but Otari builds the app in-process from your resolved config, and its startup "
            "hooks (schema init and bootstrap key creation) are not safe to run once per worker. "
            "To scale out, run several otari processes behind a load balancer or process manager. "
            "Re-run with --workers 1 (the default)."
        )
    try:
        gateway_config = load_config(config)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    setup_logger(level=log_level)

    if host:
        gateway_config.host = host
    if port:
        gateway_config.port = port
    if database_url:
        gateway_config.database_url = database_url
    if master_key:
        gateway_config.master_key = master_key
    if auto_migrate is not None:
        gateway_config.auto_migrate = auto_migrate

    gateway_config.validate_mode_selection()

    if gateway_config.is_hybrid_mode:
        platform_base_url = gateway_config.platform.get("base_url")
        if not platform_base_url:
            raise click.ClickException("platform.base_url is required when hybrid mode is active")
        if gateway_config.providers:
            raise click.ClickException(
                "Local provider credentials are not supported in hybrid mode. Remove configured providers."
            )
        logger.info("Hybrid mode active. Base URL: %s", platform_base_url)

    if not gateway_config.master_key and not gateway_config.is_hybrid_mode:
        logger.info(
            "No master key configured; one will be generated and printed at startup. "
            "Set OTARI_MASTER_KEY (or --master-key) to choose your own instead.",
        )

    logger.info("Starting Otari on %s:%s", gateway_config.host, gateway_config.port)
    if gateway_config.is_hybrid_mode:
        logger.info("Database: disabled (hybrid mode)")
    else:
        logger.info("Database: %s", gateway_config.database_url)

    if gateway_config.providers:
        logger.info("Configured providers: %s", ", ".join(gateway_config.providers.keys()))

    app = create_app(gateway_config)

    try:
        uvicorn.run(
            app,
            host=gateway_config.host,
            port=gateway_config.port,
        )
    except KeyboardInterrupt:
        logger.info("\nShutting down Otari...")
        sys.exit(0)


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--database-url", envvar="DATABASE_URL", help="Database connection URL")
def init_db(config: str | None, database_url: str | None) -> None:
    """Initialize the database schema."""
    from gateway.db import init_db as db_init

    gateway_config = load_config(config)

    if database_url:
        gateway_config.database_url = database_url

    click.echo(f"Initializing database: {gateway_config.database_url}")

    db_init(gateway_config)

    click.echo("Database initialized successfully!")


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--database-url", envvar="DATABASE_URL", help="Database connection URL")
@click.option("--revision", default="head", help="Target revision (default: head)")
def migrate(config: str | None, database_url: str | None, revision: str) -> None:
    """Run database migrations using Alembic."""
    gateway_config = load_config(config)

    if database_url:
        gateway_config.database_url = database_url

    if not re.match(r"^[a-zA-Z0-9_+\-]+$", revision):
        click.echo(f"Invalid revision format: {revision}", err=True)
        sys.exit(1)

    alembic_path = shutil.which("alembic")
    if not alembic_path:
        click.echo("alembic command not found in PATH", err=True)
        sys.exit(1)

    click.echo(f"Running migrations on: {gateway_config.database_url}")
    click.echo(f"Target revision: {revision}")

    env = os.environ.copy()
    env["OTARI_DATABASE_URL"] = gateway_config.database_url

    try:
        result = subprocess.run(  # noqa: S603 validated up a few lines
            [alembic_path, "upgrade", revision],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        click.echo(result.stdout)
        click.echo("Migrations completed successfully!")
    except subprocess.CalledProcessError as e:
        click.echo(f"Migration failed: {e.stderr}", err=True)
        sys.exit(1)


@cli.command(name="gen-secret-key")
def gen_secret_key() -> None:
    """Print a fresh OTARI_SECRET_KEY for encrypting stored provider credentials.

    Set the printed value as OTARI_SECRET_KEY before adding provider keys in the
    dashboard. Keep it safe: losing it makes every stored provider key
    undecryptable.
    """
    from gateway.services.secret_box import generate_secret_key

    click.echo(generate_secret_key())


@cli.group()
def routing() -> None:
    """Inspect routing policies."""


@routing.command(name="explain")
@click.argument("policy_name", required=False)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config YAML file",
    default=None,
)
@click.option("--user", "user_id", default=None, help="Evaluate conditions as this user id.")
@click.option("--key-id", default=None, help="Evaluate conditions as this API key id.")
@click.option(
    "--budget-used-pct",
    type=float,
    default=None,
    help="Pretend this much of the caller's budget is committed, to exercise a tier-down rule.",
)
@click.option(
    "--budget-remaining-usd",
    type=float,
    default=None,
    help="Pretend this much budget is left.",
)
@click.option(
    "--allowed-model",
    "allowed_models",
    multiple=True,
    help="Restrict to these instance:model entries (repeatable), as an API key's allow-list would.",
)
def routing_explain(
    policy_name: str | None,
    config: str | None,
    user_id: str | None,
    key_id: str | None,
    budget_used_pct: float | None,
    budget_remaining_usd: float | None,
    allowed_models: tuple[str, ...],
) -> None:
    """Show what a policy compiles to, without sending a request anywhere.

    A routing policy's whole job is to make a choice the caller cannot see, so
    there has to be a way to see it. This prints the ordered plan, why the first
    candidate was selected, and every candidate that was dropped with the reason,
    which is the failure mode worth catching early: a "failover" policy whose
    fallbacks were all filtered out is a single attempt wearing a chain's name.

    Reads config only. No database, no provider call, nothing billed. The budget
    options let a tier-down rule be exercised without waiting for real spend to
    cross the threshold.
    """
    from gateway.models.routing import PolicySpec
    from gateway.services.routing import BudgetState, NoEligibleCandidatesError, compile_policy
    from gateway.services.routing.backends import backend_is_weighted
    from gateway.services.routing.decide import explain_router_ordering

    cfg = load_config(config)
    if not cfg.routing.policies:
        click.echo(
            "No routing policies are configured in config.yml. Add a `routing.policies` block there, or, if "
            "your policies were created through the dashboard or the API, note that this command reads config "
            f"only: it has no database. Use `POST {API_ROOT}/routing/policies/explain` against a running gateway to "
            "compile a stored policy."
        )
        raise SystemExit(1)
    if not cfg.routing.enabled:
        click.echo("Note: routing.enabled is false, so these policies are not in effect for requests.\n")

    if policy_name is None:
        click.echo("Configured policies:")
        for name, listed in cfg.routing.policies.items():
            shape = f"router:{listed.router_backend}" if listed.router_backend else (
                "dynamic" if listed.is_dynamic else "static"
            )
            candidates = len(listed.router_candidates) or 1
            click.echo(f"  {name}  ({shape}, {candidates + len(listed.on_failure)} candidate(s))")
        click.echo("\nPass a policy name to see its compiled plan.")
        return

    spec: PolicySpec | None = cfg.routing.policies.get(policy_name)
    if spec is None:
        known = ", ".join(cfg.routing.policies) or "none"
        raise click.BadParameter(f"unknown policy {policy_name!r}. Configured policies: {known}")

    budget = BudgetState(used_pct=budget_used_pct, remaining_usd=budget_remaining_usd)
    # A weighted policy's split is written in the policy, so it is knowable without
    # a request and this command shows it. Every other router needs request state
    # and gets None, which compiles to the decline path explained below.
    weighted_ordering, weighted_shares = explain_router_ordering(
        cfg, spec, user_id=user_id, allowlist=list(allowed_models) or None
    )
    try:
        plan = compile_policy(
            cfg,
            policy_name,
            spec,
            user_id=user_id,
            key_id=key_id,
            allowlist=list(allowed_models) or None,
            budget=budget,
            router_ordering=weighted_ordering,
        )
    except NoEligibleCandidatesError as exc:
        click.echo(f"{policy_name}: NO USABLE CANDIDATE")
        click.echo(f"  {exc.operator_detail}")
        raise SystemExit(1) from exc

    shares = {item.canonical: item.share_pct for item in weighted_shares}
    click.echo(f"{policy_name}: {len(plan.attempts)} candidate(s), selected by {plan.selection_reason}")
    for attempt in plan.attempts:
        canonical = f"{attempt.instance}:{attempt.model}"
        label = (
            f"weighted {shares[canonical]:.0f}%" if canonical in shares else attempt.selection_reason
        )
        click.echo(
            f"  {attempt.position}. {canonical}    [{label}]  dispatches as {attempt.dispatch_model}"
        )
    for dropped in plan.dropped:
        click.echo(f"  x  {dropped.selector}    dropped: {dropped.detail}")
    # Keyed on the backend rather than on the shares: a weighted policy whose whole
    # split is filtered out for this caller has no shares to print, and the decline
    # text below is the learned router's vocabulary, which would misdescribe it.
    if backend_is_weighted(spec.router_backend):
        click.echo(
            "  weighted: one candidate is drawn per request in proportion to its share, and a candidate "
            "that fails before responding falls to the next draw before on_failure. Shares are normalized "
            "over the candidates this caller may use, so they reflect the filtering above."
            if weighted_shares
            else "  weighted: no candidate in the split is usable by this caller, so the plan above is "
            "whatever the failure chain leaves. Every candidate in the split is listed as dropped, with "
            "the reason it went."
        )
    elif spec.router_backend is not None:
        # The plan above is the *decline* path, because a router needs a live
        # request (a prompt to embed, stored examples to compare it against) and
        # this command deliberately touches neither. Saying so beats printing a
        # one-candidate plan that looks like the router was ignored.
        click.echo(
            f"  router: '{spec.router_backend}' ranks {', '.join(spec.router_candidates)} at request time. "
            f"The plan above is what serves when it declines (cold pool, low confidence, tools present, "
            f"or Otari-Router: off)."
        )
    if plan.guardrails:
        click.echo("  guardrails (always enforced):")
        for guardrail in plan.guardrails:
            click.echo(
                f"    {guardrail.profile}  mode={guardrail.mode}  on_unavailable={guardrail.on_unavailable}"
            )
    if spec.is_dynamic:
        click.echo(
            "  note: this policy selects per request, so it has no single target or price. It works on "
            "/v1/chat/completions, /v1/messages and /v1/responses; on the other model-taking endpoints "
            "(embeddings, images, moderations, rerank, batches) it is not a resolvable model name."
        )


@cli.group(name="import")
def import_group() -> None:
    """Import usage that Otari did not proxy."""


@import_group.command(name="claude-code")
@click.option("--url", envvar="OTARI_URL", default="http://localhost:8000", help="Base URL of the Otari gateway.")
@click.option(
    "--api-key",
    envvar=["OTARI_API_KEY", "OTARI_MASTER_KEY"],
    default=None,
    help=(
        "Credential for the import endpoint: a budget-exempt API key "
        "(exclude_from_budget: true) or the master key. Imported usage is never "
        "budget-enforceable. Not needed with --dry-run."
    ),
)
@click.option(
    "--projects-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path.home() / ".claude" / "projects",
    show_default=True,
    help="Where Claude Code keeps its transcripts.",
)
@click.option(
    "--since",
    default=None,
    help="Only read transcripts modified since this ISO date or duration (7d, 24h, 2w).",
)
@click.option(
    "--user-id",
    default=None,
    help=(
        "Default user for the batch. Required when authenticating with the master key, and the user "
        "must already exist. Ignored with an API key, which binds usage to its own user."
    ),
)
@click.option(
    "--label-prefix",
    default=None,
    help="First half of session_label. Defaults to this machine's short hostname.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(1),
    default=None,
    help="Events per request. Defaults to the endpoint's own maximum.",
)
@click.option("--dry-run", is_flag=True, help="Parse and summarize without sending anything.")
def import_claude_code(
    url: str,
    api_key: str | None,
    projects_dir: Path,
    since: str | None,
    user_id: str | None,
    label_prefix: str | None,
    batch_size: int | None,
    dry_run: bool,
) -> None:
    """Backfill historical Claude Code usage from this machine's transcripts.

    The OTLP exporter documented in docs/use-with-claude-code.md only carries
    sessions that run after it is configured. This reads the transcripts Claude
    Code has already written and posts them to /v1/usage/external-events, which
    is idempotent on (source, source_event_id): re-running imports only what is
    new and reports the rest as duplicates.

    Do not backfill sessions that were routed through Otari. Their usage is
    already recorded, and the proxied and imported rows cannot be correlated, so
    the cost would appear twice.
    """
    import socket

    import httpx

    from gateway.services.claude_code_import import parse_since, scan_transcripts
    from gateway.services.external_usage_service import MAX_EVENTS_PER_BATCH

    try:
        cutoff = parse_since(since) if since is not None else None
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--since") from exc

    # The endpoint owns the cap, so it is read rather than restated on the option:
    # a decorator default would need the constant at import time, and importing the
    # ingest service to build the CLI would load the API stack for every command.
    if batch_size is None:
        batch_size = MAX_EVENTS_PER_BATCH
    elif batch_size > MAX_EVENTS_PER_BATCH:
        raise click.BadParameter(
            f"the endpoint accepts at most {MAX_EVENTS_PER_BATCH} events per request.",
            param_hint="--batch-size",
        )

    prefix = label_prefix or socket.gethostname().split(".")[0]
    result = scan_transcripts(projects_dir, label_prefix=prefix, since=cutoff)
    if result.unparsable_lines:
        # Reported before the empty check: a truncated transcript is exactly the
        # case that finds nothing to import, and silence there reads as "no usage".
        click.echo(
            f"Warning: {result.unparsable_lines} line(s) could not be decoded and were skipped. "
            "Any usage they carried was not imported."
        )
    if not result.events:
        click.echo(f"No usage found in {projects_dir}. Nothing to import.")
        return

    click.echo(
        f"Scanned {result.files_scanned} transcript(s): {len(result.events)} event(s), "
        f"{result.duplicates_skipped} repeated response id(s) collapsed, "
        f"{result.synthetic_skipped} local (non-API) message(s) skipped."
    )
    for model, tokens in sorted(result.tokens_by_model.items(), key=lambda item: -item[1]):
        click.echo(f"  {model}: {tokens:,} tokens")
    if dry_run:
        click.echo("Dry run: nothing was sent.")
        return

    if not api_key:
        raise click.UsageError(
            "A credential is required to send events: a budget-exempt API key, or the master key. "
            "Pass --api-key, or set OTARI_API_KEY or OTARI_MASTER_KEY. "
            "Re-run with --dry-run to preview a scan without one."
        )

    endpoint = f"{url.rstrip('/')}{API_ROOT}/usage/external-events"
    headers = {"Authorization": f"Bearer {api_key}"}
    # The first event is posted alone, so a mistake that will reject every event
    # (an unknown --user-id, a key that is not budget-exempt) costs one request and
    # one error line instead of the whole history and a truncated list of identical
    # rejections. The rest follow in full batches.
    batches = [result.events[:1]]
    batches += [result.events[start : start + batch_size] for start in range(1, len(result.events), batch_size)]

    accepted = duplicate = rejected = 0
    with httpx.Client(timeout=120.0) as client:
        for position, batch in enumerate(batches):
            body: dict[str, object] = {
                "source": "claude_code",
                "events": [event.as_payload() for event in batch],
            }
            if user_id is not None:
                body["user_id"] = user_id
            try:
                response = client.post(endpoint, json=body, headers=headers)
            except httpx.HTTPError as exc:
                click.echo(
                    f"Import stopped after {accepted} event(s): {exc}. "
                    "Re-running is safe: what already landed comes back as duplicates."
                )
                rejected += len(batch)
                break
            if response.status_code >= 400:
                # The whole batch failed validation or auth. Show what the server
                # said rather than a count, because the reason is the fix.
                click.echo(
                    f"Batch {position + 1} of {len(batches)} was refused "
                    f"({response.status_code}): {response.text[:500]}"
                )
                rejected += len(batch)
                if position == 0:
                    click.echo("Nothing else was sent. Fix the above and re-run.")
                    break
                continue
            outcome = response.json()
            accepted += int(outcome.get("accepted", 0))
            duplicate += int(outcome.get("duplicate", 0))
            batch_rejected = int(outcome.get("rejected", 0))
            rejected += batch_rejected
            for error in outcome.get("errors", [])[:5]:
                click.echo(f"  rejected: {error.get('detail')}")
            if position == 0 and batch_rejected:
                click.echo(
                    f"The first event was rejected, so the remaining {len(result.events) - len(batch)} "
                    "were not sent. Fix the above and re-run."
                )
                break

    click.echo(f"Imported {accepted} event(s); {duplicate} already present; {rejected} rejected.")
    if rejected:
        raise SystemExit(1)



def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
