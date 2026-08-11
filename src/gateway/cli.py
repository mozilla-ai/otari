import logging
import os
import re
import shutil
import subprocess
import sys

import click
import uvicorn
from uvicorn.config import logger

from gateway.core.config import load_config
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
            "only: it has no database. Use `POST /v1/routing/policies/explain` against a running gateway to "
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


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
