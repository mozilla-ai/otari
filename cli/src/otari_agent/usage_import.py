"""`otari import`: bring in usage that Otari did not proxy."""

from pathlib import Path

import click

from otari_agent.claude_code_import import parse_since, scan_transcripts
from otari_agent.settings import API_ROOT

# The per-request cap of POST /api/v1/usage/external-events. It lives in the
# distribution both sides depend on, and the gateway's external_usage_service
# imports it from here, so the CLI's --batch-size check and the endpoint's own
# validation cannot drift apart.
MAX_EVENTS_PER_BATCH = 1000


@click.group(name="import")
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
    Code has already written and posts them to /api/v1/usage/external-events, which
    is idempotent on (source, source_event_id): re-running imports only what is
    new and reports the rest as duplicates.

    Do not backfill sessions that were routed through Otari. Their usage is
    already recorded, and the proxied and imported rows cannot be correlated, so
    the cost would appear twice.
    """
    import socket

    import httpx


    try:
        cutoff = parse_since(since) if since is not None else None
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--since") from exc

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
