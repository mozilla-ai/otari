"""Read Claude Code's local transcripts into importable usage events.

Claude Code speaks the Anthropic Messages API, but a subscription-backed session
never routes through Otari, so its spend is invisible here. `Importing external
usage <../../docs/external-usage.md>`_ closes that gap going forward: point the
OTLP exporter at a gateway and new sessions arrive. What the exporter cannot
reach is the history already on disk from before it was configured, which for a
daily Claude Code user is most of the usage there has ever been. This module
reads that history so it can be posted to ``POST /v1/usage/external-events``.

Claude Code writes one JSONL transcript per session under
``~/.claude/projects/<mangled-cwd>/<session-id>.jsonl``. An assistant line carries
the API response id, the model, and the Anthropic usage block. The same response
id repeats once per content block of a single reply, so the walk is deduplicated
by response id: counting every line would multiply a session's tokens by the
number of blocks its replies happened to contain. Lines whose model is
``<synthetic>`` are dropped: those are messages Claude Code wrote itself, such as
an API error notice, and no request was ever sent for them.

Pure and offline. No network, no database, and no prompt or completion text is
read out of a transcript: a line is decoded, its usage numbers and identifiers
are taken, and the rest is dropped. The caller posts what this returns.
"""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# The endpoint's identifier grammar (``_IDENT_PATTERN`` in
# ``external_usage_service``). Anything outside it is a 422 for the whole batch,
# so model names and labels are folded into it here rather than discovered to be
# unpostable a thousand events later.
_IDENT_ALLOWED = re.compile(r"[^A-Za-z0-9._:/\-]")
# Claude Code appends a context-window tag to the model it reports, e.g.
# ``claude-sonnet-4-5[1m]`` for the 1M-context variant. The brackets are outside
# the grammar above, and the tag names a context window rather than a distinct
# priced model, so it is stripped: keeping it would leave every such event
# unpriced under a model id no price list has.
_CONTEXT_TAG = re.compile(r"\[[^\]]*\]")
# A project directory is the session's working directory with separators and dots
# replaced by dashes ("-Users-alice-Projects-otari"). Reconstructing the original
# path is impossible, because a dash in a real directory name is
# indistinguishable from a separator.
_LEADING_DASH = re.compile(r"^-+")
# Claude Code writes an assistant line with this model for text it produced
# locally rather than fetched, most often an API error notice shown in the
# transcript. It carries a usage block like any other line, but no request was
# ever made, so importing one invents a call that never happened and puts a
# model nobody can price into the breakdown.
_SYNTHETIC_MODEL = "<synthetic>"
_DURATION = re.compile(r"^(\d+)([hdw])$")
_DURATION_UNITS = {"h": "hours", "d": "days", "w": "weeks"}

# Claude Code is an Anthropic client, but ``ANTHROPIC_BASE_URL`` points it at
# anything that speaks the Messages API, and the transcript records whatever
# model that backend named. Mapping only families we can identify, and admitting
# "unknown" otherwise, keeps a mispriced guess out of the usage log: an unpriced
# row is visibly unpriced, while one priced as Anthropic is silently wrong.
_PROVIDER_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("gemini", "google"),
    ("kimi", "moonshot"),
)
UNKNOWN_PROVIDER = "unknown"


@dataclass(frozen=True)
class UsageEvent:
    """One API call, shaped for ``POST /v1/usage/external-events``."""

    source_event_id: str
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cache_write_1h_tokens: int
    session_label: str

    def as_payload(self) -> dict[str, Any]:
        """The event as the endpoint's JSON body wants it."""
        return {
            "source_event_id": self.source_event_id,
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_write_1h_tokens": self.cache_write_1h_tokens,
            # Claude Code reports the Anthropic shape, where the cache buckets are
            # additive rather than a subset of the input count.
            "cache_tokens_in_prompt": False,
            "session_label": self.session_label,
        }


@dataclass
class ScanResult:
    """Everything a scan found, plus the counts a caller reports."""

    events: list[UsageEvent] = field(default_factory=list)
    files_scanned: int = 0
    duplicates_skipped: int = 0
    unparsable_lines: int = 0
    synthetic_skipped: int = 0

    @property
    def tokens_by_model(self) -> dict[str, int]:
        """Total tokens per model, for the summary a run prints."""
        totals: dict[str, int] = {}
        for event in self.events:
            totals[event.model] = totals.get(event.model, 0) + (
                event.input_tokens
                + event.output_tokens
                + event.cache_read_tokens
                + event.cache_write_tokens
                + event.cache_write_1h_tokens
            )
        return totals


def parse_since(value: str) -> datetime:
    """Read ``--since`` as an ISO-8601 timestamp or a duration like ``7d``.

    Returns an aware UTC datetime. A bare date is taken as midnight UTC.
    """
    match = _DURATION.match(value.strip().lower())
    if match is not None:
        amount, unit = int(match.group(1)), _DURATION_UNITS[match.group(2)]
        return datetime.now(timezone.utc) - timedelta(**{unit: amount})
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{value!r} is neither an ISO-8601 date/time nor a duration like '7d', '24h', or '2w'."
        ) from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def normalize_model(raw: str | None) -> str:
    """Fold a reported model id into the endpoint's identifier grammar."""
    stripped = _CONTEXT_TAG.sub("", raw or "")
    cleaned = _IDENT_ALLOWED.sub("-", stripped).strip("-")
    return cleaned or "unknown"


def provider_for_model(model: str) -> str:
    """The provider a model id belongs to, or ``unknown`` when it is not ours to guess."""
    lowered = model.lower()
    for prefix, provider in _PROVIDER_BY_PREFIX:
        if lowered.startswith(prefix):
            return provider
    return UNKNOWN_PROVIDER


def mangle_path(path: Path) -> str:
    """Encode a path the way Claude Code names its project directories.

    Not a path operation, which is why pathlib cannot do it: the encoding flattens
    a path into one directory name, replacing dots as well as separators, so
    ``~/.claude/worktrees`` arrives as ``-Users-alice--claude-worktrees``. Only the
    separators are pathlib's business, and ``as_posix`` is what makes them one
    character to replace on Windows too.
    """
    return path.as_posix().replace("/", "-").replace(".", "-")


def session_label(project_dir: str, prefix: str, *, home: Path | None = None) -> str:
    """Build ``<prefix>:<project>`` from a transcript's project directory name.

    The home directory's own prefix is dropped, which keeps a username out of the
    label; what remains is kept whole rather than reduced to its last segment,
    because the last segment collides badly in practice. Sessions run from
    worktrees under ``.claude/worktrees`` would all label as a bare branch name,
    identical across every repository on the machine.
    """
    trimmed = _LEADING_DASH.sub("", project_dir)
    home_prefix = _LEADING_DASH.sub("", mangle_path(home if home is not None else Path.home()))
    if home_prefix and trimmed.startswith(f"{home_prefix}-"):
        trimmed = trimmed[len(home_prefix) + 1 :]
    host = _IDENT_ALLOWED.sub("-", prefix).strip("-") or "local"
    name = _IDENT_ALLOWED.sub("-", trimmed).strip("-") or "unknown"
    return f"{host}:{name}"[:256]


def iter_transcripts(projects_dir: Path, since: datetime | None = None) -> Iterator[Path]:
    """Yield every transcript under ``projects_dir``, optionally filtered by ``since``.

    ``since`` is compared against file modification time rather than the timestamps
    inside, so a run can skip whole files without opening them. A session appended
    to after the cutoff is therefore read in full and its older events are
    re-submitted, which the endpoint absorbs as duplicates.
    """
    if not projects_dir.is_dir():
        return
    cutoff = since.timestamp() if since is not None else None
    for path in sorted(projects_dir.rglob("*.jsonl")):
        if cutoff is not None:
            try:
                if path.stat().st_mtime < cutoff:
                    continue
            except OSError:
                # Tolerated for the same reason the read loop tolerates one: a
                # transcript that vanished or cannot be read is not a reason to
                # abandon the scan.
                continue
        yield path


def _usage_event(record: dict[str, Any], label: str) -> UsageEvent | None:
    """Turn one decoded transcript line into an event, or None when it carries no usage."""
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    usage = message.get("usage")
    response_id = message.get("id")
    timestamp = record.get("timestamp")
    if not isinstance(usage, dict) or not isinstance(response_id, str) or not isinstance(timestamp, str):
        return None
    # Anthropic splits cache writes by TTL. ``cache_creation_input_tokens`` is the
    # total; the 1h share is priced differently, so it is reported separately and
    # subtracted rather than counted twice.
    cache_creation = usage.get("cache_creation")
    write_1h = 0
    if isinstance(cache_creation, dict):
        write_1h = _int(cache_creation.get("ephemeral_1h_input_tokens"))
    write_total = _int(usage.get("cache_creation_input_tokens"))
    model = normalize_model(message.get("model"))
    return UsageEvent(
        source_event_id=response_id,
        timestamp=timestamp,
        provider=provider_for_model(model),
        model=model,
        input_tokens=_int(usage.get("input_tokens")),
        output_tokens=_int(usage.get("output_tokens")),
        cache_read_tokens=_int(usage.get("cache_read_input_tokens")),
        cache_write_tokens=max(write_total - write_1h, 0),
        cache_write_1h_tokens=write_1h,
        session_label=label,
    )


def _is_synthetic(record: dict[str, Any]) -> bool:
    """Whether a line is Claude Code's own text rather than a provider response."""
    message = record.get("message")
    return isinstance(message, dict) and message.get("model") == _SYNTHETIC_MODEL


def _int(value: Any) -> int:
    """Read a token count, treating a missing or non-numeric one as zero."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def scan_transcripts(projects_dir: Path, *, label_prefix: str, since: datetime | None = None) -> ScanResult:
    """Walk every transcript under ``projects_dir`` and collect its usage events.

    Deduplicated by response id across the whole scan, not per file: a resumed
    session can repeat a reply in a second transcript, and the endpoint would
    reject neither, it would simply record the first and call the second a
    duplicate. Doing it here keeps the batch honest and the summary accurate.
    """
    result = ScanResult()
    seen: set[str] = set()
    for path in iter_transcripts(projects_dir, since):
        result.files_scanned += 1
        label = session_label(path.relative_to(projects_dir).parts[0], label_prefix)
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if '"usage"' not in line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        result.unparsable_lines += 1
                        continue
                    if not isinstance(record, dict):
                        continue
                    if _is_synthetic(record):
                        result.synthetic_skipped += 1
                        continue
                    event = _usage_event(record, label)
                    if event is None:
                        continue
                    if event.source_event_id in seen:
                        result.duplicates_skipped += 1
                        continue
                    seen.add(event.source_event_id)
                    result.events.append(event)
        except OSError:
            # A transcript being rewritten, or one this user cannot read, is not a
            # reason to abandon the other several thousand.
            continue
    result.events.sort(key=lambda event: event.timestamp)
    return result
