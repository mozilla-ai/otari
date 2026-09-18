import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
import uvicorn
from uvicorn.config import logger

from gateway.agent_runtime.domain.evaluators import matched_changed_paths
from gateway.agent_runtime.domain.policy import PolicyError, parse_policy
from gateway.agent_runtime.domain.types import JudgeGate
from gateway.core.config import API_KEY_HEADER, API_ROOT, load_config
from gateway.log_config import setup_logger

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
        f"{value!r} is not a valid log level. Choose one of {choices} (case-insensitive) or a numeric level such as 20."
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
    from gateway.main import create_app

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


# Claude Code's own edit tools and the tool_input field naming their target.
_HOOK_EDIT_TOOL_PATH_FIELDS = {"Edit": "file_path", "Write": "file_path", "NotebookEdit": "notebook_path"}

# Claude Code's shell tool and the tool_input field naming the command it is
# about to run. A PreToolUse call for this tool is the only evidence a
# command_match gate gets before the command runs; see docs/agent-gates.md.
_HOOK_COMMAND_TOOL_FIELDS = {"Bash": "command"}

# Mirrors the Hook Server's own per-command bound (routes/hooks.py's
# _MAX_COMMAND_LENGTH). A literal rather than an import: this command talks to
# a gateway over HTTP that may be a different build, so the number it truncates
# to is its own best guess at the far side's limit, not a shared constant that
# would imply the two are always one process.
_HOOK_MAX_COMMAND_LENGTH = 4096

# Mirror routes/hooks.py's own _MAX_COMMANDS/_MAX_TOTAL_COMMAND_CHARS, for the
# same reason _HOOK_MAX_COMMAND_LENGTH does: per-command truncation alone does
# not bound the total. A Stop event now submits every Bash command the whole
# session ran, not the single command a PreToolUse call would carry, so
# reaching this aggregate is a real, not pathological, outcome of a long
# session with several long commands (501 commands truncated to
# _HOOK_MAX_COMMAND_LENGTH each already clears 2,000,000 characters). Left
# unbounded, the server 422s the whole request, and that failure is total: it
# takes every gate in the policy with it, changed_path included, not just the
# command-evidence ones.
_HOOK_MAX_COMMANDS = 10_000
_HOOK_MAX_TOTAL_COMMAND_CHARS = 2_000_000


def _bound_commands_for_submission(commands: list[str]) -> list[str] | None:
    """Keep a Stop event's collected commands within the Hook Server's own request-size bounds,
    or submit no command evidence at all rather than an arbitrary subset of it.

    Dropping whole commands, unlike truncating one to its head
    (`_HOOK_MAX_COMMAND_LENGTH`, applied before this is called), loses each
    one entirely: which commands survive is an accident of chronological
    order with no relationship to which one a gate actually cared about. A
    dropped forbidden command would read as a false pass; a dropped required
    one would read as a false fail. Evidence the caller could not submit in
    full is None, the same principle `evaluators.py`'s own module docstring
    already states for a missing evidence list altogether: a required
    `command_match`/`command_if_changed` gate then resolves `unknown` and
    blocks, rather than risking either outcome on data known to be
    incomplete.

    This does not revisit `_HOOK_MAX_COMMAND_LENGTH`'s own, separately
    reasoned trade-off: keeping the head of one oversize command (rather
    than dropping it, or the whole submission, outright) is deliberate,
    since a Bash call carrying a heredoc clears that limit routinely and a
    forbidden/required phrase is usually near a command's own head, its own
    invocation.
    """
    if len(commands) > _HOOK_MAX_COMMANDS:
        click.echo(
            f"otari hook: session ran {len(commands):,} commands, over the {_HOOK_MAX_COMMANDS:,} "
            "limit; submitting no command evidence rather than an arbitrary subset of it.",
            err=True,
        )
        return None

    total_chars = sum(len(command) for command in commands)
    if total_chars > _HOOK_MAX_TOTAL_COMMAND_CHARS:
        click.echo(
            f"otari hook: session command evidence totals {total_chars:,} characters, over the "
            f"{_HOOK_MAX_TOTAL_COMMAND_CHARS:,} limit; submitting no command evidence rather than an "
            "arbitrary subset of it.",
            err=True,
        )
        return None

    return commands


def _hook_find_repo_root(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _hook_collect_changed_paths(repo_root: Path) -> list[str] | None:
    """Evidence for a `changed_path` gate on a Stop event: what Git sees changed.

    Claude Code's Stop payload carries no file list of its own (unlike
    PreToolUse, whose tool_input already names a target), so a Stop-time
    changed_path check has nothing to evaluate unless something goes and
    finds out what changed. Git status is that something: harness-agnostic
    (the same command regardless of which tool wrote the change, unlike
    parsing Claude Code's own transcript format) and ground truth for the
    working tree, including a change a `Bash` call made that no tool_input
    ever named. Specific to changed_path: a future gate type collects its
    own evidence in its own way, not through this function.
    """
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, explicit cwd
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",  # not the platform locale default, which is not always UTF-8
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return None
    # -z: NUL-delimited and never quotes or octal-escapes a path (unlike the
    # human-readable format, which renders a non-ASCII name like "café.txt"
    # as the escaped "caf\303\251.txt" and would report an untracked file
    # literally named "weird -> name.txt" as a rename by matching " -> " as
    # a substring of the one path it has, rather than the separator between
    # two). A rename or copy (status X or Y is 'R'/'C') is two consecutive
    # tokens, new path then old path, not one token with an arrow in it.
    tokens = result.stdout.split("\0")
    paths = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token:
            index += 1
            continue
        status, path = token[:2], token[3:]
        paths.append(path)
        index += 2 if ("R" in status or "C" in status) else 1
    return paths


# Claude Code's own wrapper around a hook's blocking stderr, written as the
# denied tool call's `tool_result` content: "{event}:{tool_name} hook error:
# [{hook_command}]: {stderr}". Both substrings, not just "hook error:" alone,
# because a *PostToolUse* hook can also fail this way and that call already
# executed; only a PreToolUse denial means the command never ran. This is
# Claude Code's own internal message shape, not a documented contract, so it
# is a best-effort signal: failing to recognize a denial (an unmatched
# format change) leaves the command in evidence rather than dropping it,
# which is the safer direction for a footgun-catcher to fail in.
_PRETOOLUSE_DENIAL_MARKERS = ("PreToolUse:", "hook error:")


def _tool_result_text(content: object) -> str:
    """Flatten a `tool_result` block's `content` to plain text, whichever shape it is.

    Anthropic's own API allows either a bare string or a list of content
    blocks; Claude Code's transcripts use the bare-string form for a hook
    denial specifically (confirmed against a real transcript), but nothing
    guarantees that stays true, so both are handled.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block["text"] for block in content if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def _hook_collect_transcript_commands(transcript_path: Path) -> list[str] | None:
    """Evidence for a `command_match`/`command_if_changed` gate on a Stop event.

    Claude Code's own Stop payload names no commands either, same as it
    names no changed files (see `_hook_collect_changed_paths`), but it does
    carry `transcript_path`: the session's own JSONL transcript on disk,
    one record per line. Each Bash call the session made is recorded as a
    `message.content[]` block with `type: "tool_use"`, `name: "Bash"`, and
    `input.command`; this walks every line collecting those, in the order
    they appear. A record with `isSidechain: true` (a subagent's own turn)
    is skipped: its commands are not commands this policy's own agent ran,
    even if Claude Code ever starts interleaving them into the same file
    (it does not today; a subagent transcript is its own file).

    A `Bash` call a `PreToolUse` hook denied is excluded: it is recorded in
    the transcript as a `tool_use` block like any other, whether or not it
    was allowed to run, and the transcript's only record of the denial is a
    later `tool_result` block naming the same `tool_use_id`, `is_error:
    true`, with content matching `_PRETOOLUSE_DENIAL_MARKERS`. Without this,
    a command a policy already blocked once at `PreToolUse` keeps failing
    every later `Stop` for the same, never-executed attempt, and worse for
    `command_if_changed`: a *denied* attempt at the required command would
    read as though it had run, satisfying a gate it never actually did.

    Returns None only when the transcript itself cannot be read (missing,
    permissions, not a file): the same fail-open sentinel
    `_hook_collect_changed_paths` uses, so the caller can tell "collected,
    and there are none" (an empty list) apart from "could not collect at
    all". A single malformed line is skipped, not fatal, matching
    `services/claude_code_import.py`'s tolerance of the same file format.
    """
    try:
        lines = transcript_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    # tool_use_id is None for a block missing or misshaping its own id: kept
    # in the requested list regardless (never silently dropped for that),
    # just ineligible to ever match an entry in denied_ids.
    requested: list[tuple[str | None, str]] = []
    denied_ids: set[str] = set()
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("isSidechain"):
            continue
        message = record.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_use" and block.get("name") == "Bash":
                tool_input = block.get("input")
                command = tool_input.get("command") if isinstance(tool_input, dict) else None
                if isinstance(command, str) and command:
                    tool_use_id = block.get("id")
                    requested.append((tool_use_id if isinstance(tool_use_id, str) else None, command))
            elif block_type == "tool_result" and block.get("is_error"):
                tool_use_id = block.get("tool_use_id")
                text = _tool_result_text(block.get("content"))
                if isinstance(tool_use_id, str) and all(marker in text for marker in _PRETOOLUSE_DENIAL_MARKERS):
                    denied_ids.add(tool_use_id)

    return [command for tool_use_id, command in requested if tool_use_id is None or tool_use_id not in denied_ids]


# A judge gate's prompt is rubric + diff + transcript excerpt, each bounded
# independently so one huge file or one long session can't build an unbounded
# `claude -p` argv. Sized against a real measurement, not the "~4 chars/token"
# English-prose estimate `_hook_estimate_tokens` uses for its dry-run label:
# a real `claude -p` call against a 501,517-char prompt (diff + this transcript
# format) came back "Prompt is too long · the request is ~290,782 tokens
# (limit 200000)" — a ratio of ~1.7 chars/token, not ~4, because a session
# transcript is JSONL (escaped strings, tool payloads), not prose. A second
# real call at 100,505 chars consumed ~73,383 tokens total and succeeded with
# comfortable headroom under the 200K limit. These two bounds keep the worst
# case (diff + transcript, before the rubric/template's own much smaller
# overhead) near that validated-safe combined size rather than the model's
# own much larger context window, leaving margin for `claude -p`'s own fixed
# per-invocation overhead (system prompt, tool definitions) on top.
_HOOK_JUDGE_MAX_DIFF_CHARS = 60_000
_HOOK_JUDGE_MAX_TRANSCRIPT_CHARS = 40_000
# A judge call's own bound, deliberately separate from the 10s git status/diff
# calls above: those are local filesystem operations with nothing to wait on
# but disk, while this one is a full model invocation. 120s measured too tight
# in practice: a trivial `claude -p` call with no otari involvement at all
# measured over 2 minutes of wall-clock invocation overhead in one real run,
# unrelated to prompt size. Chosen with real headroom over that.
_HOOK_JUDGE_TIMEOUT_SECONDS = 300

# Each judge gate costs one sequential model invocation, unlike the other gate
# types (near-instant pattern matching), so an unbounded gate count means
# unbounded wall-clock on a single Stop event: N gates at the timeout above
# would be N * 300s in the worst case. Capped, with a visible truncation
# message, the same "never let something scale unbounded and silently" rule
# _bound_commands_for_submission and the Hook Server's own work-estimate
# budgets (routes/hooks.py) already follow. Evaluated in declaration order, so
# the same gates run first every time rather than an arbitrary subset.
_HOOK_JUDGE_MAX_GATES_PER_RUN = 5

# Haiku, not the session's own (often larger) default model: a judge call is a
# small, structured pass/fail classification over bounded text, not the kind
# of task that needs a frontier model, and every judge gate in a policy costs
# one full invocation against the caller's own subscription (see
# _hook_run_judge). Overridable per-invocation with --judge-model /
# OTARI_HOOK_JUDGE_MODEL for a rubric that genuinely needs more capability.
_HOOK_JUDGE_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_HOOK_JUDGE_PROMPT_TEMPLATE = """\
You are reviewing a code change against exactly one rule. Reply with exactly \
one JSON object and nothing else, no other prose, no markdown fence: \
{{"outcome": "pass" or "fail", "reasoning": "one or two sentences"}}.

Rule to judge:
{rubric}

Diff of the changes made this session:
{diff}

Transcript of the session that made this change:
{transcript}
"""


def _hook_extract_judge_transcript(transcript_path: Path) -> str:
    """The assistant's own text replies from the session transcript, for a judge gate's prompt.

    A raw transcript is mostly `tool_use`/`tool_result` payloads (a Bash
    call's own stdout, a Read's file contents, ...): bytes that dominate the
    file's size but carry no "why was this change made" signal a judge rubric
    can use, and the reason `_HOOK_JUDGE_MAX_TRANSCRIPT_CHARS` needed a real
    ratio measurement rather than the char-per-token heuristic (see that
    constant's own comment). Keeping only each assistant record's own `text`
    content blocks is both smaller and more relevant than a raw byte slice of
    the file. `isSidechain` records (a subagent's own turn) are excluded, the
    same as `_hook_collect_transcript_commands`: not reasoning this session's
    own agent produced about the change under judgment. `thinking` blocks are
    excluded too: usually the more verbose, less-final restatement of the
    same `text` reply that follows it.

    Returns "" when the transcript cannot be read at all (missing,
    permissions) or carries no assistant text, the same as an empty
    transcript otherwise would: a judge gate degrades to diff-only evidence
    rather than treating this as a collection failure.
    """
    try:
        lines = transcript_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""

    texts: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("isSidechain") or record.get("type") != "assistant":
            continue
        message = record.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        texts.extend(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
    return "\n".join(texts)


def _hook_collect_diff(repo_root: Path) -> str | None:
    """The working tree's own diff against HEAD, for a judge gate's prompt.

    Tracked changes only (`git diff HEAD`): a new, untracked file's content is
    a known gap in this first iteration, not a silent one, since
    `_hook_collect_changed_paths` already reports its path in `changed_paths`
    even though this diff carries none of its content. Returns None only when
    Git itself could not answer (no HEAD yet, not a repository), mirroring
    `_hook_collect_changed_paths`'s own fail-open sentinel.
    """
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, explicit cwd
        ["git", "diff", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return None
    diff = result.stdout
    if len(diff) > _HOOK_JUDGE_MAX_DIFF_CHARS:
        click.echo(
            f"otari hook: diff is {len(diff):,} characters, over the {_HOOK_JUDGE_MAX_DIFF_CHARS:,} limit; "
            "judge gates will see only the first that many.",
            err=True,
        )
        diff = diff[:_HOOK_JUDGE_MAX_DIFF_CHARS] + "\n... (diff truncated)"
    return diff


def _hook_build_judge_prompt(rubric: str, diff: str, transcript: str) -> str:
    return _HOOK_JUDGE_PROMPT_TEMPLATE.format(
        rubric=rubric,
        diff=diff or "(no diff collected)",
        transcript=transcript or "(no transcript collected)",
    )


# ~4 characters per token is the usual rule of thumb for English prose (the
# same order of magnitude Anthropic's own docs use for rate-limit planning).
# Not a real tokenizer count: getting an exact one would mean either bundling
# a tokenizer or a network call to a counting endpoint, and this estimate is
# only ever surfaced in the dry-run audit trail, never used for anything that
# needs to be exact.
_HOOK_JUDGE_CHARS_PER_TOKEN_ESTIMATE = 4


def _hook_estimate_tokens(text: str) -> int:
    return len(text) // _HOOK_JUDGE_CHARS_PER_TOKEN_ESTIMATE


def _hook_judge_log_path() -> Path:
    """Where every real judge-gate model call is locally, append-only logged.

    Under the user's home directory, not the repo: a `claude -p` invocation
    is billed against the machine's own subscription regardless of which
    repo triggered it, so the audit trail belongs somewhere that survives a
    `git clean` and is never accidentally committed.
    """
    return Path.home() / ".otari" / "judge-calls.log"


def _hook_judge_workdir() -> Path:
    """Where the judge gate's own `claude -p` call runs, instead of the caller's repo.

    That call's prompt is fully self-contained text (rubric, diff, transcript
    excerpt), so it never needs to run from the repo it is judging. Running it
    there anyway is how a real early version of this recursed into itself: the
    repo's own `.claude/settings.local.json` registers `otari hook` for
    `Stop`, and Claude Code resolves that file by directory, not by "is this
    the top-level session", so an unguarded call whose own `Stop` hook is this
    same command triggered it again, and again.

    A plain isolated directory, not a "disable hooks" flag (`--safe-mode` or
    `--bare`; the latter also breaks OAuth/keychain auth) for one deliberate
    reason: `--safe-mode`/`--bare` foreclose ever attaching a hook to this
    specific call on purpose, which is very nearly the point of a `judge`
    gate calling out to a model at all. A dedicated directory under
    `~/.otari/` (not the shared system temp root, and not the repo being
    judged) is a stable, otari-owned place a future judge-specific hook or
    its own `.otari-gates.yml` could live, the same reasoning
    `_hook_judge_log_path` already applies to the audit log. Today it holds
    nothing, so nothing resolves from it: no hooks, since Claude Code walks
    up from `cwd` looking for a `.claude/settings.local.json` and finds none
    there, narrower than `--safe-mode`'s guarantee (project-scoped only, not
    a hypothetical user- or enterprise-level hook), which does not matter
    here because `otari hook setup` only ever writes to a repo's own
    project-scoped settings, never to the user's.
    """
    workdir = Path.home() / ".otari" / "judge-workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def _hook_log_judge_call(repo_root: Path, gate_id: str, outcome: str, *, detail: str | None = None) -> None:
    """Append one line for a judge-gate model call this process actually attempted (or, in
    `--judge-dry-run`, would have attempted).

    Best-effort: a failure to write this log (a read-only home directory, a
    full disk) must never turn into a failed hook, so any OSError here is
    swallowed rather than propagated. Records only enough to answer "how
    many real model calls has this run, for which gate, and when", plus, for
    a dry-run line, the estimated prompt size: never the rubric, diff,
    transcript, or the model's own output, none of which belongs in a
    plaintext file kept indefinitely.
    """
    try:
        log_path = _hook_judge_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = f"{datetime.now(UTC).isoformat()} repo={repo_root} gate={gate_id!r} outcome={outcome}"
        if detail:
            line += f" detail={detail!r}"
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")
    except OSError:
        pass


# The prompt tells the model to reply with exactly one JSON object, no
# markdown fence, but a model wrapping it in one anyway (```json ... ```)
# is common enough in practice (confirmed against a real `claude -p` call)
# that treating it as a parse failure would report "error" on an answer
# that was, in substance, a real and well-formed verdict.
_HOOK_JUDGE_CODE_FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```\s*$", re.DOTALL)


def _hook_strip_judge_code_fence(raw: str) -> str:
    stripped = raw.strip()
    fence_match = _HOOK_JUDGE_CODE_FENCE.match(stripped)
    return fence_match.group(1).strip() if fence_match else stripped


# The exact wording of `claude -p`'s own "prompt is too long" rejection,
# confirmed against a real call, on stdout rather than stderr and with a
# zero-token usage report (the request was rejected before any tokenization
# was billed). Matched case-insensitively as a substring, not parsed further:
# this is a signal to retry smaller, not a value this command needs to carry.
_HOOK_JUDGE_PROMPT_TOO_LONG_MARKER = "prompt is too long"

# Mirrors the Hook Server's own JudgeVerdictRequest.reasoning cap
# (routes/hooks.py, _MAX_REASONING_LENGTH): the prompt asks for "one or two
# sentences" but nothing enforces that on the model's side, and an oversize
# reasoning otherwise 422s the *whole* /hooks/check request, which this
# command's own fail-open handling for a rejected request (not blocking)
# would then silently skip every other gate in the same policy along with
# it, mechanical and required ones included.
_HOOK_MAX_JUDGE_REASONING_LENGTH = 4_096


def _hook_call_claude_p(claude_path: str, model: str, prompt: str) -> tuple[str, str]:
    """One `claude -p --model <model>` invocation; return (outcome, reasoning).

    outcome is always one of "pass"/"fail"/"error": a nonzero exit, a
    timeout, or output that is not the single JSON object the prompt demands
    are all "error", carrying the failure detail as reasoning rather than
    raising. `claude -p`'s own "prompt is too long" rejection exits nonzero
    with the message on stdout, not stderr (confirmed against a real call),
    so the error detail falls back to stdout when stderr is empty.

    Runs with `cwd` set to `_hook_judge_workdir()`, never the repo being
    judged: see that function's own docstring for why (a real recursive
    incident) and why that is an isolated directory rather than a
    hooks-disabling flag.
    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, resolved executable path
            [claude_path, "--model", model, "-p", prompt],
            cwd=_hook_judge_workdir(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_HOOK_JUDGE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "error", f"claude -p did not respond within {_HOOK_JUDGE_TIMEOUT_SECONDS}s"

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return "error", f"claude -p exited {result.returncode}: {detail[:500]}"

    try:
        verdict = json.loads(_hook_strip_judge_code_fence(result.stdout))
    except ValueError:
        return "error", f"claude -p did not return valid JSON: {result.stdout[:500]!r}"

    outcome = verdict.get("outcome") if isinstance(verdict, dict) else None
    reasoning = verdict.get("reasoning") if isinstance(verdict, dict) else None
    if outcome not in ("pass", "fail") or not isinstance(reasoning, str):
        return "error", f"claude -p returned an unrecognized verdict shape: {result.stdout[:500]!r}"

    return outcome, reasoning[:_HOOK_MAX_JUDGE_REASONING_LENGTH]


def _hook_run_judge(
    rubric: str, diff: str, transcript: str, *, model: str, dry_run: bool = False
) -> tuple[str, str]:
    """Invoke `claude -p --model <model>` for one judge gate's rubric; return (outcome, reasoning).

    Otari itself never calls a model (see JudgeGate's own docstring); this is
    that call, made locally against the caller's own Claude Code
    subscription, not billed through Otari.

    `dry_run` skips the real call entirely, before ever touching `shutil.which`
    or `subprocess`: the wire contract has no fourth outcome to spell "this
    was never really run", so it reports the same `"error"` a real failed
    call would, with a `reasoning` that says so explicitly and estimates the
    prompt's size, never `"pass"`/`"fail"`, which would misrepresent a
    verdict nothing actually produced.

    A "prompt is too long" rejection retries once with `transcript` dropped
    entirely: the transcript is supplementary "why" context for a judge
    rubric, the diff is the primary evidence, so a diff-only retry is a
    strictly better fallback than reporting no verdict at all. Only for that
    specific rejection, and only once: any other failure, or a rejection that
    persists with no transcript left to drop, reports "error" as it always
    has.
    """
    if dry_run:
        prompt = _hook_build_judge_prompt(rubric, diff, transcript)
        estimated_tokens = _hook_estimate_tokens(prompt)
        return (
            "error",
            f"--judge-dry-run: real claude -p call skipped; prompt would have been "
            f"{len(prompt):,} chars (~{estimated_tokens:,} tokens estimated at "
            f"~{_HOOK_JUDGE_CHARS_PER_TOKEN_ESTIMATE} chars/token).",
        )

    claude_path = shutil.which("claude")
    if not claude_path:
        return "error", "the `claude` CLI was not found on PATH"

    outcome, reasoning = _hook_call_claude_p(claude_path, model, _hook_build_judge_prompt(rubric, diff, transcript))
    if outcome == "error" and transcript and _HOOK_JUDGE_PROMPT_TOO_LONG_MARKER in reasoning.lower():
        outcome, reasoning = _hook_call_claude_p(claude_path, model, _hook_build_judge_prompt(rubric, diff, ""))
    return outcome, reasoning


def _hook_collect_judge_verdicts(
    policy_yaml: str,
    gates_file: Path,
    repo_root: Path,
    transcript_path: str | None,
    changed_paths: list[str],
    *,
    judge_model: str,
    judge_dry_run: bool = False,
) -> list[dict[str, str]]:
    """Run every applicable judge gate in the local policy, one `claude -p` call each.

    Parses the policy locally with the same pure `domain.policy.parse_policy`
    the Hook Server itself uses, purely to find which gates are judge gates
    and read their `rubric`/`when_changed`; the Hook Server still re-parses
    and validates the submitted `policy_yaml` on its own, so a mismatch here
    only means judge evidence for a gate the server would reject anyway. A
    local parse failure collects no verdicts rather than raising: the
    existing fail-open request below still submits the policy text for the
    server to report the same error on.

    A judge gate with `when_changed` is skipped locally, before ever reading
    the diff/transcript or shelling out to `claude -p`, when none of
    `changed_paths` matches its globs (`domain.evaluators.matched_changed_paths`,
    the same grammar `evaluate_judge`'s own applicability check uses
    server-side). This is a local optimization only: submitting no verdict
    for a skipped gate resolves `not_applicable` there independently, the
    same as it would if this function ran the model call and got `pass`
    anyway. A gate with no `when_changed` at all keeps its unconditional,
    every-Stop-event behavior.

    `judge_dry_run` (see `hook`'s own `--judge-dry-run`) still runs this whole
    applicability check, still reads the diff and transcript, and still
    writes the same `_hook_log_judge_call` audit lines; only `_hook_run_judge`
    itself skips the real `claude -p` call. This is what makes the resulting
    log a real count of how often the model would have been invoked, not a
    guess: everything up to the call itself runs exactly as it would for real.
    """
    try:
        spec = parse_policy(policy_yaml, source=str(gates_file))
    except PolicyError:
        return []

    changed_paths_tuple = tuple(changed_paths)
    judge_gates = [
        gate
        for gate in spec.gates
        if isinstance(gate, JudgeGate)
        and (not gate.when_changed or matched_changed_paths(gate.when_changed, changed_paths_tuple))
    ]
    if not judge_gates:
        return []
    if len(judge_gates) > _HOOK_JUDGE_MAX_GATES_PER_RUN:
        skipped = [gate.id for gate in judge_gates[_HOOK_JUDGE_MAX_GATES_PER_RUN:]]
        click.echo(
            f"otari hook: {len(judge_gates):,} judge gates in this policy, over the "
            f"{_HOOK_JUDGE_MAX_GATES_PER_RUN:,} limit; skipping: {', '.join(skipped)}.",
            err=True,
        )
        judge_gates = judge_gates[:_HOOK_JUDGE_MAX_GATES_PER_RUN]

    diff = _hook_collect_diff(repo_root) or ""
    transcript = _hook_extract_judge_transcript(Path(transcript_path)) if transcript_path else ""
    if len(transcript) > _HOOK_JUDGE_MAX_TRANSCRIPT_CHARS:
        click.echo(
            f"otari hook: transcript is {len(transcript):,} characters, over the "
            f"{_HOOK_JUDGE_MAX_TRANSCRIPT_CHARS:,} limit; judge gates will see only the most recent that many.",
            err=True,
        )
        # Tail, not head: the most recent turns are the ones that produced
        # the change under judgment, the same "keep what's still relevant"
        # tradeoff _bound_commands_for_submission makes by dropping the
        # oldest commands first.
        transcript = transcript[-_HOOK_JUDGE_MAX_TRANSCRIPT_CHARS:]

    results = []
    for gate in judge_gates:
        # Logged before the call, not after: a hung or killed `claude -p`
        # invocation must still show up in the audit trail rather than
        # silently vanishing along with the process that would have logged
        # its outcome.
        _hook_log_judge_call(repo_root, gate.id, "invoking")
        outcome, reasoning = _hook_run_judge(gate.rubric, diff, transcript, model=judge_model, dry_run=judge_dry_run)
        _hook_log_judge_call(repo_root, gate.id, outcome, detail=reasoning if judge_dry_run else None)
        results.append({"gate_id": gate.id, "outcome": outcome, "reasoning": reasoning})
    return results


@cli.group(name="hook", invoke_without_command=True)
@click.option(
    "--harness",
    type=click.Choice(["claude-code"]),
    default="claude-code",
    show_default=True,
    help="Agent integration sending this callback.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to config YAML file, used to resolve --url/--api-key when they are not given.",
)
@click.option("--url", envvar="OTARI_URL", default=None, help="Base URL of the Otari gateway.")
@click.option("--api-key", envvar="OTARI_API_KEY", default=None, help="Credential for the Hook Server.")
@click.option(
    "--judge-model",
    envvar="OTARI_HOOK_JUDGE_MODEL",
    default=_HOOK_JUDGE_DEFAULT_MODEL,
    show_default=True,
    help="Model `claude -p` uses for a judge gate's model call.",
)
@click.option(
    "--judge-dry-run",
    envvar="OTARI_HOOK_JUDGE_DRY_RUN",
    is_flag=True,
    default=False,
    help=(
        "Run every applicable judge gate's own logic (policy parse, when_changed filtering, "
        "diff/transcript collection) but skip the real `claude -p` call, logging what would have "
        "run instead. For measuring how often a judge gate would fire before spending real model calls."
    ),
)
@click.pass_context
def hook(
    ctx: click.Context,
    harness: str,
    config: str | None,
    url: str | None,
    api_key: str | None,
    judge_model: str,
    judge_dry_run: bool,
) -> None:
    """Native callback entry point for a supported agent's hook protocol.

    Reads one JSON hook payload on stdin, collects the evidence that payload
    carries (a PreToolUse call's own target path, or a Stop event's Git
    status), and calls POST /api/v1/hooks/check. Never reads or evaluates the
    policy itself: gateway.agent_runtime does that; this command is a thin,
    harness-specific transport. See docs/agent-gates.md.

    Exit code is this harness's own protocol, not otari policy check's:
    Claude Code's PreToolUse and Stop hooks both take 0 (proceed) or 2 (block,
    stderr shown to the agent). Never blocks on a problem that is not a
    required gate failing: a missing policy, an unreachable gateway, or a
    missing credential all exit 0, with a message on stderr where there is
    one worth surfacing.

    A group, not a plain command, so `otari hook setup` can live alongside
    it: invoked with no subcommand (the shape every existing settings file
    already calls), it runs the callback above unchanged.
    """
    if ctx.invoked_subcommand is not None:
        return
    import httpx

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    event = payload.get("hook_event_name")
    repo = Path(payload.get("cwd") or Path.cwd())
    root = _hook_find_repo_root(repo)
    if root is None:
        return

    gates_file = root / ".otari-gates.yml"
    if not gates_file.is_file():
        return
    policy_yaml = gates_file.read_text(encoding="utf-8")

    changed_paths: list[str] = []
    # `[]`, not None, by default: PreToolUse's edit-tool branch below leaves
    # this as `[]` on purpose, meaning "no command evidence for this call",
    # the same not_applicable-not-unknown contract every other command-less
    # event has always had. Only the Stop branch may set this to None, when
    # it collected no evidence at all rather than collecting and finding
    # nothing.
    commands: list[str] | None = []
    # "call" unless the Stop branch below really does collect the whole
    # session: this is what tells the server which command-evidence gates can
    # resolve at all, rather than leaving each to guess from an empty list.
    command_scope = "call"
    # Only a Stop event collects judge verdicts (see _hook_collect_judge_verdicts):
    # a PreToolUse call has neither a full diff nor a finished transcript to
    # judge against yet, so it always submits none.
    judge_results: list[dict[str, str]] = []
    if event == "PreToolUse":
        tool_name = payload.get("tool_name", "")
        tool_input = payload.get("tool_input") or {}
        # A tool call is either an edit or a shell command, never both, so at
        # most one of these evidence lists is ever populated per call.
        path_field = _HOOK_EDIT_TOOL_PATH_FIELDS.get(tool_name)
        command_field = _HOOK_COMMAND_TOOL_FIELDS.get(tool_name)
        if path_field:
            target = tool_input.get(path_field)
            if not target:
                return
            try:
                # as_posix(), not str(): a forbidden glob is a repo-relative
                # POSIX path and the evaluator splits it on "/", so a
                # WindowsPath's native "docs\\foo.md" spelling matches
                # nothing. That fails open and silently, a passing gate being
                # indistinguishable from no forbidden change, so every
                # PreToolUse gate would pass on Windows.
                changed_paths = [Path(target).resolve().relative_to(root).as_posix()]
            except ValueError:
                return  # Outside the repo: nothing this policy can name.
        elif command_field:
            command = tool_input.get(command_field)
            if not command:
                return
            # Truncated rather than sent whole: the Hook Server rejects an
            # oversize command with a 422, and a 422 fails the *whole* check
            # open, taking every changed_path gate in the same policy with it.
            # A Bash call carrying a heredoc clears this limit routinely, so
            # that is the common case rather than a pathological one. A tool
            # name is argv[0], so keeping the head is what preserves detection
            # for the shape this gate is actually for.
            if len(command) > _HOOK_MAX_COMMAND_LENGTH:
                click.echo(
                    f"otari hook: command is {len(command):,} characters, checking only the first "
                    f"{_HOOK_MAX_COMMAND_LENGTH:,}.",
                    err=True,
                )
                command = command[:_HOOK_MAX_COMMAND_LENGTH]
            commands = [command]
        else:
            return  # A tool this harness integration does not check yet.
    elif event == "Stop":
        collected = _hook_collect_changed_paths(root)
        if collected is None:
            click.echo("otari hook: could not read Git state, not blocking.", err=True)
            return
        changed_paths = collected

        # transcript_path is Claude Code's own name for the session's JSONL
        # transcript on disk. Absent, or unreadable, submits None rather than
        # `[]`: `[]` means "collected, and there is none", which would let a
        # required command_match/command_if_changed gate read a failed
        # collection as a clean pass instead of the unresolved `unknown` it
        # actually is (see docs/agent-gates.md).
        transcript_path = payload.get("transcript_path")
        commands = _hook_collect_transcript_commands(Path(transcript_path)) if transcript_path else None
        command_scope = "session"
        if commands:
            # Same truncation the PreToolUse Bash branch applies to its one
            # command, applied per command here: a whole session's worth of
            # transcript-collected commands makes it more likely, not less,
            # that at least one clears _HOOK_MAX_COMMAND_LENGTH (a heredoc
            # anywhere in the session, not just in the single command a
            # PreToolUse call would carry), and one oversize entry would
            # otherwise 422 the whole check open.
            oversize = sum(1 for command in commands if len(command) > _HOOK_MAX_COMMAND_LENGTH)
            if oversize:
                click.echo(
                    f"otari hook: {oversize} command(s) from the transcript exceeded "
                    f"{_HOOK_MAX_COMMAND_LENGTH:,} characters, checking only the first that many of each.",
                    err=True,
                )
                commands = [command[:_HOOK_MAX_COMMAND_LENGTH] for command in commands]
            commands = _bound_commands_for_submission(commands)

        judge_results = _hook_collect_judge_verdicts(
            policy_yaml,
            gates_file,
            root,
            transcript_path,
            changed_paths,
            judge_model=judge_model,
            judge_dry_run=judge_dry_run,
        )
    else:
        return  # An event this harness integration does not check yet.

    try:
        gateway_config = load_config(config)
    except ValueError as exc:
        # load_config runs GatewayConfig.validate_mode_selection(), which
        # raises on a real misconfiguration (e.g. OTARI_MODE=hybrid with no
        # OTARI_AI_TOKEN). That is a setup problem, not a required gate
        # failing, so it falls under this command's own fail-open contract.
        click.echo(f"otari hook: could not load config ({exc}), not blocking.", err=True)
        return
    # host is a bind address (0.0.0.0 is the documented default), not a connect
    # target; a client dials localhost instead.
    connect_host = "localhost" if gateway_config.host == "0.0.0.0" else gateway_config.host  # noqa: S104
    resolved_url = url or f"http://{connect_host}:{gateway_config.port}"
    resolved_key = api_key or gateway_config.master_key
    if not resolved_key:
        click.echo("otari hook: no API key or master key resolved, not blocking.", err=True)
        return

    try:
        response = httpx.post(
            f"{resolved_url.rstrip('/')}{API_ROOT}/hooks/check",
            json={
                "policy_yaml": policy_yaml,
                "changed_paths": changed_paths,
                "commands": commands,
                "command_scope": command_scope,
                "judge_results": judge_results,
            },
            headers={API_KEY_HEADER: resolved_key},
            timeout=15.0,
        )
        response.raise_for_status()
        result = response.json()
        failing = [gate for gate in result["results"] if gate["outcome"] not in ("pass", "not_applicable")]
        blocked = result["blocked"]
    except httpx.HTTPStatusError as exc:
        # Split from the transport branch below on purpose: the request did
        # arrive and was answered, so "could not reach" would send whoever
        # debugs this to the network instead of to the status and body that
        # say what was actually wrong (a policy this build cannot parse, or
        # evidence over one of the route's limits).
        detail = exc.response.text[:500]
        click.echo(
            f"otari hook: {resolved_url} rejected the check ({exc.response.status_code}: {detail}), not blocking.",
            err=True,
        )
        return
    except httpx.HTTPError as exc:
        click.echo(f"otari hook: could not reach {resolved_url} ({exc}), not blocking.", err=True)
        return
    except (ValueError, TypeError, KeyError) as exc:
        # A body that is not JSON, or is JSON of a shape this command does not
        # recognize. Same fail-open contract as an unreachable gateway: this
        # command blocks on a required gate failing and on nothing else, so a
        # response it cannot read must not surface as a traceback.
        click.echo(f"otari hook: unreadable response from {resolved_url} ({exc!r}), not blocking.", err=True)
        return

    # `failing` mirrors Outcome's own non-blocking set (types.py), not just
    # "pass": a future gate type's not_applicable is a clean result too, and
    # must not get reported here as something the caller needs to look at.
    if not failing:
        return

    # .get(), not [...]: the try/except above only protects the shape checks
    # that build `failing` itself (result["results"], gate["outcome"]), not a
    # gate dict's other fields. A gate missing 'enforcement'/'gate_id'/
    # 'message' (an older or otherwise mismatched otari serve behind --url)
    # must not raise KeyError here, outside that protection, and surface as a
    # traceback in place of the fail-open message this command promises.
    summary = "\n".join(
        f"  [{'x' if gate.get('enforcement') == 'required' else '!'}] "
        f"{gate.get('gate_id', '?')}: {gate.get('message', '(no message)')}"
        for gate in failing
    )
    if blocked:
        # stop_hook_active is Claude Code's own signal that this Stop is
        # already the continuation a previous block forced. It matters because
        # Claude Code overrides a Stop hook that blocks eight times running
        # without progress, and then simply lets the turn end: a required gate
        # that quietly stops enforcing at the moment it is firing hardest is
        # worse than one that never fired, because the turn ends looking
        # clean. Exiting 0 here instead (the shape a hook with no fixable
        # failure wants) is not right for this one: every gate that can block
        # here is fixable, by running the required command or reverting the
        # forbidden change. So keep blocking, and say plainly that the block
        # is finite, so the agent spends the remaining attempts fixing the
        # gate or telling the user it cannot, rather than retrying blind.
        repeat_note = (
            "\n  (already blocked once this turn; Claude Code overrides a Stop hook after 8 "
            "consecutive blocks, so fix this now or say why you cannot.)"
            if payload.get("stop_hook_active")
            else ""
        )
        click.echo(f"otari hook: blocked ({harness}, {event}):\n{summary}{repeat_note}", err=True)
        raise SystemExit(2)
    # An advisory gate failed but nothing required did: warn without
    # blocking. Checking `blocked` alone here would silently drop this,
    # since only a required failure can ever set it true. Exit 0 with a
    # plain stderr message is invisible to the user: Claude Code only
    # surfaces a non-blocking hook's stderr in its own debug log, never in
    # the transcript or to the model. `systemMessage` on stdout is the
    # documented field for a visible, non-blocking hook message.
    click.echo(json.dumps({"systemMessage": f"otari hook: advisory warning(s) ({harness}, {event}):\n{summary}"}))


def _otari_binary_path() -> str:
    """Absolute path to this otari install's own binary.

    Claude Code's hook subprocess does not inherit an activated shell's PATH,
    so a bare "otari" often will not resolve. otari's own console-script
    wrapper sits next to the interpreter running it (same venv/bin), which is
    what sys.executable already names.
    """
    return str(Path(sys.executable).with_name("otari"))


def _resolve_hook_credential() -> str | None:
    """Whatever `otari hook` would resolve automatically at runtime, no flags given."""
    try:
        return load_config(None).master_key
    except ValueError:
        return None


def _gates_file_allows_bash(gates_file: Path) -> bool:
    """Whether the matcher should include Bash: only if a command_match gate exists.

    Parses gates_file the same way the Hook Server does. A missing or
    unparseable policy defaults to False, the narrower matcher: setup cannot
    know what a broken policy would have wanted, and the round trip is
    otherwise harmless but pointless to pay for nothing.
    """
    if not gates_file.is_file():
        return False
    from gateway.agent_runtime.domain.policy import PolicyError, parse_policy
    from gateway.agent_runtime.domain.types import CommandMatchGate

    try:
        spec = parse_policy(gates_file.read_text(encoding="utf-8"), source=str(gates_file))
    except PolicyError:
        return False
    return any(isinstance(gate, CommandMatchGate) for gate in spec.gates)


def _starter_gates_yaml(repo_name: str) -> str:
    return (
        'schema_version: "1.0"\n'
        "policy:\n"
        f"  id: {repo_name}/gates\n"
        "  description: Rules this repo checks on its own working tree.\n"
        "\n"
        "gates:\n"
        "  - id: no-force-push\n"
        "    type: command_match\n"
        "    enforcement: advisory\n"
        '    forbidden: ["git push --force"]\n'
        "    message: >-\n"
        "      Force-pushing rewrites shared history. Use --force-with-lease\n"
        "      if you must.\n"
    )


def _merge_hook_entry(settings_path: Path, event: str, command: str, *, matcher: str | None = None) -> bool:
    """Add or update an `event` hook entry (e.g. "PreToolUse", "Stop") pointing at otari hook.

    `matcher` is omitted (no key at all, not a null one) for an event that
    is not tool-scoped, `Stop` being the one this integration registers:
    Claude Code's own Stop hooks carry no `matcher`, unlike `PreToolUse`'s.

    Returns True if a new entry was appended, False if an existing one was
    found (by its command already starting with this same otari binary
    invoked as "hook", whatever flags it had) and updated in place instead of
    duplicated. Every other key in the file, including other hooks and
    permissions, other events, and any sibling hook command under the same
    matcher, is preserved untouched.
    """
    if settings_path.is_file():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"{settings_path} is not valid JSON: {exc}") from exc
        if not isinstance(settings, dict):
            raise click.ClickException(f"{settings_path} must contain a JSON object at the top level.")
    else:
        settings = {}

    hooks_section = settings.setdefault("hooks", {})
    if not isinstance(hooks_section, dict):
        raise click.ClickException(f'{settings_path}\'s "hooks" must be a JSON object.')
    entries = hooks_section.setdefault(event, [])
    if not isinstance(entries, list):
        raise click.ClickException(f'{settings_path}\'s "hooks.{event}" must be a JSON array.')
    otari_hook_prefix = command.split(" --", 1)[0]  # "<path> hook", before any flags

    updated = False
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("hooks"), list):
            continue
        for hook_item in entry["hooks"]:
            existing_command = hook_item.get("command") if isinstance(hook_item, dict) else None
            if isinstance(existing_command, str) and existing_command.startswith(otari_hook_prefix):
                if matcher is not None:
                    entry["matcher"] = matcher
                hook_item["command"] = command
                updated = True
                break
        if updated:
            break

    if not updated:
        new_entry: dict[str, object] = {"hooks": [{"type": "command", "command": command}]}
        if matcher is not None:
            new_entry = {"matcher": matcher, **new_entry}
        entries.append(new_entry)

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return not updated


@hook.command(name="setup")
@click.option(
    "--harness",
    type=click.Choice(["claude-code"]),
    default="claude-code",
    show_default=True,
    help="Agent integration to configure.",
)
@click.option(
    "--api-key",
    default=None,
    help="Skip automatic/interactive credential resolution and use this.",
)
def hook_setup(harness: str, api_key: str | None) -> None:
    """Register otari hook in a supported agent's own settings.

    Writes or updates a PreToolUse hook entry and a Stop hook entry in
    .claude/settings.local.json (personal, gitignored, never committed) so
    registering the Hook Server is not a manual JSON edit. Both point at the
    same otari hook invocation; Claude Code passes its own hook_event_name in
    the payload, so one callback serves either event. Offers to scaffold a
    starter .otari-gates.yml when this repo has none yet, and picks the
    PreToolUse matcher (whether it needs to cover Bash) from whatever gates
    the policy turns out to have; Stop needs no matcher; see
    docs/agent-gates.md for why both are registered unconditionally.
    """
    root = _hook_find_repo_root(Path.cwd())
    if root is None:
        raise click.ClickException("Not inside a Git repository.")

    gates_file = root / ".otari-gates.yml"
    if not gates_file.is_file():
        if click.confirm(f"No {gates_file.name} found in {root}. Create a starter policy?", default=True):
            gates_file.write_text(_starter_gates_yaml(root.name), encoding="utf-8")
            click.echo(f"Wrote {gates_file}.")
        else:
            click.echo(
                f"Skipping. otari hook will still be registered below, but every gate check "
                f"passes until {gates_file.name} exists; see docs/agent-gates.md."
            )

    include_bash = _gates_file_allows_bash(gates_file)
    matcher = "Edit|Write|NotebookEdit|Bash" if include_bash else "Edit|Write|NotebookEdit"

    embedded_key = api_key
    if not embedded_key:
        resolved_key = _resolve_hook_credential()
        if not resolved_key:
            click.echo(
                "Could not resolve a credential automatically (no master_key in config.yml, .env, or the environment)."
            )
            embedded_key = click.prompt("Enter an Otari API key or master key", hide_input=True)
        # A key resolved automatically is not embedded: the same resolution
        # otari hook already does at runtime keeps working, and this repeats
        # it rather than pinning today's value (e.g. a master key that later
        # rotates).

    command_parts = [_otari_binary_path(), "hook", "--harness", harness]
    if embedded_key:
        command_parts += ["--api-key", embedded_key]
    command = shlex.join(command_parts)

    settings_path = root / ".claude" / "settings.local.json"
    pretooluse_created = _merge_hook_entry(settings_path, "PreToolUse", command, matcher=matcher)
    click.echo(f"{'Added' if pretooluse_created else 'Updated'} the PreToolUse hook in {settings_path}.")
    click.echo(f"Matcher: {matcher}" + ("" if include_bash else " (add a command_match gate to also cover Bash)"))

    # Registered unconditionally, not only when the policy has a gate that
    # benefits: changed_path already falls back to `git status` on Stop
    # (catching a Bash-written change PreToolUse never saw coming), and
    # command_if_changed/command_match now read real command evidence there
    # too (from the session's own transcript; see docs/agent-gates.md). A
    # PreToolUse-only install left both silently unreachable.
    stop_created = _merge_hook_entry(settings_path, "Stop", command)
    click.echo(f"{'Added' if stop_created else 'Updated'} the Stop hook in {settings_path}.")


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
            shape = (
                f"router:{listed.router_backend}"
                if listed.router_backend
                else ("dynamic" if listed.is_dynamic else "static")
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
        label = f"weighted {shares[canonical]:.0f}%" if canonical in shares else attempt.selection_reason
        click.echo(f"  {attempt.position}. {canonical}    [{label}]  dispatches as {attempt.dispatch_model}")
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
            click.echo(f"    {guardrail.profile}  mode={guardrail.mode}  on_unavailable={guardrail.on_unavailable}")
    if spec.is_dynamic:
        click.echo(
            "  note: this policy selects per request, so it has no single target or price. It works on "
            f"{API_ROOT}/chat/completions, {API_ROOT}/messages and {API_ROOT}/responses; on the other "
            "model-taking endpoints (embeddings, images, moderations, rerank, batches) it is not a "
            "resolvable model name."
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
    Code has already written and posts them to /api/v1/usage/external-events, which
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
