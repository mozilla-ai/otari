"""`otari hook`, `otari hook setup` and the `otari guardrails` group: the agent-side half of Agent Guardrails.

Reads one hook payload from a supported coding agent, collects the evidence it
names, composes the repository's own guardrail files and evaluates them in
process (or, when opted
in, asks a gateway's Hook Server to) and answers in the harness's own exit-code
protocol. See docs/agent-guardrails.md.
"""

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import click
import yaml

from otari_agent.domain.check import PolicyCheckError, check_policy
from otari_agent.domain.evaluators import matched_changed_paths
from otari_agent.domain.policy import (
    MAX_POLICY_BYTES,
    MAX_POLICY_FILES,
    PolicyError,
    PolicyFile,
    compose_policy,
    parse_policy,
)
from otari_agent.domain.types import (
    CheckVerdict,
    CommandGate,
    EvidenceScope,
    GateResult,
    JudgeGate,
    JudgeVerdict,
    Outcome,
    PathGate,
    PolicySpec,
    RunsAt,
    VerifierGate,
    by_priority,
)
from otari_agent.domain.validation import validate_policy
from otari_agent.settings import API_KEY_HEADER, API_ROOT, load_settings

# Claude Code's own edit tools and the tool_input field naming their target.
_HOOK_EDIT_TOOL_PATH_FIELDS = {"Edit": "file_path", "Write": "file_path", "NotebookEdit": "notebook_path"}

# Claude Code's whole-file read tool and the tool_input field naming its
# target. Deliberately only Read: Grep and Glob return matching lines and file
# names rather than whole contents, and reading one line through a narrow
# pattern is the mitigation a secret gate's own message should recommend, so
# gating them would refuse the workaround. Codex has no entry here because it
# has no read tool: its reads go through the shell, where they are already
# `pre_tool_use.command` evidence.
_HOOK_READ_TOOL_PATH_FIELDS = {"Read": "file_path"}

# Claude Code's shell tool and the tool_input field naming the command it is
# about to run. A PreToolUse call for this tool is the only evidence a
# command gate gets before the command runs; see docs/agent-guardrails.md.
_HOOK_COMMAND_TOOL_FIELDS = {"Bash": "command"}

# Codex hook-dispatches its own shell tool under the same canonical name
# Claude Code uses ("Bash"; confirmed against openai/codex's own
# HookToolName), plus, once a turn runs through Code Mode, "code_mode_exec":
# a freeform JS snippet that can wrap any number of tools.exec_command()/
# tools.apply_patch() calls rather than naming a single command ("exec" is
# also accepted, in case a build reports the pre-canonicalization name). That
# whole snippet is kept as "the command" here rather than parsed apart: a
# forbidden phrase a command gate looks for still matches wherever it
# appears in it, and Code Mode's own PreToolUse dispatch is not complete yet
# (openai/codex#23411), so there is no reliable per-argument shape to parse
# even if it were worth the fragility.
_HOOK_COMMAND_TOOL_FIELDS_BY_HARNESS = {
    "claude-code": _HOOK_COMMAND_TOOL_FIELDS,
    "codex": {"Bash": "command", "code_mode_exec": "command", "exec": "command"},
}

# Codex's own edit tool, sharing Claude Code's apply_patch envelope
# convention: unlike Edit/Write, tool_input carries no bare file_path;
# "command" holds the whole patch text, and the path(s) it touches are named
# on the envelope's own header lines instead (_hook_extract_patch_paths).
_CODEX_PATCH_TOOL_NAME = "apply_patch"
# A rename is its own two-line shape, not a fourth header verb: "*** Update
# File: <old path>" immediately followed by "*** Move to: <new path>",
# neither one alone naming where the file ends up.
_PATCH_HEADER_RE = re.compile(r"^\*\*\* (?:(?:Add|Update|Delete) File|Move to): (.+)$", re.MULTILINE)


def _hook_evidence_paths(target: str, repo: Path, root: Path) -> list[str]:
    """Repo-relative spellings of a tool's declared target: the name asked for, and the file reached.

    Two candidates, because a symlink makes those different questions and a
    gate needs both answered.

    The lexical spelling is what a policy author wrote a glob against: `.env`
    is `.env` whatever it points at. Resolving first, and only, is what let a
    `.env` symlinked outside the repo escape every glob naming it, since the
    resolved path then sits outside the root and the whole check was
    abandoned. That is the ordinary layout where a checkout's `.env` points at
    a shared or home secrets file, so it failed open in exactly the case the
    rule was written for (CWE-59).

    The resolved spelling catches the other direction, which the lexical one
    cannot: an innocuous name aliased to a real secret.

    Either may land outside the repo, and one that does is dropped rather than
    abandoning the collection: a policy can only name paths inside the repo,
    but the other spelling is often still nameable. An empty list means
    neither was, and there is genuinely nothing for a gate to match.

    as_posix() throughout (via `_guardrails_relative_to`), not str(): a
    forbidden glob is a repo-relative POSIX path and the evaluator splits it
    on "/", so a WindowsPath's native "docs\\foo.md" spelling matches
    nothing, and that failure is silent and open too.
    """
    requested = Path(target)
    # normpath, not resolve: it collapses "." and ".." lexically, which is what
    # keeps a link's own name intact. Anchored to `repo`, the call's own
    # working directory, so a relative target does not silently resolve
    # against this process's cwd instead.
    lexical = Path(os.path.normpath(requested if requested.is_absolute() else repo / requested))
    candidates = [
        relative
        for candidate in (lexical, lexical.resolve())
        if (relative := _guardrails_relative_to(candidate, root)) is not None
    ]
    return list(dict.fromkeys(candidates))


def _hook_extract_patch_paths(patch_text: str) -> list[str]:
    """Target path(s) named in an apply_patch envelope's own header lines.

    One apply_patch call can touch several files, each named on its own
    "*** Add/Update/Delete File: <path>" header line; order-preserving and
    de-duplicated, since a policy gate cares about the set of touched paths,
    not how many headers happened to name each one. A rename's own "*** Move
    to: <path>" line is matched too, alongside the "Update File:" line naming
    its old path that always precedes one: both the vacated and the landed-on
    path are evidence a path gate could care about, and reporting
    only one would silently miss whichever gate is scoped to the other.
    """
    seen: dict[str, None] = {}
    for match in _PATCH_HEADER_RE.finditer(patch_text):
        path = match.group(1).strip()
        if path:
            seen[path] = None
    return list(seen)


# Mirrors the evaluator's own per-command bound
# (otari_agent.domain.check's _MAX_COMMAND_LENGTH). A literal rather than
# an import: the opt-in remote mode talks to a gateway over HTTP that may be
# a different build, so the number it truncates to is its own best guess at
# the far side's limit, not a shared constant that would imply the two are
# always one process. The default, local mode calls the same evaluator
# in process and would raise `PolicyCheckError` on the same bound anyway;
# truncating here means an oversize command still gets checked, just with
# its tail cut, rather than failing the whole check open.
_HOOK_MAX_COMMAND_LENGTH = 4096

# Mirror otari_agent.domain.check's own _MAX_COMMANDS/_MAX_TOTAL_COMMAND_CHARS,
# for the same reason _HOOK_MAX_COMMAND_LENGTH does: per-command truncation
# alone does not bound the total. A Stop event now submits every Bash
# command the whole session ran, not the single command a PreToolUse call
# would carry, so reaching this aggregate is a real, not pathological,
# outcome of a long session with several long commands (501 commands
# truncated to _HOOK_MAX_COMMAND_LENGTH each already clears 2,000,000
# characters). Left unbounded, the evaluator (local or remote) 422s/raises
# on the whole request, and that failure is total: it takes every gate in
# the policy with it, path included, not just the command-evidence
# ones.
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
    `command`/`command_if_changed` gate then resolves `unknown` and
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


# The one dot-directory this tool asks a repository for, organized inside the
# way `.github/`, `.vscode/` and `.circleci/` are. Any script a `verifier`
# gate names lives under `verifiers/`, a sibling of whichever guardrail shape
# is in use rather than something inside it.
#
# Two shapes, and neither is the older one: `guardrails.yml` when one file
# says it all, `guardrails/` when it is worth splitting by concern. A repo can
# have both, which is what an ordinary "this got long, split the rest out"
# looks like partway through; the file composes first.
GUARDRAIL_FILE = ".otari/guardrails.yml"
GUARDRAIL_DIR = ".otari/guardrails"

# Both spellings, because the one a file happens to carry is not worth a
# silently ignored guardrail.
_GUARDRAIL_SUFFIXES = frozenset({".yml", ".yaml"})

# Where a guardrail lived before it moved under `.otari/`. Nothing reads it,
# and the only thing this name is for is saying so: a repo still carrying one
# has no guardrail this build can find, so every gate in it, `required` ones
# included, has stopped being enforced. That is the one state worth a message
# on every hook event, because it is silent otherwise and looks exactly like a
# repo that passes every check.
_MOVED_GUARDRAIL_FILE = ".otari-guardrails.yml"


def _hook_not_enforcing(message: str) -> None:
    """Report, visibly, that this event enforced nothing, and leave the turn unblocked.

    stderr alone will not do it. Claude Code surfaces a non-blocking hook's
    stderr in its own debug log and nowhere else, never in the transcript and
    never to the model (the advisory-warning tail at the end of `hook` records
    the same finding), so a guardrail that has quietly stopped running would
    announce itself only to whoever thinks to turn on debug output. That is
    the one state this must not be quiet about, because a session with no
    enforcement looks exactly like a session that passed every check.

    Both channels, not one: stdout's `systemMessage` is what a person and the
    model actually see, and the stderr line is what a CI log or `--debug`
    transcript keeps.
    """
    click.echo(f"otari hook: {message}", err=True)
    click.echo(json.dumps({"systemMessage": f"otari hook: {message}"}))


def _guardrail_moved_notice(root: Path) -> str | None:
    """The "your guardrail is not running" line for a repo still on the old path, or None."""
    if not (root / _MOVED_GUARDRAIL_FILE).is_file():
        return None
    return (
        f"{_MOVED_GUARDRAIL_FILE} is not read any more and no gate in it is being enforced. "
        f"Move it to {GUARDRAIL_FILE}, or split it into {GUARDRAIL_DIR}/."
    )


def _hook_discover_guardrail_files(root: Path) -> list[Path]:
    """Every guardrail file this repo composes, in the order they compose.

    `.otari/guardrails.yml` first where it exists, then every `.yml`/`.yaml`
    under `.otari/guardrails/`, ordered by repo-relative path. The file first
    is the least surprising for a repo splitting a guardrail it already had,
    and the order is only ever a tiebreak: `priority` on a gate is what
    decides which judge and verifier gates survive a capped run
    (`domain.types.by_priority`), so nothing here is load-bearing beyond
    being stable.

    The scan is recursive, so gates can be grouped by concern
    (`architecture/repository-pattern.yml`). Nothing has to be excluded from
    it: a verifier script lives under `.otari/verifiers/`, outside the scanned
    root entirely, so there is no non-guardrail file down there to mistake for
    one. `Path.glob` does not descend through a symlinked directory, so the
    scan cannot be walked out of the repo either.
    """
    files: list[Path] = []
    single = root / GUARDRAIL_FILE
    if single.is_file():
        files.append(single)
    directory = root / GUARDRAIL_DIR
    if directory.is_dir():
        files.extend(
            sorted(
                (path for path in directory.glob("**/*") if path.suffix in _GUARDRAIL_SUFFIXES and path.is_file()),
                key=lambda path: path.relative_to(root).as_posix(),
            )
        )
    return files


def _composed_guardrail_id(files: list[Path], root: Path) -> str:
    """A name for the composed set, for a report that has to call it something.

    Where it came from, not an identity assembled out of the parts: each file
    keeps its own `policy.id`, and a composed set has no declared one. Which
    file a given gate came from travels per gate instead
    (`PolicySpec.gate_sources`), which is what a reader actually needs.

    One file is named as itself, since there is a real file to point at.
    """
    if len(files) == 1:
        return _guardrails_relative_to(files[0], root) or files[0].name
    if files and files[0] == root / GUARDRAIL_FILE:
        return f"{GUARDRAIL_FILE} + {GUARDRAIL_DIR}/"
    return GUARDRAIL_DIR


class GuardrailReadError(Exception):
    """A discovered guardrail file could not be read. The message names it.

    Its own type so a caller catches it beside `PolicyError` and phrases one
    outcome for "this repo's guardrail could not be loaded", whether the file
    was unreadable or its contents unparseable.
    """


def _read_guardrail_files(files: list[Path], root: Path) -> list[PolicyFile]:
    """Read every discovered file, naming the one that fails.

    `is_file()` during discovery does not guarantee the read that follows
    succeeds (a race, a permissions change, a non-UTF-8 file), and
    `UnicodeDecodeError` does not name the file it was reading, so the path
    is put into the message here rather than left to the caller to guess.
    """
    sources: list[PolicyFile] = []
    for path in files:
        try:
            name = path.relative_to(root).as_posix()
        except ValueError:
            # `otari guardrails validate --guardrail-file` may name a file
            # outside the repo, which is how a snippet is checked before being
            # dropped in; it has no repo-relative spelling to report it by.
            name = path.as_posix()
        try:
            body = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise GuardrailReadError(f"could not read {name} ({exc})") from exc
        sources.append(PolicyFile(name=name, body=body))
    return sources


def _compose_guardrail(sources: list[PolicyFile], policy_id: str) -> PolicySpec:
    """Parse one guardrail file, or compose several, into a single spec.

    One file goes through `parse_policy` rather than a one-element
    composition, so a repo with a single guardrail keeps reporting that
    file's own `policy.id` and gets no per-gate source it has no use for.
    """
    if len(sources) == 1:
        return parse_policy(sources[0].body, source=sources[0].name)
    return compose_policy(sources, policy_id=policy_id)


def _hook_guardrail_spec(root: Path) -> PolicySpec | None:
    """This repo's composed guardrail, or None when it has none, cannot read it, or cannot parse it.

    For a caller that only wants to look at the gates and has nothing useful
    to say about a broken guardrail, which is what `otari hook setup` and
    `_guardrail_allows_bash` both want. A caller that has to report the
    failure reads the pieces itself.
    """
    files = _hook_discover_guardrail_files(root)
    if not files:
        return None
    try:
        return _compose_guardrail(_read_guardrail_files(files, root), _composed_guardrail_id(files, root))
    except (GuardrailReadError, PolicyError):
        return None


def _merged_guardrail_yaml(sources: list[PolicyFile], policy_id: str) -> str:
    """One YAML document carrying every composed file's gates, for the opt-in remote mode.

    `POST /hooks/check` takes one policy body per request, so a composed
    guardrail is merged back into one document here rather than changing that
    contract for a mode most callers never turn on. Nothing is lost by it: the
    gates are the already-validated mappings from each file, and which file a
    gate came from is matched back from `PolicySpec.gate_sources`, which this
    process holds either way.

    A single-file guardrail is sent exactly as it sits on disk, comments and
    all, so the common case puts nothing on the wire that was not written by
    hand.
    """
    if len(sources) == 1:
        return sources[0].body
    documents = [yaml.safe_load(source.body) for source in sources]
    merged = {
        "schema_version": documents[0]["schema_version"],
        "policy": {"id": policy_id, "description": f"Composed from {len(sources)} guardrail files."},
        "gates": [gate for document in documents for gate in document["gates"]],
    }
    return yaml.safe_dump(merged, sort_keys=False)


def _hook_collect_changed_paths(repo_root: Path) -> list[str] | None:
    """Evidence for a `path` gate on a Stop event: what Git sees changed.

    Claude Code's Stop payload carries no file list of its own (unlike
    PreToolUse, whose tool_input already names a target), so a Stop-time
    path check has nothing to evaluate unless something goes and
    finds out what changed. Git status is that something: harness-agnostic
    (the same command regardless of which tool wrote the change, unlike
    parsing Claude Code's own transcript format) and ground truth for the
    working tree, including a change a `Bash` call made that no tool_input
    ever named. Specific to path: a future gate type collects its
    own evidence in its own way, not through this function.
    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, explicit cwd
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",  # not the platform locale default, which is not always UTF-8
            errors="replace",  # a pathologically-named file's bytes need not be valid UTF-8 either
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
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
    """Evidence for a `command`/`command_if_changed` gate on a Stop event.

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
    `claude_code_import.py`'s tolerance of the same file format.
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


# Codex's own equivalent of _PRETOOLUSE_DENIAL_MARKERS: no confirmed wrapper
# string for a PreToolUse-denied call has been observed in a real Codex
# transcript, so _hook_collect_codex_transcript_commands makes no attempt to
# exclude one. That is the same safer direction Claude Code's own denial
# handling argues for: keeping a denied command in evidence costs an
# occasional false "ran", never a missed "ran".
_CODEX_COMMAND_TOOL_NAMES = frozenset({"Bash", "shell", "local_shell", "exec_command"})


def _hook_collect_codex_transcript_commands(transcript_path: Path) -> list[str] | None:
    """Evidence for a `command`/`command_if_changed` gate on a Codex Stop event.

    Codex's own rollout file (its `transcript_path`) is a JSONL log of
    `response_item` records, a different shape from Claude Code's own
    Message-API transcript that `_hook_collect_transcript_commands` reads. A
    classic shell call appears as a `function_call` item whose `arguments` is
    a JSON-encoded string carrying a `command` field (a string, or an argv
    list joined with spaces here); a turn run through Code Mode instead wraps
    any number of shell/apply_patch calls in one `custom_tool_call`
    (`name: "exec"`) whose `input` is the raw JavaScript that issued them.
    That JS text is kept whole as "the command", the same choice
    `_HOOK_COMMAND_TOOL_FIELDS_BY_HARNESS` makes for a live PreToolUse call
    and for the same reason: a forbidden phrase still matches wherever it
    appears in it, with no per-argument parsing to get wrong.

    Returns None only when the transcript itself cannot be read, the same
    sentinel `_hook_collect_transcript_commands` uses.
    """
    try:
        lines = transcript_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    commands: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("type") != "response_item":
            continue
        item = record.get("payload")
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "custom_tool_call" and item.get("name") == "exec":
            text = item.get("input")
            if isinstance(text, str) and text:
                commands.append(text)
        elif item_type == "function_call" and item.get("name") in _CODEX_COMMAND_TOOL_NAMES:
            raw_arguments = item.get("arguments")
            # isinstance first, not a bare json.loads(... or "{}"): "arguments" is
            # documented as a JSON-encoded string, but a malformed record or a
            # future Codex shape carrying it pre-parsed (a dict/list) would
            # otherwise reach json.loads and raise TypeError, which nothing here
            # catches, crashing this whole Stop-event invocation instead of
            # skipping the one record, the same fail-open contract every other
            # per-record parse in this function already keeps.
            if not isinstance(raw_arguments, str):
                continue
            try:
                arguments = json.loads(raw_arguments)
            except ValueError:
                continue
            if not isinstance(arguments, dict):
                continue
            command = arguments.get("command")
            if isinstance(command, list):
                command = " ".join(str(part) for part in command)
            if isinstance(command, str) and command:
                commands.append(command)
    return commands


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

# Each judge gate costs one model invocation, unlike the other gate types
# (near-instant pattern matching), so an unbounded gate count means unbounded
# resource use on a single Stop event: _hook_collect_judge_verdicts bounds
# concurrency to _HOOK_GATE_MAX_WORKERS workers (see below), so N gates over
# that count still queue in batches of the timeout above. Capped, with a
# visible truncation message, the same "never let something scale unbounded
# and silently" rule _bound_commands_for_submission and the evaluator's own
# work-estimate budgets (otari_agent.domain.check) already follow.
# Evaluated in declaration order, so the same gates run first every time
# rather than an arbitrary subset.
_HOOK_JUDGE_MAX_GATES_PER_RUN = 5

# Shared by _hook_collect_judge_verdicts and _hook_collect_check_verdicts: how
# many of one run's applicable gates that function invokes at once. Independent
# of either gate type's own per-run count cap above/below: those bound how many
# gates a policy may apply at all (verifier's own is 20), this bounds how
# many of that count run at the same time, so one Stop event does not fork 20
# subprocesses simultaneously.
_HOOK_GATE_MAX_WORKERS = 8

# A per-call cap does not bound the total: 5 gates at up to 300s each, each
# with its own possible retry (_HOOK_JUDGE_PROMPT_TOO_LONG_MARKER), is up to
# 3,000s of judge calls alone. Claude Code's own command-hook timeout
# defaults to 600s, after which it kills the hook and discards its output
# entirely (see the hooks reference) -- meaning the policy check itself never
# gets run at all, and every gate in the policy, mechanical and required
# ones included, goes unevaluated for that Stop event, not just the slow
# judge gates. This is a *total* elapsed-time budget shared across every
# judge gate and retry in one run (_hook_collect_judge_verdicts computes one
# deadline before its gate loop, not one budget per gate), leaving real
# margin under the 600s default for the git evidence collection and the
# `run_policy_check` call (or, opted in, the /hooks/check request) that
# still has to happen afterward. A gate whose turn comes up after the
# deadline has passed reports "error" without attempting the call at all,
# the same fail-open contract a missing `claude` binary already has.
_HOOK_JUDGE_TOTAL_BUDGET_SECONDS = 480

# Haiku, not the session's own (often larger) default model: a judge call is a
# small, structured pass/fail classification over bounded text, not the kind
# of task that needs a frontier model, and every judge gate in a policy costs
# one full invocation against the caller's own subscription (see
# _hook_run_judge). Overridable per-invocation with --judge-model /
# OTARI_HOOK_JUDGE_MODEL for a rubric that genuinely needs more capability.
#
# claude only: Codex's own model catalog has no equally stable "small model"
# name to hardcode the same way (confirmed against a real account: its own
# session history names a current default of "gpt-6-astra", not any of the
# "cheap tier" ids OpenAI's own docs name elsewhere, which is exactly the
# kind of drift a hardcoded guess here would silently go stale against).
# `_hook_run_judge` leaves `--model` off the codex backend's own invocation
# entirely when neither this nor --judge-model apply, falling back to
# whatever model that account already has configured as its own default,
# rather than risk naming one Codex might reject outright.
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


def _hook_extract_codex_judge_transcript(transcript_path: Path) -> str:
    """The assistant's own text replies from a Codex rollout, for a judge gate's prompt.

    Codex's equivalent of `_hook_extract_judge_transcript`: an assistant
    reply is a `response_item` of type `message`, `role: "assistant"`, its
    own text under `content[].type == "output_text"` (`input_text` is the
    role Codex gives the other direction (developer/user turns), which
    carry no judgment about this session's own work).

    Returns "" when the transcript cannot be read at all, or carries no
    assistant text, the same as `_hook_extract_judge_transcript`.
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
        if not isinstance(record, dict) or record.get("type") != "response_item":
            continue
        item = record.get("payload")
        if not isinstance(item, dict) or item.get("type") != "message" or item.get("role") != "assistant":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        texts.extend(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "output_text" and isinstance(block.get("text"), str)
        )
    return "\n".join(texts)


def _hook_collect_diff(repo_root: Path) -> str | None:
    """The working tree's own diff against HEAD, for a judge gate's prompt.

    Tracked changes only (`git diff HEAD`): a new, untracked file's content is
    a known gap in this first iteration, not a silent one, since
    `_hook_collect_changed_paths` already reports its path in `changed_paths`
    even though this diff carries none of its content. Returns None only when
    Git itself could not answer (no HEAD yet, not a repository, a timeout, or
    `git` itself missing), mirroring `_hook_collect_changed_paths`'s own
    fail-open sentinel: a diff collection failure must degrade this one
    judge gate's own evidence, never crash `otari hook` and take every other
    gate in the policy, mechanical and required ones included, down with it
    before the request ever reaches the Hook Server (confirmed: an uncaught
    `subprocess.run` exception here does exactly that, exiting nonzero
    without ever calling `httpx.post`).

    `errors="replace"`: `git diff` emits a tracked file's own content bytes,
    which are not necessarily valid UTF-8 (a Latin-1-encoded tracked file, a
    binary blob committed by mistake, ...); `encoding="utf-8"` alone decodes
    strictly and raises `UnicodeDecodeError` from inside `subprocess.run`
    itself on the first non-UTF-8 byte (confirmed against a real repo with
    such a file), which is not one of the exceptions below and would
    otherwise still crash this command outright.
    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, explicit cwd
            ["git", "diff", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
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
    its own guardrail could live, the same reasoning
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


# Guards _hook_log_judge_call's own read-modify-write (open, write, close)
# against interleaving: _hook_collect_judge_verdicts invokes multiple gates'
# judge calls concurrently (_HOOK_GATE_MAX_WORKERS), each logging through
# this same function from its own worker thread. A single write() of one
# short line happens to be atomic on a POSIX append-mode fd for anything
# under the platform's own pipe-buffer size, but that is an OS-level
# coincidence this module should not depend on, and does not hold the same
# way on every platform this codebase supports (Windows included).
_HOOK_JUDGE_LOG_LOCK = threading.Lock()


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
        with _HOOK_JUDGE_LOG_LOCK, log_path.open("a", encoding="utf-8") as log_file:
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


def _hook_run_judge_subprocess(argv: list[str], prompt: str, *, deadline: float, label: str) -> tuple[str, str]:
    """Shared tail of every judge-CLI backend's own invocation: run `argv`, parse its
    stdout as the one JSON verdict object the judge prompt demands; return (outcome, reasoning).

    Both `_hook_call_claude_p` and `_hook_call_codex_exec` build their own
    `argv` (each backend's own flags are backend-specific: see each
    function's own docstring for why) and hand it here for everything after
    that: launching it, bounding it to what is left of the shared
    `deadline`, and turning its stdout into a verdict. `label` (`"claude
    -p"`/`"codex exec"`) names the backend in every message this produces,
    the only difference in what each backend's own error/success text reads.

    outcome is always one of "pass"/"fail"/"error": a nonzero exit, a
    timeout, or output that is not the single JSON object the prompt demands
    are all "error", carrying the failure detail as reasoning rather than
    raising.

    Runs with `cwd` set to `_hook_judge_workdir()`, never the repo being
    judged: see that function's own docstring for why (a real recursive
    incident) and why that is an isolated directory rather than a
    hooks-disabling flag.

    `deadline` (a `time.monotonic()` timestamp, see
    `_HOOK_JUDGE_TOTAL_BUDGET_SECONDS`) is shared across every gate and retry
    in one run, not a fresh budget per call: already past it, this returns
    "error" without ever touching `subprocess`; still short of it, the
    subprocess timeout is capped to whatever is left, never more than
    `_HOOK_JUDGE_TIMEOUT_SECONDS`.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return "error", "judge time budget exhausted before this call could start"

    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, resolved executable path
            argv,
            input=prompt,
            cwd=_hook_judge_workdir(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=min(_HOOK_JUDGE_TIMEOUT_SECONDS, remaining),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "error", f"{label} did not respond within {min(_HOOK_JUDGE_TIMEOUT_SECONDS, remaining):.0f}s"
    except (OSError, ValueError) as exc:
        # OSError: `_hook_judge_workdir()`'s own `mkdir` (permissions, disk
        # full) or the subprocess launch itself (the binary disappearing
        # between `shutil.which` and this call). ValueError: an embedded NUL
        # byte, which a diff or transcript can carry (confirmed: `subprocess`
        # raises "embedded null byte" for one in an argv element, the reason
        # the prompt goes over stdin above rather than as a trailing
        # argument). Both used to propagate uncaught, exiting `otari hook`
        # before it ever reached `httpx.post` and skipping every other gate
        # in the policy, mechanical and required ones included.
        return "error", f"could not run {label} ({exc})"

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return "error", f"{label} exited {result.returncode}: {detail[:500]}"

    try:
        verdict = json.loads(_hook_strip_judge_code_fence(result.stdout))
    except ValueError:
        return "error", f"{label} did not return valid JSON: {result.stdout[:500]!r}"

    outcome = verdict.get("outcome") if isinstance(verdict, dict) else None
    reasoning = verdict.get("reasoning") if isinstance(verdict, dict) else None
    if outcome not in ("pass", "fail") or not isinstance(reasoning, str):
        return "error", f"{label} returned an unrecognized verdict shape: {result.stdout[:500]!r}"

    return outcome, reasoning[:_HOOK_MAX_JUDGE_REASONING_LENGTH]


def _hook_call_claude_p(claude_path: str, model: str, prompt: str, *, deadline: float) -> tuple[str, str]:
    """One `claude -p --model <model>` invocation; return (outcome, reasoning).

    `claude -p`'s own "prompt is too long" rejection exits nonzero with the
    message on stdout, not stderr (confirmed against a real call), which is
    why `_hook_run_judge_subprocess` falls back to stdout for its own error
    detail when stderr is empty.

    `--tools ""` and `--strict-mcp-config`: this call's own prompt embeds the
    diff and transcript verbatim, both attacker-influenceable (a crafted diff
    or transcript could talk the model into more than a verdict; see the
    `judge` gate type's own docstring on this), and it never needs a tool to
    do its one job (read a prompt, emit one JSON object). `--tools ""`
    disables every built-in tool; `--strict-mcp-config` with no `--mcp-config`
    means no MCP server loads either, including one configured for the
    repo being judged. Confirmed this still produces a normal verdict (and,
    since it skips loading tool/MCP definitions into the system prompt,
    measured cheaper than the same call without these flags).
    """
    argv = [claude_path, "--model", model, "--tools", "", "--strict-mcp-config", "-p"]
    return _hook_run_judge_subprocess(argv, prompt, deadline=deadline, label="claude -p")


def _hook_call_codex_exec(codex_path: str, model: str | None, prompt: str, *, deadline: float) -> tuple[str, str]:
    """One `codex exec -` invocation; return (outcome, reasoning).

    `model` is `None` when `--judge-model`/`OTARI_HOOK_JUDGE_MODEL` named
    none specifically (see
    `_HOOK_JUDGE_DEFAULT_MODEL`'s own comment on why this backend gets no
    hardcoded "small model" default the way `claude` does): `--model` is then
    left off the invocation entirely, letting Codex fall back to whatever
    model this account already has configured as its own default, rather
    than risk naming one this build/account might reject outright.

    Codex's own non-interactive one-shot mode: the prompt goes over stdin
    (`-` in place of a positional prompt argument, `codex exec`'s own way of
    reading one), matching `_hook_call_claude_p`'s own choice for the same
    reason (a diff or transcript can carry an embedded NUL byte, which
    `subprocess` rejects in an argv element but not in piped input).

    `--sandbox read-only` and `--ask-for-approval never` keep this call from
    taking any action even if the model attempts one: this call's own prompt
    embeds the diff and transcript verbatim, both attacker-influenceable (see
    JudgeGate's own docstring on this), and Codex documents no flag to drop
    tool/MCP definitions from the prompt entirely the way `_hook_call_claude_p`'s
    `--tools ""`/`--strict-mcp-config` do, so this only bounds what an
    attempted tool call could *do*, not whether the model attempts one; the
    isolated `_hook_judge_workdir()` this runs in already limits what a
    read-only sandboxed attempt could see either way. `--skip-git-repo-check`
    because that workdir is a plain directory, not a repository, and
    `--ephemeral` so this one-shot call leaves no rollout file behind for a
    future Stop event's own transcript scan to mistake for real session
    evidence.

    outcome is always one of "pass"/"fail"/"error", the same contract
    `_hook_call_claude_p` returns: `codex exec` prints only the final agent
    message to stdout, progress to stderr, matched here by parsing that
    stdout as the single JSON object the prompt demands.

    No equivalent to `_hook_call_claude_p`'s "prompt is too long" retry:
    Codex's own rejection wording for an oversize prompt has not been
    confirmed against a real call, so `_hook_run_judge` never applies that
    retry to this backend rather than match a marker string that might never
    fire.
    """
    argv = [codex_path, "exec", "-"]
    if model is not None:
        argv += ["--model", model]
    argv += [
        "--sandbox",
        "read-only",
        "--ask-for-approval",
        "never",
        "--skip-git-repo-check",
        "--ephemeral",
        "--color",
        "never",
    ]
    return _hook_run_judge_subprocess(argv, prompt, deadline=deadline, label="codex exec")


# Binary name `shutil.which` resolves for each judge_cli backend.
# _hook_run_judge's own inline dispatch (not a dict of the two caller
# functions: their `model` parameter is optional for codex, required for
# claude, and a dict's value type would otherwise have to widen to the union
# of both, losing the distinction a type checker could otherwise hold onto)
# picks which one to call.
_JUDGE_CLI_BINARY_NAMES = {"claude": "claude", "codex": "codex"}

# Which judge_cli backend a gate gets when neither it nor --judge-cli names
# one: whichever CLI the harness actually invoking this hook run is itself
# built on. See _hook_collect_judge_verdicts for the full precedence order.
_JUDGE_CLI_DEFAULT_BY_HARNESS = {"claude-code": ("claude",), "codex": ("codex",)}


def _hook_resolve_judge_cli(candidates: tuple[str, ...]) -> tuple[str, str] | None:
    """First of `candidates` (in that order) whose own binary is found on PATH; None if none are."""
    for name in candidates:
        binary = shutil.which(_JUDGE_CLI_BINARY_NAMES[name])
        if binary:
            return name, binary
    return None


def _parse_judge_cli(ctx: click.Context, param: click.Parameter, value: str | None) -> tuple[str, ...] | None:
    """Parse `--judge-cli`/`OTARI_HOOK_JUDGE_CLI`: a comma-separated, ordered judge_cli override.

    None (unset) means "no session-wide override": a gate's own `judge_cli`
    still wins over it either way, and a gate naming none of its own falls
    back to the invoking harness's own default (`_JUDGE_CLI_DEFAULT_BY_HARNESS`).
    """
    if value is None:
        return None
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    unsupported = [name for name in names if name not in _JUDGE_CLI_BINARY_NAMES]
    if not names or unsupported:
        raise click.BadParameter(
            f"must be a comma-separated list of: {', '.join(sorted(_JUDGE_CLI_BINARY_NAMES))} (got {value!r})."
        )
    return names


def _hook_run_judge(
    rubric: str,
    diff: str,
    transcript: str,
    *,
    judge_cli: tuple[str, ...],
    model: str | None,
    deadline: float,
    dry_run: bool = False,
) -> tuple[str, str]:
    """Invoke the first available `judge_cli` backend for one judge gate's rubric; return (outcome, reasoning).

    Otari itself never calls a model (see JudgeGate's own docstring); this is
    that call, made locally against whatever subscription the resolved CLI
    itself is signed into, not billed through Otari.

    `model` is the caller's own explicit choice (`--judge-model`/
    `OTARI_HOOK_JUDGE_MODEL`), or `None` for "no explicit choice, use this
    backend's own default": `claude` gets `_HOOK_JUDGE_DEFAULT_MODEL`
    (Haiku); `codex` gets none at all, `_hook_call_codex_exec` then omitting
    `--model` entirely (see `_HOOK_JUDGE_DEFAULT_MODEL`'s own comment on why
    the two are not symmetric here). Resolved after `judge_cli`, not before:
    which default applies depends on which backend actually gets picked.

    `judge_cli` is the already-resolved preference order for this one gate
    (a gate's own `judge_cli`, else `--judge-cli`/`OTARI_HOOK_JUDGE_CLI`, else
    the invoking harness's own default; see `_hook_collect_judge_verdicts`).
    `_hook_resolve_judge_cli` picks the first entry whose own binary is on
    PATH; this is what lets a gate listing `judge_cli: [claude, codex]` still
    get a verdict on a machine with only one of the two installed, and what
    makes a bare, single-entry list behave exactly as a hardcoded "claude"
    always did before this existed.

    `dry_run` skips the real call entirely, before ever touching `shutil.which`
    or `subprocess`: the wire contract has no fourth outcome to spell "this
    was never really run", so it reports the same `"error"` a real failed
    call would, with a `reasoning` that says so explicitly and estimates the
    prompt's size, never `"pass"`/`"fail"`, which would misrepresent a
    verdict nothing actually produced. `deadline` (see
    `_HOOK_JUDGE_TOTAL_BUDGET_SECONDS`) is passed through to both attempts
    below unchanged either way, since it is one shared budget across the
    whole run, not a fresh one per call.

    A "prompt is too long" rejection retries once with `transcript` dropped
    entirely: the transcript is supplementary "why" context for a judge
    rubric, the diff is the primary evidence, so a diff-only retry is a
    strictly better fallback than reporting no verdict at all. Only for that
    specific rejection, only once, and only against the `claude` backend
    (`_hook_call_codex_exec`'s own docstring says why Codex gets no
    equivalent yet): any other failure, or a rejection that persists with no
    transcript left to drop, reports "error" as it always has.
    """
    if dry_run:
        prompt = _hook_build_judge_prompt(rubric, diff, transcript)
        estimated_tokens = _hook_estimate_tokens(prompt)
        return (
            "error",
            f"--judge-dry-run: real {'/'.join(judge_cli)} call skipped; prompt would have been "
            f"{len(prompt):,} chars (~{estimated_tokens:,} tokens estimated at "
            f"~{_HOOK_JUDGE_CHARS_PER_TOKEN_ESTIMATE} chars/token).",
        )

    resolved = _hook_resolve_judge_cli(judge_cli)
    if resolved is None:
        tried = ", ".join(_JUDGE_CLI_BINARY_NAMES[name] for name in judge_cli)
        return "error", f"none of the configured judge CLI(s) were found on PATH: {tried}"
    backend, binary_path = resolved

    def call(prompt_text: str) -> tuple[str, str]:
        if backend == "claude":
            return _hook_call_claude_p(
                binary_path, model if model is not None else _HOOK_JUDGE_DEFAULT_MODEL, prompt_text, deadline=deadline
            )
        return _hook_call_codex_exec(binary_path, model, prompt_text, deadline=deadline)

    outcome, reasoning = call(_hook_build_judge_prompt(rubric, diff, transcript))
    if (
        backend == "claude"
        and outcome == "error"
        and transcript
        and _HOOK_JUDGE_PROMPT_TOO_LONG_MARKER in reasoning.lower()
    ):
        outcome, reasoning = call(_hook_build_judge_prompt(rubric, diff, ""))
    return outcome, reasoning


def _hook_collect_judge_verdicts(
    spec: PolicySpec,
    repo_root: Path,
    transcript_path: str | None,
    changed_paths: list[str],
    *,
    judge_model: str | None,
    judge_dry_run: bool = False,
    harness: str = "claude-code",
    judge_cli_override: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    """Run every applicable judge gate in the local policy, one model-CLI call each,
    up to `_HOOK_GATE_MAX_WORKERS` of them concurrently.

    Reads `rubric`/`when_changed` off the already-parsed policy the caller
    evaluates below, so there is one parse of a guardrail per hook event and
    no way for the gates judged here to differ from the gates checked there.

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
    itself skips the real model-CLI call. This is what makes the resulting
    log a real count of how often the model would have been invoked, not a
    guess: everything up to the call itself runs exactly as it would for real.

    `harness` picks which transcript format `transcript_path` is read as
    (Claude Code's Message-API transcript vs. Codex's rollout JSONL). It also
    supplies the *default* judge_cli order (`_JUDGE_CLI_DEFAULT_BY_HARNESS`)
    for a gate that names none of its own: precedence, most specific first,
    is a gate's own `JudgeGate.judge_cli`, then this call's own
    `judge_cli_override` (`hook`'s own `--judge-cli`/`OTARI_HOOK_JUDGE_CLI`),
    then that harness default. A gate or override naming more than one CLI is
    an ordered fallback list, resolved by `_hook_resolve_judge_cli`: the first
    entry whose own binary is on PATH is what actually gets invoked.

    Gates run concurrently (a `ThreadPoolExecutor`, not `asyncio`: each
    worker's own time is spent blocked inside `subprocess.run`, ordinary
    blocking I/O a thread waits on fine, not a coroutine-friendly awaitable),
    capped at `_HOOK_GATE_MAX_WORKERS` at once. `_hook_log_judge_call` is
    safe to call from every worker thread (`_HOOK_JUDGE_LOG_LOCK`); nothing
    else a worker touches is shared mutable state (`diff`/`transcript`/
    `deadline` are read-only from every worker's own perspective, and
    `_hook_judge_workdir()` is a directory each `claude -p`/`codex exec`
    subprocess uses independently, not a file the workers write themselves).
    The returned list keeps the order the gates run in:
    `ThreadPoolExecutor.map` yields results in the order its inputs were
    given, not completion order.
    """
    changed_paths_tuple = tuple(changed_paths)
    judge_gates = by_priority(
        [
            gate
            for gate in spec.gates
            if isinstance(gate, JudgeGate)
            and (not gate.when_changed or matched_changed_paths(gate.when_changed, changed_paths_tuple))
        ]
    )
    if not judge_gates:
        return []
    if len(judge_gates) > _HOOK_JUDGE_MAX_GATES_PER_RUN:
        skipped = [gate.id for gate in judge_gates[_HOOK_JUDGE_MAX_GATES_PER_RUN:]]
        click.echo(
            f"otari hook: {len(judge_gates):,} judge gates in this guardrail, over the "
            f"{_HOOK_JUDGE_MAX_GATES_PER_RUN:,} limit; skipping the lowest priority: {', '.join(skipped)}.",
            err=True,
        )
        judge_gates = judge_gates[:_HOOK_JUDGE_MAX_GATES_PER_RUN]

    # None (collection genuinely failed: no HEAD, git missing, a timeout) is
    # kept distinct from "" (collected, and there is none) until the loop
    # below: collapsing them here, as `_hook_collect_diff(repo_root) or ""`
    # used to, sends the model a diff-less prompt indistinguishable from a
    # real empty one, and a model asked to judge a change it cannot see can
    # (and, verified against a real call, does) still say "pass".
    diff = _hook_collect_diff(repo_root)
    diff_collection_failed = diff is None
    diff = diff or ""
    extract_transcript = _hook_extract_codex_judge_transcript if harness == "codex" else _hook_extract_judge_transcript
    transcript = extract_transcript(Path(transcript_path)) if transcript_path else ""
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

    # One deadline for the whole run, computed once, not a fresh budget per
    # gate: see _HOOK_JUDGE_TOTAL_BUDGET_SECONDS for why a per-call cap alone
    # does not bound the total, and why that total must stay under Claude
    # Code's own outer hook timeout.
    deadline = time.monotonic() + _HOOK_JUDGE_TOTAL_BUDGET_SECONDS

    def run_one(gate: JudgeGate) -> dict[str, str]:
        # Logged before the call, not after: a hung or killed model-CLI
        # invocation must still show up in the audit trail rather than
        # silently vanishing along with the process that would have logged
        # its outcome.
        _hook_log_judge_call(repo_root, gate.id, "invoking")
        if diff_collection_failed:
            # No model call at all: a diff this gate cannot see is not
            # evidence to judge against, and every other pre-flight failure
            # here (no configured judge_cli found on PATH, an exhausted time
            # budget) already reports "error" without one either.
            outcome, reasoning = "error", "could not collect the working tree diff"
        else:
            judge_cli = gate.judge_cli or judge_cli_override or _JUDGE_CLI_DEFAULT_BY_HARNESS.get(harness, ("claude",))
            outcome, reasoning = _hook_run_judge(
                gate.rubric,
                diff,
                transcript,
                judge_cli=judge_cli,
                model=judge_model,
                deadline=deadline,
                dry_run=judge_dry_run,
            )
        _hook_log_judge_call(repo_root, gate.id, outcome, detail=reasoning if judge_dry_run else None)
        return {"gate_id": gate.id, "outcome": outcome, "reasoning": reasoning}

    with ThreadPoolExecutor(max_workers=min(len(judge_gates), _HOOK_GATE_MAX_WORKERS)) as executor:
        return list(executor.map(run_one, judge_gates))


# A verifier script is expected to be fast and deterministic (a grep, a lint
# rule, a small local test), not a model call, so this is bounded an order of
# magnitude tighter than the judge gate's own per-call timeout
# (_HOOK_JUDGE_TIMEOUT_SECONDS). One call taking this long is already a sign
# something is wrong, not a slow-but-normal case to accommodate.
_HOOK_CHECK_TIMEOUT_SECONDS = 30

# Mirrors _HOOK_JUDGE_MAX_GATES_PER_RUN's own reasoning: each verifier
# gate costs one subprocess run, not a near-instant pattern match, so an
# unbounded gate count must not turn one Stop event into unbounded
# wall-clock.
_HOOK_CHECK_MAX_GATES_PER_RUN = 20

# One shared elapsed-time budget across every verifier gate in one run,
# the same shape _HOOK_JUDGE_TOTAL_BUDGET_SECONDS takes, scaled down for the
# same reason _HOOK_CHECK_TIMEOUT_SECONDS is: verifier scripts are expected
# to run in seconds, not minutes, and this budget still has to leave margin
# under Claude Code's own 600s Stop-hook default alongside whatever judge
# gates already claimed out of that same 600s in this run.
_HOOK_CHECK_TOTAL_BUDGET_SECONDS = 60

# Mirrors the Hook Server's own CheckVerdictRequest.detail cap
# (routes/hooks.py, _MAX_CHECK_DETAIL_LENGTH): an oversize detail otherwise
# 422s the *whole* /hooks/check request, fail-open, taking every other gate
# in the same policy down with it.
_HOOK_MAX_CHECK_DETAIL_LENGTH = 4_096


def _hook_run_check_verifier(repo_root: Path, verifier: str, *, deadline: float) -> tuple[str, str]:
    """Run one verifier gate's verifier script; return (outcome, detail).

    The exit-code contract is fixed, not something a caller or this command
    decides: 0 is "pass", 1 is "fail", anything else -- a different exit
    code, a crash, a missing or non-executable script -- is "error". This is
    what lets `enforcement: required` genuinely block for this gate type,
    unlike `judge`: the contract is reproducible, not a model's opinion.

    `verifier` is resolved against `repo_root` and, before it is ever run,
    confirmed to still resolve inside it (mirrors the same guard the
    PreToolUse edit-path branch above applies to its own target path): a
    policy naming `../../etc/passwd` or an absolute path domain.policy
    already rejects at parse time, but a relative path can still climb out
    with enough `..` segments, and running whatever that resolves to would
    be a materially different, undocumented capability, not "run a
    repo-local script".

    No sandboxing beyond that check, and no guard requiring the script to
    predate the diff under check, deliberately: see VerifierGate's own
    docstring and docs/agent-guardrails.md for why. `cwd` is the repo root, so a
    verifier that wants to inspect the working tree (`git diff`, `git
    status`, a plain file scan) can do so exactly the way a Makefile target
    or a pre-commit hook already checked into the repo would.

    Captured stdout, capped at `_HOOK_MAX_CHECK_DETAIL_LENGTH`, is the
    verdict's detail regardless of outcome; stderr is not read, since the
    wire contract has no separate slot for it and stdout is what a verifier
    author is expected to write the human-readable reason to. It is decoded
    as UTF-8 with `errors="replace"` for the same reason
    `_hook_collect_diff` does it: a verifier that echoes a tracked file's
    own bytes can emit something that is not valid UTF-8, and strict
    decoding raises `UnicodeDecodeError` from inside `subprocess` itself,
    which would escape this function and take every other gate in the
    policy down with it before the request ever reached the Hook Server.

    The verifier leads a process group of its own (`start_new_session`) so
    that the timeout reaches its descendants too. Stopping the verifier alone
    leaves anything it backgrounded running, still holding the stdout and
    stderr pipes it inherited, past both this call's cap and the whole run's
    shared budget; on a platform whose timeout cleanup reads those pipes
    (Windows) that is a hang, and everywhere it is at least an orphan the
    gate spawned and never reclaimed.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return "error", "check time budget exhausted before this verifier could run"

    resolved_root = repo_root.resolve()
    script_path = (repo_root / verifier).resolve()
    try:
        script_path.relative_to(resolved_root)
    except ValueError:
        return "error", f"verifier {verifier!r} resolves outside the repo root"

    if not script_path.is_file():
        return "error", f"verifier {verifier!r} does not exist"

    timeout = min(_HOOK_CHECK_TIMEOUT_SECONDS, remaining)
    try:
        process = subprocess.Popen(  # noqa: S603 - no shell, resolved path checked against repo_root above
            [str(script_path)],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        # A script that is not executable, has no shebang, or does not exist
        # by the time exec actually runs (a race after the is_file() check
        # above) all raise OSError here rather than letting a crashed
        # subprocess look any different from a script that genuinely ran
        # and exited nonzero.
        return "error", f"could not run verifier {verifier!r}: {exc}"

    with process:
        try:
            stdout, _stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _hook_kill_process_group(process)
            process.communicate()
            return "error", f"verifier did not respond within {timeout:.0f}s"

    detail = stdout[:_HOOK_MAX_CHECK_DETAIL_LENGTH]
    if process.returncode == 0:
        return "pass", detail
    if process.returncode == 1:
        return "fail", detail
    return "error", detail or f"verifier exited with status {process.returncode}"


def _hook_kill_process_group(process: subprocess.Popen[str]) -> None:
    """SIGKILL a timed-out verifier along with anything it left running.

    The verifier was started in a session of its own, so one `killpg` reaches
    a background child too, which is the point: `process.kill()` alone leaves
    such a child running with the captured pipes still open. SIGKILL, not
    SIGTERM, because a verifier that ignored the deadline has already had its
    chance to exit. Falls back to killing the verifier alone where there are
    no process groups (Windows) or where the group is already gone.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        return
    except (AttributeError, OSError):
        pass
    try:
        process.kill()
    except OSError:
        pass


def _hook_collect_check_verdicts(
    spec: PolicySpec,
    repo_root: Path,
    changed_paths: list[str],
) -> list[dict[str, str]]:
    """Run every applicable verifier gate's verifier locally; return check_results.

    Structured exactly like `_hook_collect_judge_verdicts`, and reads its
    gates off the same already-parsed policy the caller evaluates below.

    A gate with `when_changed` is skipped locally, before ever running its
    verifier, when none of `changed_paths` matches its globs
    (`domain.evaluators.matched_changed_paths`, the same grammar
    `evaluate_verifier`'s own applicability check uses server-side): the
    same local optimization `_hook_collect_judge_verdicts` already applies to
    a judge gate's own `when_changed`. A gate with no `when_changed` at all
    keeps its unconditional, every-Stop-event behavior.

    Verifiers run concurrently too, the same `ThreadPoolExecutor`/
    `_HOOK_GATE_MAX_WORKERS` shape `_hook_collect_judge_verdicts` uses and for
    the same reason (see that function's own docstring); `_hook_run_check_verifier`
    already runs each verifier in its own subprocess with its own timeout, so
    nothing here needs a lock the way judge gates' shared audit log does.
    This does shift a real assumption onto verifier authors, though: two or
    more `verifier` gates applicable to the same Stop event now run at
    the same time against the same working tree, not one after another, so a
    verifier that is not safe under that (one that writes to a fixed
    temporary path another verifier might also use, or that mutates the
    working tree itself rather than only reading it, e.g. `git stash`) can
    now race in a way it could not before this build. `.otari/verifiers/`
    in this repo only ever reads the tree (`git status`/`git diff`, a file
    scan), which is safe under concurrency for free; a verifier that needs to
    write should not assume it is the only one running.
    """
    changed_paths_tuple = tuple(changed_paths)
    check_gates = by_priority(
        [
            gate
            for gate in spec.gates
            if isinstance(gate, VerifierGate)
            and (not gate.when_changed or matched_changed_paths(gate.when_changed, changed_paths_tuple))
        ]
    )
    if not check_gates:
        return []
    if len(check_gates) > _HOOK_CHECK_MAX_GATES_PER_RUN:
        skipped = [gate.id for gate in check_gates[_HOOK_CHECK_MAX_GATES_PER_RUN:]]
        click.echo(
            f"otari hook: {len(check_gates):,} verifier gates in this guardrail, over the "
            f"{_HOOK_CHECK_MAX_GATES_PER_RUN:,} limit; skipping the lowest priority: {', '.join(skipped)}.",
            err=True,
        )
        check_gates = check_gates[:_HOOK_CHECK_MAX_GATES_PER_RUN]

    # One deadline for the whole run, computed once, not a fresh budget per
    # gate: see _HOOK_CHECK_TOTAL_BUDGET_SECONDS for why a per-call cap alone
    # does not bound the total.
    deadline = time.monotonic() + _HOOK_CHECK_TOTAL_BUDGET_SECONDS

    def run_one(gate: VerifierGate) -> dict[str, str]:
        outcome, detail = _hook_run_check_verifier(repo_root, gate.verifier, deadline=deadline)
        return {"gate_id": gate.id, "outcome": outcome, "detail": detail}

    with ThreadPoolExecutor(max_workers=min(len(check_gates), _HOOK_GATE_MAX_WORKERS)) as executor:
        return list(executor.map(run_one, check_gates))


@click.group(name="hook", invoke_without_command=True)
@click.option(
    "--harness",
    type=click.Choice(["claude-code", "codex"]),
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
    default=None,
    help=(
        f"Model the resolved judge CLI uses for a judge gate's model call. Defaults to "
        f"{_HOOK_JUDGE_DEFAULT_MODEL!r} when the resolved backend is claude; when it is codex, "
        "left unset (that account's own default model applies) unless given explicitly here."
    ),
)
@click.option(
    "--judge-cli",
    envvar="OTARI_HOOK_JUDGE_CLI",
    default=None,
    callback=_parse_judge_cli,
    help=(
        "Comma-separated, ordered judge-gate CLI backend(s) to try (claude, codex); the first one "
        "found on PATH is used. A gate's own judge_cli overrides this; with neither set, defaults to "
        "whichever CLI --harness itself implies."
    ),
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
    judge_model: str | None,
    judge_cli: tuple[str, ...] | None,
    judge_dry_run: bool,
) -> None:
    """Native callback entry point for a supported agent's hook protocol.

    Reads one JSON hook payload on stdin, collects the evidence that payload
    carries (a PreToolUse call's own target path, or a Stop event's Git
    status), and evaluates it against the local guardrail itself, in
    process, through `otari_agent.domain.check.run_policy_check`: no server,
    no credential, needed for this by default. `--url`/`--api-key` (or
    `OTARI_URL`/`OTARI_API_KEY`) are the opt-in exception: give either and
    this instead calls a gateway's `POST /api/v1/hooks/check` over HTTP the
    way every version of this command before local evaluation existed did,
    for whoever wants a shared/hosted gateway to be the one deciding rather
    than the machine the agent is running on. See docs/agent-guardrails.md.

    Exit code is this harness's own protocol, not otari policy check's:
    Claude Code's and Codex's PreToolUse and Stop hooks both take 0 (proceed)
    or 2 (block, stderr shown to the agent). Never blocks on a problem that is
    not a required gate failing: a missing or malformed policy, or (only in
    the opt-in remote mode) an unreachable gateway or a missing credential,
    all exit 0, with a message on stderr where there is one worth surfacing.

    `--harness` picks which payload/transcript shape is expected and which
    tool names are read as an edit vs. a command (see
    `_HOOK_COMMAND_TOOL_FIELDS_BY_HARNESS`, `_CODEX_PATCH_TOOL_NAME`); Codex's
    own Code Mode wraps shell/apply_patch calls in a JS snippet rather than
    naming one tool, and its PreToolUse dispatch does not yet cover that
    surface at all (openai/codex#23411), so a `path`/`command`
    gate scoped to `PreToolUse` will not see a Code Mode edit until upstream
    fixes that; `Stop`'s own Git-status fallback and transcript scan still do.

    A group, not a plain command, so `otari hook setup` can live alongside
    it: invoked with no subcommand (the shape every existing settings file
    already calls), it runs the callback above unchanged.
    """
    if ctx.invoked_subcommand is not None:
        return

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    event = payload.get("hook_event_name")
    repo = Path(payload.get("cwd") or Path.cwd())
    root = _hook_find_repo_root(repo)
    if root is None:
        return

    guardrail_files = _hook_discover_guardrail_files(root)
    if not guardrail_files:
        # Silence is right for a repo that never had a guardrail, and wrong for
        # one whose guardrail this build stopped finding; see
        # `_guardrail_moved_notice`.
        moved = _guardrail_moved_notice(root)
        if moved is not None:
            _hook_not_enforcing(moved)
        return
    guardrail_name = _composed_guardrail_id(guardrail_files, root)
    try:
        sources = _read_guardrail_files(guardrail_files, root)
        spec = _compose_guardrail(sources, guardrail_name)
    except (GuardrailReadError, PolicyError) as exc:
        # Fail-open, like every other collection failure in this command: a
        # guardrail this build cannot load must not exit nonzero and block the
        # turn on its own malformedness. Said visibly, though, because
        # composing several files puts this state within reach of a file
        # somebody else added rather than only the one you just edited.
        _hook_not_enforcing(f"could not load {guardrail_name} ({exc}); no gate is being enforced.")
        return

    paths: list[str] = []
    # `[]`, not None, by default: PreToolUse's edit-tool branch below leaves
    # this as `[]` on purpose, meaning "no command evidence for this call",
    # the same not_applicable-not-unknown contract every other command-less
    # event has always had. Only the Stop branch may set this to None, when
    # it collected no evidence at all rather than collecting and finding
    # nothing.
    commands: list[str] | None = []
    # "call" unless the Stop branch below really does collect the whole
    # session: this is what tells the evaluator which command-evidence gates
    # can resolve at all, rather than leaving each to guess from an empty list.
    command_scope: EvidenceScope = "call"
    # Which moment the submitted paths were read at, the counterpart to a
    # gate's own `runs`. Set in every branch below that submits a path list;
    # run_policy_check refuses a path list without one, because a gate cannot
    # otherwise tell a PreToolUse call from a Stop event on a clean tree, nor
    # a path about to be read from one about to be written.
    path_source: RunsAt | None = None
    # None, not [], by default: a PreToolUse call has neither a full diff nor
    # a finished transcript to judge against yet, and never runs
    # _hook_collect_judge_verdicts at all, so submitting None (rather than an
    # empty list, which would mean "ran judge gates, found none applicable")
    # is what resolves every judge gate not_applicable on PreToolUse instead
    # of unknown (see PolicyCheckRequest.judge_results, evaluate_judge). Only
    # the Stop branch below ever reassigns this, to a real (possibly empty)
    # list.
    judge_results: list[dict[str, str]] | None = None
    # None, not [], by default, for exactly the same reason judge_results is:
    # a PreToolUse call has no finished session for a verifier to check yet,
    # and never runs _hook_collect_check_verdicts at all, so submitting None
    # resolves every verifier gate not_applicable rather than the
    # unknown a caller that does run verifier gates but is missing one
    # gets (see PolicyCheckRequest.check_results, evaluate_verifier).
    # Only the Stop branch below ever reassigns this.
    check_results: list[dict[str, str]] | None = None
    if event == "PreToolUse":
        tool_name = payload.get("tool_name", "")
        tool_input = payload.get("tool_input") or {}
        command_fields = _HOOK_COMMAND_TOOL_FIELDS_BY_HARNESS.get(harness, _HOOK_COMMAND_TOOL_FIELDS)
        is_apply_patch = harness == "codex" and tool_name == _CODEX_PATCH_TOOL_NAME
        # A tool call is an edit, a read or a shell command, never more than
        # one, so at most one of these evidence lists is ever populated per
        # call.
        path_field = None if is_apply_patch else _HOOK_EDIT_TOOL_PATH_FIELDS.get(tool_name)
        read_field = _HOOK_READ_TOOL_PATH_FIELDS.get(tool_name)
        command_field = command_fields.get(tool_name)
        if is_apply_patch:
            # apply_patch carries no bare file_path the way Edit/Write do:
            # tool_input["command"] is the whole patch envelope, one or more
            # files named on its own header lines.
            patch_text = tool_input.get("command")
            if not patch_text:
                return
            # An apply_patch header names its target relative to the tool
            # call's own working directory, unlike Edit/Write's
            # always-absolute file_path; _hook_evidence_paths anchors both
            # shapes on `repo` and submits each target's lexical and resolved
            # spellings, one header at a time.
            resolved_paths = [
                candidate
                for patch_path in _hook_extract_patch_paths(patch_text)
                for candidate in _hook_evidence_paths(patch_path, repo, root)
            ]
            if not resolved_paths:
                return
            paths = list(dict.fromkeys(resolved_paths))
            path_source = "pre_tool_use.edit_target"
        elif path_field:
            target = tool_input.get(path_field)
            if not target:
                return
            paths = _hook_evidence_paths(target, repo, root)
            if not paths:
                return  # Outside the repo under both spellings: nothing a policy can name.
            path_source = "pre_tool_use.edit_target"
        elif read_field:
            target = tool_input.get(read_field)
            if not target:
                return
            # Collected exactly the way the edit branch above collects its
            # own target, and for a reason that bites harder here: the
            # symlinked-out-of-the-repo case _hook_evidence_paths exists to
            # close is the ordinary layout for the very file a read gate is
            # written about.
            paths = _hook_evidence_paths(target, repo, root)
            if not paths:
                return  # Outside the repo under both spellings: nothing a policy can name.
            path_source = "pre_tool_use.read_target"
        elif command_field:
            command = tool_input.get(command_field)
            if not command:
                return
            # Truncated rather than sent whole: the Hook Server rejects an
            # oversize command with a 422, and a 422 fails the *whole* check
            # open, taking every path gate in the same policy with it.
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
            path_source = "pre_tool_use.command"
        else:
            return  # A tool this harness integration does not check yet.
    elif event == "Stop":
        collected = _hook_collect_changed_paths(root)
        if collected is None:
            click.echo("otari hook: could not read Git state, not blocking.", err=True)
            return
        paths = collected
        path_source = "stop.working_tree"

        # transcript_path is the session's JSONL transcript on disk (each
        # harness's own name/format for it). Absent, or unreadable, submits
        # None rather than `[]`: `[]` means "collected, and there is none",
        # which would let a required command/command_if_changed gate
        # read a failed collection as a clean pass instead of the unresolved
        # `unknown` it actually is (see docs/agent-guardrails.md).
        transcript_path = payload.get("transcript_path")
        collect_transcript_commands = (
            _hook_collect_codex_transcript_commands if harness == "codex" else _hook_collect_transcript_commands
        )
        commands = collect_transcript_commands(Path(transcript_path)) if transcript_path else None
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
            spec,
            root,
            transcript_path,
            paths,
            judge_model=judge_model,
            judge_dry_run=judge_dry_run,
            harness=harness,
            judge_cli_override=judge_cli,
        )
        check_results = _hook_collect_check_verdicts(spec, root, paths)
    else:
        return  # An event this harness integration does not check yet.

    # No `--url`/`--api-key` (nor their envvars): the common case, and the
    # default now. Evaluate the local policy in process, the same pure
    # `run_policy_check` the Hook Server route itself calls, so nothing here
    # needs a running gateway, a credential, or the network at all.
    if url is None and api_key is None:
        try:
            check_result = check_policy(
                spec,
                paths=paths,
                commands=commands,
                path_source=path_source,
                command_scope=command_scope,
                judge_results=(
                    None
                    if judge_results is None
                    else [
                        JudgeVerdict(
                            gate_id=v["gate_id"],
                            outcome=cast(Literal["pass", "fail", "error"], v["outcome"]),
                            reasoning=v["reasoning"],
                        )
                        for v in judge_results
                    ]
                ),
                check_results=(
                    None
                    if check_results is None
                    else [
                        CheckVerdict(
                            gate_id=v["gate_id"],
                            outcome=cast(Literal["pass", "fail", "error"], v["outcome"]),
                            detail=v["detail"],
                        )
                        for v in check_results
                    ]
                ),
            )
        except PolicyCheckError as exc:
            click.echo(f"otari hook: could not evaluate {guardrail_name} ({exc}), not blocking.", err=True)
            return
        failing = [
            {
                "gate_id": gate_result.gate_id,
                "enforcement": gate_result.enforcement,
                "outcome": gate_result.outcome.value,
                "message": gate_result.message,
                "detail": gate_result.detail,
                "source": gate_result.source,
            }
            for gate_result in check_result.results
            if gate_result.outcome.value not in ("pass", "not_applicable")
        ]
        blocked = check_result.blocked
    else:
        # Explicit opt-in: check against a gateway over HTTP instead, the way
        # every version of this command before local evaluation existed did.
        # For whoever wants a shared/hosted gateway to be the one deciding,
        # or a central place data about the check could eventually land.
        import httpx

        policy_yaml = _merged_guardrail_yaml(sources, guardrail_name)
        if len(policy_yaml.encode("utf-8")) > MAX_POLICY_BYTES:
            _hook_not_enforcing(
                f"{guardrail_name} composes to more than the {MAX_POLICY_BYTES:,} bytes the Hook Server "
                "accepts in one request, so no gate is being enforced. Evaluate it locally "
                "(drop --url/--api-key) or split the check across fewer gates."
            )
            return

        try:
            settings = load_settings(config)
        except ValueError as exc:
            # An unreadable or malformed config file, or a port that is not a
            # number: a setup problem, not a required gate failing, so it falls
            # under this command's own fail-open contract.
            click.echo(f"otari hook: could not load config ({exc}), not blocking.", err=True)
            return
        # host is a bind address (0.0.0.0 is the documented default), not a
        # connect target; a client dials localhost instead.
        connect_host = "localhost" if settings.host == "0.0.0.0" else settings.host  # noqa: S104
        resolved_url = url or f"http://{connect_host}:{settings.port}"
        resolved_key = api_key or settings.master_key
        if not resolved_key:
            click.echo("otari hook: no API key or master key resolved, not blocking.", err=True)
            return

        try:
            response = httpx.post(
                f"{resolved_url.rstrip('/')}{API_ROOT}/hooks/check",
                json={
                    "policy_yaml": policy_yaml,
                    "paths": paths,
                    "commands": commands,
                    "path_source": path_source,
                    "command_scope": command_scope,
                    "judge_results": judge_results,
                    "check_results": check_results,
                },
                headers={API_KEY_HEADER: resolved_key},
                timeout=15.0,
            )
            response.raise_for_status()
            result = response.json()
            failing = [gate for gate in result["results"] if gate["outcome"] not in ("pass", "not_applicable")]
            # The route evaluated one merged document and has no file names to
            # report; they are restored here from the composition this process
            # did, so a failure reads the same whichever mode produced it.
            for gate in failing:
                gate["source"] = spec.gate_sources.get(str(gate.get("gate_id", "")))
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
    # detail carries the specific "why" behind message's generic, fixed
    # policy text (a judge gate's own model reasoning, a path gate's
    # matched paths, ...); without it, every gate of the same id shows the
    # exact same static line no matter what a judge model actually found
    # (confirmed: a real verdict naming "src/module.py:42" surfaced only the
    # configured message, never that). Bounded the same way the error
    # details elsewhere in this command are, at 500 characters: `reasoning`
    # itself can run up to _HOOK_MAX_JUDGE_REASONING_LENGTH (4,096), too long
    # for one stderr line.
    # `source` is set only where the guardrail was composed from more than
    # one file, which is exactly when naming the file saves a reader from
    # hunting for which one to edit. A single-file repo would only be told
    # what it already knows.
    summary = "\n".join(
        f"  [{'x' if gate.get('enforcement') == 'required' else '!'}] "
        f"{gate.get('gate_id', '?')}: {gate.get('message', '(no message)')}"
        + (f" ({str(gate['detail'])[:500]})" if gate.get("detail") else "")
        + (f" [{gate['source']}]" if gate.get("source") else "")
        for gate in failing
    )
    if blocked:
        # stop_hook_active is the harness's own signal that this Stop is
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
        #
        # Codex's own retry cap (if it has a fixed one) has not been
        # confirmed against a real session the way Claude Code's has, so its
        # note names no specific number rather than guessing one.
        if payload.get("stop_hook_active"):
            repeat_note = (
                "\n  (already blocked once this turn; Claude Code overrides a Stop hook after 8 "
                "consecutive blocks, so fix this now or say why you cannot.)"
                if harness == "claude-code"
                else "\n  (already blocked once this turn; fix this now or say why you cannot.)"
            )
        else:
            repeat_note = ""
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
    """Absolute path to the otari binary this process was started through.

    A hook subprocess (Claude Code's, Codex's) does not inherit an activated
    shell's PATH, so the generated entry names the binary absolutely. argv[0]
    is preferred and left unresolved: under Homebrew it is the stable
    /opt/homebrew/bin/otari link, and resolving it would pin the entry to a
    versioned Cellar path the next upgrade breaks. When argv[0] is not a file
    named otari (python -m, a test runner), fall back to the console script
    beside the running interpreter, which is where a venv install puts it.
    """
    if sys.argv and sys.argv[0]:
        invoked = Path(os.path.abspath(sys.argv[0]))
        if invoked.name == "otari" and invoked.is_file():
            return str(invoked)
    return str(Path(sys.executable).with_name("otari"))


def _policy_checks_reads(spec: PolicySpec) -> bool:
    """Whether any path gate here declares `pre_tool_use.read_target`.

    The one `runs` value that decides whether a tool matcher needs a group,
    because it is the only one naming a tool no other gate type reaches: an
    edit gate and a read gate are the same type and differ only here.
    """
    return any(isinstance(gate, PathGate) and "pre_tool_use.read_target" in gate.runs for gate in spec.gates)


class _MatcherNeeds(NamedTuple):
    """Which PreToolUse matcher groups this guardrail gives work to, beyond the edit tools.

    The edit group is unconditional (a path gate at either `pre_tool_use`
    source is the common case, and `stop.working_tree` needs no matcher at
    all), so only these two are decided per guardrail.
    """

    bash: bool
    read: bool


def _guardrail_matcher_needs(spec: PolicySpec | None) -> _MatcherNeeds:
    """Which matcher groups this repo's composed guardrail earns.

    A missing or unparseable guardrail (``None``) earns neither, the narrowest
    matcher: setup cannot know what a broken guardrail would have wanted.

    `bash` deliberately does not consult any gate's `runs`, which looks like it
    should matter and does not: no `runs` value a path gate can name is
    collectable from a Bash call. `pre_tool_use.edit_target` and
    `pre_tool_use.read_target` both need a tool that declares a path, and a
    Bash call declares none; `stop.working_tree` is not readable at PreToolUse
    at all. Only a command gate gives the Bash matcher anything to do. That
    changes the day a source for shell write or read targets exists, and not
    before.

    `read` is the opposite: it turns on `runs` alone, since a read gate and an
    edit gate are the same `path` type and nothing else tells them apart.
    """
    if spec is None:
        return _MatcherNeeds(bash=False, read=False)
    return _MatcherNeeds(
        bash=any(isinstance(gate, CommandGate) for gate in spec.gates),
        read=_policy_checks_reads(spec),
    )


def _policy_header(repo_name: str) -> str:
    """The `schema_version`/`policy` preamble both policy writers emit.

    One function rather than two near-identical strings, so a schema bump is
    one edit. The id is quoted because a directory name is not guaranteed to
    be a bare YAML scalar: one containing ": " or leading with "*"/"&"/"@"
    parses as something else entirely, or not at all.
    """
    return (
        'schema_version: "1.0"\n'
        "policy:\n"
        f"  id: {json.dumps(f'{repo_name}/guardrails')}\n"
        "  description: Rules this repo checks on its own working tree.\n"
        "\n"
        "gates:\n"
    )


def _starter_gates_yaml(repo_name: str) -> str:
    """A two-gate starter policy, chosen to teach `runs` rather than to be useful.

    One gate of each shape a `runs` value can take: a command gate, which can
    only ever refuse a call before it happens, and a path gate naming both
    moments, which is what most real rules want. Between them a reader sees
    that the field is a real choice and that the two halves of an entry say
    different things.
    """
    return _policy_header(repo_name) + (
        "  - id: no-force-push\n"
        "    type: command\n"
        "    runs: [pre_tool_use.command]\n"
        "    enforcement: advisory\n"
        '    forbidden: ["git push --force"]\n'
        "    message: >-\n"
        "      Force-pushing rewrites shared history. Use --force-with-lease\n"
        "      if you must.\n"
        "\n"
        "  - id: no-committed-env-file\n"
        "    type: path\n"
        # Both moments: the first refuses an edit tool before it writes, the
        # second catches anything else (a shell redirect, a script) once the
        # turn is over. Neither alone covers a path.
        #
        # A path nothing legitimately generates, on purpose. A gate over a
        # generated file warns on the very command that regenerates it, which
        # is the trap this repo's own policy documents twice (see the
        # postman-collection and pyproject gates in
        # .otari/guardrails/generated-artifacts.yml).
        "    runs: [pre_tool_use.edit_target, stop.working_tree]\n"
        "    enforcement: advisory\n"
        '    forbidden: [".env", "**/.env"]\n'
        "    message: >-\n"
        "      A .env file holds secrets and does not belong in the repo.\n"
        "      Keep it untracked and out of commits.\n"
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


class _HookSetup(NamedTuple):
    """Where one harness reads its own hook registration from, and the matcher vocabulary
    (see _HOOK_EDIT_TOOL_PATH_FIELDS, _CODEX_PATCH_TOOL_NAME, _HOOK_COMMAND_TOOL_FIELDS_BY_HARNESS)
    its own PreToolUse dispatch expects. Codex's own upstream docs say "exec" is accepted as a
    matcher alias for what its payload actually reports as tool_name "code_mode_exec"; that has
    not been confirmed against a real dispatch, so `command_matcher` below names both literally
    rather than depend on the alias translation actually being implemented.
    """

    settings_dir: str
    settings_name: str
    edit_matcher: str
    command_matcher: str
    read_matcher: str | None


_HOOK_SETUP_BY_HARNESS = {
    "claude-code": _HookSetup(".claude", "settings.local.json", "Edit|Write|NotebookEdit", "Bash", "Read"),
    # Codex has no read tool to match: its reads go through the shell, so a
    # read gate is not collectable there at all and the command matcher is
    # already the only thing that could see one.
    "codex": _HookSetup(".codex", "hooks.json", "apply_patch", "Bash|exec|code_mode_exec", None),
}


@hook.command(name="setup")
@click.option(
    "--harness",
    type=click.Choice(["claude-code", "codex"]),
    default="claude-code",
    show_default=True,
    help="Agent integration to configure.",
)
@click.option(
    "--api-key",
    default=None,
    help=(
        "Embed this credential in the generated command, opting the registered hook into checking "
        "against a gateway over HTTP instead of evaluating the guardrail locally. Omit for the default: "
        "no credential, no server, evaluated in process."
    ),
)
def hook_setup(harness: str, api_key: str | None) -> None:
    """Register otari hook in a supported agent's own settings.

    Writes or updates a PreToolUse hook entry and a Stop hook entry in the
    harness's own personal, gitignored settings file (see
    _HOOK_SETUP_BY_HARNESS) so registering it is not a manual JSON edit. Both
    point at the same otari hook invocation; the harness passes its own
    hook_event_name in the payload, so one callback serves either event.
    Offers to scaffold a starter .otari/guardrails.yml when this repo has no
    guardrail yet, and picks the PreToolUse matcher (whether it needs to cover
    the shell tool, the read tool, or both) from whatever gates the guardrail
    turns out to have; Stop needs no matcher; see docs/agent-guardrails.md for
    why both are registered unconditionally.
    """
    root = _hook_find_repo_root(Path.cwd())
    if root is None:
        raise click.ClickException("Not inside a Git repository.")

    # The single file, not a directory holding one: it is the smaller thing to
    # start with, and splitting it later is moving gates into
    # `.otari/guardrails/` rather than a migration.
    if not _hook_discover_guardrail_files(root):
        moved = _guardrail_moved_notice(root)
        if moved is not None:
            click.echo(moved)
        starter = root / GUARDRAIL_FILE
        if click.confirm(f"No guardrail found in {root}. Create a starter {GUARDRAIL_FILE}?", default=True):
            starter.parent.mkdir(parents=True, exist_ok=True)
            starter.write_text(_starter_gates_yaml(root.name), encoding="utf-8")
            click.echo(f"Wrote {starter}. Split it into {GUARDRAIL_DIR}/ when it grows.")
        else:
            click.echo(
                f"Skipping. otari hook will still be registered below, but every gate check "
                f"passes until {GUARDRAIL_FILE} or {GUARDRAIL_DIR}/ exists; see docs/agent-guardrails.md."
            )

    setup = _HOOK_SETUP_BY_HARNESS[harness]
    needs = _guardrail_matcher_needs(_hook_guardrail_spec(root))
    matcher_parts = [setup.edit_matcher]
    if needs.read and setup.read_matcher is not None:
        matcher_parts.append(setup.read_matcher)
    if needs.bash:
        matcher_parts.append(setup.command_matcher)
    matcher = "|".join(matcher_parts)

    # No credential resolution, and no prompt: `otari hook` evaluates the
    # local policy in process by default and needs neither. `--api-key` here
    # is the explicit opt-in to the other mode, checking against a gateway
    # over HTTP instead (see `hook`'s own docstring); embedding it is what
    # lets that mode work from a hook subprocess that inherits no shell
    # environment. Given no `--api-key`, the generated command carries none,
    # and stays that way even if a `master_key` happens to be configured
    # somewhere on this machine: resolving one here anyway would silently
    # decide, on this install's behalf, that gate checks should hit the
    # network at all.
    command_parts = [_otari_binary_path(), "hook", "--harness", harness]
    if api_key:
        command_parts += ["--api-key", api_key]
    command = shlex.join(command_parts)

    settings_path = root / setup.settings_dir / setup.settings_name
    pretooluse_created = _merge_hook_entry(settings_path, "PreToolUse", command, matcher=matcher)
    click.echo(f"{'Added' if pretooluse_created else 'Updated'} the PreToolUse hook in {settings_path}.")
    uncovered = [
        f"a command gate to also cover {setup.command_matcher}" if not needs.bash else "",
        (
            f"a path gate running at pre_tool_use.read_target to also cover {setup.read_matcher}"
            if not needs.read and setup.read_matcher is not None
            else ""
        ),
    ]
    missing = [item for item in uncovered if item]
    click.echo(f"Matcher: {matcher}" + (f" (add {', or '.join(missing)})" if missing else ""))

    # Registered unconditionally, not only when the policy has a gate that
    # benefits: path already falls back to `git status` on Stop
    # (catching a Bash-written change PreToolUse never saw coming), and
    # command_if_changed/command now read real command evidence there
    # too (from the session's own transcript; see docs/agent-guardrails.md). A
    # PreToolUse-only install left both silently unreachable.
    stop_created = _merge_hook_entry(settings_path, "Stop", command)
    click.echo(f"{'Added' if stop_created else 'Updated'} the Stop hook in {settings_path}.")


# otari's own doc, checked first; CLAUDE.md is a one-line @AGENTS.md import in
# this repo (see AGENTS.md, top) and elsewhere it copies that pairing, so it
# carries real content only when AGENTS.md itself is missing.
_GATES_GENERATE_DEFAULT_SOURCE_NAMES = ("AGENTS.md", "CLAUDE.md")
_GATES_GENERATE_DEFAULT_CLI_ORDER = ("claude", "codex")
_GATES_GENERATE_TIMEOUT_SECONDS = 300
# A developer-authored doc, not a data export; bounds a pathological input
# (and an accidental binary) before it goes into a model prompt, the same
# reasoning domain/policy.py's MAX_POLICY_BYTES applies to a policy body.
_GATES_GENERATE_MAX_SOURCE_BYTES = 200 * 1024
_GATES_GENERATE_MAX_PROPOSALS = 15

_GATES_GENERATE_SCHEMA_REFERENCE = """A gate is one YAML mapping with these fields:

Common to every gate: `id` (unique, short, kebab-case, at most 200 characters),
`type`, `enforcement` (`required` or `advisory`), `message` (shown when it
fails; should point back at the rule/section it came from), and `runs`, a
non-empty list naming when the gate runs and what it can see there. Each gate
type accepts only certain `runs` values, and there is no default:

- path: `pre_tool_use.edit_target`, `pre_tool_use.read_target`, `stop.working_tree`, or any combination
- command: `pre_tool_use.command`
- command_if_changed: `stop.session`
- judge: `stop.session`
- verifier: `stop.verifier`

`pre_tool_use.edit_target` is the path an Edit/Write/NotebookEdit call names
before it runs, so a match refuses the write. It cannot see a path a shell
command writes (a redirect, sed -i, cp, a script). `stop.working_tree` is what
git status reports once the turn is over: complete whatever wrote the file, but
always after the fact. Propose both for a path gate unless the path is
only ever written by a build or a generator, in which case propose
`[stop.working_tree]` alone, because no tool call will ever name it.

`pre_tool_use.read_target` is the path a Read call names before it runs, so a
match refuses the read and the file's contents never enter the transcript.
Propose it only for a rule about *reading* a file (a secret, a credential, a
key), never as an extra entry on a "do not hand-edit this" gate, where it would
refuse merely looking at the file. It cannot see a shell read (cat, less), and
unlike a write there is no `stop` value that catches one afterwards, because a
read changes nothing: if the doc also wants shell reads refused, propose a
separate command gate for that.

- path: fails when a changed path matches one of `forbidden`, a list
  of repo-relative POSIX globs (`*` within one path segment, at most one `**`
  crossing segments per glob). Use for "this generated/forbidden file must
  never be hand-edited".
- command: fails when a run command matches one of `forbidden`, a list
  of shell phrases (e.g. "npm install", "git push --force"), matched as a
  contiguous token run, not a substring. Use for "use tool X, not tool Y".
- command_if_changed: fails when a path matching `when_changed` (same glob
  grammar as path) changed but none of `require` (same phrase
  grammar as command) ran. Use for "if this generated artifact
  changed, its generator command must have run".
- verifier: a repo-local verifier script's own exit code decides the
  outcome (0 pass, 1 fail, anything else error). Needs `verifier`, a
  repo-relative path. Only propose this when the doc names, or clearly
  implies, a script that already exists in the repo; never invent a path.
- judge: a model verdict against a free-text `rubric`. `enforcement` MUST be
  `advisory` (a required judge gate is invalid and will be rejected). Use
  only for a genuinely subjective rule no mechanical check can express;
  prefer one of the other four types whenever the rule is checkable
  mechanically.

Optional on command_if_changed/judge/verifier: `when_changed` (same glob
grammar as path's `forbidden`) scopes when the gate applies; required
for command_if_changed, optional (defaults to "always") for the other two.
"""


def _gates_generate_build_prompt(*, doc_name: str, doc_text: str, existing_ids: frozenset[str]) -> str:
    existing_ids_text = ", ".join(sorted(existing_ids)) if existing_ids else "(none yet)"
    return (
        "You are proposing gates for an otari guardrail: mechanical "
        "rules a coding agent's own hook checks against its working tree and "
        "commands before proceeding.\n\n"
        f"{_GATES_GENERATE_SCHEMA_REFERENCE}\n"
        f"Gate ids already used in this policy (never reuse one of these): {existing_ids_text}\n\n"
        f"Read the following repo doc ({doc_name}) and propose gates only for rules "
        "that are clear, specific, and checkable the way described above: an "
        "instruction to never hand-edit a named generated file, a preference for one "
        "command/tool over another, an 'if X changed, run Y' pairing, or (rarely, "
        "advisory only) a genuinely subjective style rule. Do not propose a gate for "
        "a rule that is vague, purely descriptive, or needs context this doc does not "
        f"give (a concrete file path, command, or script path). Propose at most "
        f"{_GATES_GENERATE_MAX_PROPOSALS} gates, clearest and most confidently "
        "mechanical first.\n\n"
        "Reply with ONLY a JSON array of gate objects: no markdown fence, no prose "
        "before or after. An empty array `[]` is a fine answer if nothing in the doc "
        "is a good fit.\n\n"
        f"--- {doc_name} ---\n{doc_text}\n--- end of {doc_name} ---\n"
    )


def _gates_generate_run_cli(argv: list[str], prompt: str, *, label: str) -> str:
    """Run one gate-generation CLI call; return its raw stdout.

    A thin sibling of `_hook_run_judge_subprocess`, not a reuse of it: that
    helper's own tail parses the fixed `{"outcome", "reasoning"}` shape a
    judge gate's prompt demands, not the JSON array of gate proposals this
    command's own prompt asks for. Runs from `_hook_judge_workdir()`, the
    same isolated directory a judge gate's own model call uses and for the
    same reason (see that function's own docstring): this repo's own
    guardrail can register `otari hook` on `Stop`, and running this
    call from the repo it is reading would let that fire for this call too.
    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, resolved executable path
            argv,
            input=prompt,
            cwd=_hook_judge_workdir(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_GATES_GENERATE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise click.ClickException(f"{label} did not respond within {_GATES_GENERATE_TIMEOUT_SECONDS}s.") from exc
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Could not run {label}: {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise click.ClickException(f"{label} exited {result.returncode}: {detail[:2000]}")
    return result.stdout


def _gates_generate_call_claude_p(claude_path: str, model: str | None, prompt: str) -> str:
    argv = [claude_path]
    if model:
        argv += ["--model", model]
    argv += ["--tools", "", "--strict-mcp-config", "-p"]
    return _gates_generate_run_cli(argv, prompt, label="claude -p")


def _gates_generate_call_codex_exec(codex_path: str, model: str | None, prompt: str) -> str:
    argv = [codex_path, "exec", "-"]
    if model:
        argv += ["--model", model]
    # Same flags _hook_call_codex_exec uses, and for the same reasons (see
    # that function's own docstring): --sandbox/--ask-for-approval bound what
    # an attempted tool call could do against this call's own prompt (the doc
    # text is as attacker-influenceable as a judge gate's diff/transcript),
    # --skip-git-repo-check because _hook_judge_workdir() is a plain
    # directory, and --ephemeral so this one-shot call leaves no rollout file
    # there for a later judge gate's own transcript scan to ever mistake for
    # real session evidence.
    argv += [
        "--sandbox",
        "read-only",
        "--ask-for-approval",
        "never",
        "--skip-git-repo-check",
        "--ephemeral",
        "--color",
        "never",
    ]
    return _gates_generate_run_cli(argv, prompt, label="codex exec")


def _gates_generate_call_cli(cli_name: str, cli_path: str, model: str | None, prompt: str) -> str:
    if cli_name == "codex":
        return _gates_generate_call_codex_exec(cli_path, model, prompt)
    return _gates_generate_call_claude_p(cli_path, model, prompt)


def _gates_generate_parse_response(raw: str) -> list[dict[str, Any]]:
    text = _hook_strip_judge_code_fence(raw)
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise click.ClickException(f"Model did not return valid JSON: {raw[:2000]!r}") from exc
    if isinstance(parsed, dict) and isinstance(parsed.get("gates"), list):
        parsed = parsed["gates"]
    if not isinstance(parsed, list):
        raise click.ClickException(f"Model did not return a JSON array of gate proposals: {raw[:2000]!r}")
    return [item for item in parsed if isinstance(item, dict)][:_GATES_GENERATE_MAX_PROPOSALS]


def _gates_generate_validate_gate(gate_dict: dict[str, Any]) -> None:
    """Raise PolicyError unless `gate_dict` is a well-formed gate on its own.

    Wraps it in a minimal policy skeleton and runs it through the exact
    parser a submitted guardrail/Hook Server request goes through
    (`otari_agent.domain.policy.parse_policy`), so a hallucinated field,
    type, or a `judge` gate proposed as `required` is caught here, before
    this ever gets appended to the real file, not the first time the hook
    actually runs against it.
    """
    skeleton = {"schema_version": "1.0", "policy": {"id": "guardrails-generate/preview"}, "gates": [gate_dict]}
    parse_policy(yaml.safe_dump(skeleton, sort_keys=False, allow_unicode=True), source="proposed gate")


def _gates_generate_dump(gate_dict: dict[str, Any]) -> str:
    return yaml.safe_dump(gate_dict, default_flow_style=False, sort_keys=False, allow_unicode=True)


# Matches one "key: value" (or "key:" alone, before a nested block) line of
# `_gates_generate_dump`'s own plain-block YAML output, to style the key
# without having to reimplement a YAML emitter that colors as it writes.
_GATES_GENERATE_DUMP_KEY_LINE = re.compile(r"^(\s*)([A-Za-z_][\w]*):(.*)$")


def _gates_generate_describe_gate(gate_dict: dict[str, Any]) -> str:
    """Render one gate for terminal display: each field's own key in bold cyan, a
    list item (a `forbidden`/`when_changed`/`require` entry) dimmed, everything else as-is.

    A long `rubric`/`message` string is one value PyYAML wraps across several
    physical lines; only the first of those matches `_GATES_GENERATE_DUMP_KEY_LINE`
    (it alone has "key:" at its start), so only a genuine list-item line
    ("- entry", from `forbidden`/`when_changed`/`require`) is dimmed here.
    Dimming every non-key line too, the first cut of this, made a single
    wrapped value read as two different colors mid-sentence, its own
    continuation lines mistaken for a lesser, list-item kind of line.
    """
    lines = []
    for line in _gates_generate_dump(gate_dict).rstrip("\n").split("\n"):
        key_match = _GATES_GENERATE_DUMP_KEY_LINE.match(line)
        if key_match:
            indent, key, rest = key_match.groups()
            lines.append(f"{indent}{click.style(key, fg='cyan', bold=True)}:{rest}")
        elif line.lstrip().startswith("- "):
            lines.append(click.style(line, dim=True))
        else:
            lines.append(line)
    return "\n".join(lines)


def _gates_generate_read_choice(message: str, choices: str, default: str) -> str:
    """Print `message`, then read a single keypress (no Enter needed): a char in `choices`
    (each lowercase, e.g. "yneq") returns immediately; Enter alone resolves to `default`;
    anything else reprints `message` rather than guessing.

    Falls back to an ordinary Enter-terminated line prompt when there is no
    controlling terminal to read a raw keypress from (piped/non-interactive
    stdin, no `/dev/tty`, some CI and sandboxed runners): `click.getchar()`'s
    own fallback for a non-tty `stdin` opens `/dev/tty` directly, and that
    raises a plain `OSError`, not one of click's own catchable exceptions,
    when no such device exists at all.
    """
    while True:
        click.echo(click.style(message, fg="cyan"), nl=False)
        try:
            raw = click.getchar(echo=True)
            click.echo("")
        except OSError:
            # The message above is already on screen; read a plain line rather
            # than let click.prompt print it a second time as its own prompt.
            try:
                raw = input()
            except EOFError:
                # Nothing on stdin at all is not the same as someone pressing
                # Enter: an empty line takes `default`, which for the
                # "use this CLI?" prompt is "y", and nobody asked for a model
                # call here. Decline instead wherever declining is a choice.
                return "n" if "n" in choices else default
        key = raw.strip().lower()
        if key in ("", "\r", "\n"):
            return default
        if key[0] in choices:
            return key[0]
        click.secho(f"Press one of: {', '.join(choices)} (or Enter for {default!r}).", fg="red")


def _gates_generate_render_list_item(gate_dict: dict[str, Any], indent: str = "  ") -> str:
    """Render one accepted gate as a block-sequence item for a guardrail file, its
    `- ` marker at `indent`.

    Deliberately plain block style throughout (PyYAML's own default), not
    the inline `forbidden: [...]`/folded `message: >-` conventions this
    repo's hand-written gates use: those are a human author's own
    formatting choice, not something this command needs to reproduce to
    stay valid and readable.

    `indent` is what the surrounding file already uses for its own items
    (`_gates_generate_existing_item_indent`), not a fixed two spaces: YAML
    wants every item of one block sequence at the same column, and a
    sequence written in column 0 (PyYAML's own dump style, `gates:\\n- id:`)
    is as valid as this repo's own indented one.
    """
    lines = _gates_generate_dump(gate_dict).rstrip("\n").split("\n")
    continuation = " " * (len(indent) + 2)
    rendered = [f"{indent}- {lines[0]}", *(f"{continuation}{line}" for line in lines[1:])]
    return "\n".join(rendered) + "\n"


def _gates_generate_existing_item_indent(lines: list[str], gates_line_idx: int) -> str:
    """Indentation the file's own `gates:` items already use, for a new item to match.

    Falls back to two spaces when the sequence is empty of items this can
    see, which is this repo's own style and the one the scaffolded header
    below writes.
    """
    for line in lines[gates_line_idx + 1 :]:
        stripped = line.lstrip()
        if stripped.startswith("- "):
            return line[: len(line) - len(stripped)]
        if stripped and not stripped.startswith("#") and not line.startswith((" ", "\t")):
            break
    return "  "


def _gates_generate_find_source(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for name in _GATES_GENERATE_DEFAULT_SOURCE_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise click.ClickException(f"No AGENTS.md or CLAUDE.md found in {root}. Pass --source to name a different file.")


def _gates_generate_append(gates_file: Path, repo_name: str, gate_dict: dict[str, Any]) -> None:
    """Append one already-validated gate to `gates_file`'s `gates:` sequence.

    Splices raw text rather than round-tripping the file through a YAML
    dump, so every hand-written comment already in it (as in this repo's
    own guardrail) survives untouched. `schema_version`, `policy`,
    and `gates` are a policy's only top-level keys
    (domain/policy.py's `_TOP_LEVEL_FIELDS`), so the end of the `gates:`
    sequence is wherever a following line returns to column 0 without being
    one of the sequence's own items; this appends just before that line, or
    at end of file when nothing follows (the common case, `gates:` declared
    last).

    Splicing text is a heuristic where the policy loader is a parser, so
    nothing here is written until `parse_policy` accepts the result: a
    layout this misreads (a comment a human wrote at column 0 inside the
    block, a sequence style this did not anticipate) then costs a refusal
    with the file untouched, never a corrupted policy. That distinction
    matters more than it looks: `otari hook` fails *open* on a policy it
    cannot parse, so silently writing a broken one would quietly stop every
    gate in it from being enforced, required ones included.
    """
    if not gates_file.is_file():
        header = _policy_header(repo_name)
        gates_file.parent.mkdir(parents=True, exist_ok=True)
        _gates_generate_write_checked(gates_file, header + _gates_generate_render_list_item(gate_dict))
        return

    lines = gates_file.read_text(encoding="utf-8").splitlines(keepends=True)
    gates_line_idx = next((i for i, line in enumerate(lines) if line.rstrip("\n") == "gates:"), None)
    if gates_line_idx is None:
        raise click.ClickException(f"Could not find a top-level 'gates:' key in {gates_file}.")

    item_indent = _gates_generate_existing_item_indent(lines, gates_line_idx)
    end_idx = len(lines)
    for i in range(gates_line_idx + 1, len(lines)):
        if lines[i].strip() == "":
            continue
        # A column-0 line that is itself one of this sequence's items ends
        # nothing: PyYAML's own dump style writes every item there.
        if not lines[i].startswith((" ", "\t")) and not lines[i].lstrip().startswith("- "):
            end_idx = i
            break

    gate_block = _gates_generate_render_list_item(gate_dict, item_indent)
    new_lines = [*lines[:end_idx], "\n", *gate_block.splitlines(keepends=True), *lines[end_idx:]]
    _gates_generate_write_checked(gates_file, "".join(new_lines))


def _joins_the_composed_set(target: Path, root: Path) -> bool:
    """Whether creating ``target`` would add a file to what the hook composes.

    `--guardrail-file` can name somewhere the hook never reads, in which case
    a new file there changes the composed set not at all.

    Every path here is resolved, on both sides of the comparison, because this
    compares paths and any two spellings of one path compare unequal. The
    caller resolves ``target``: a relative `--guardrail-file` is the obvious
    spelling, and an absolute one reached through a symlinked parent is the
    less obvious one (`/tmp` is a link to `/private/tmp` on macOS, so every
    path under it has two). The set side is resolved here for the case this
    feature invites: `.otari/guardrails` itself symlinked at a directory
    shared between repositories, where the files are discovered through the
    link but a resolved target lands outside the lexical path. Either way the
    answer would be a wrong "outside the set", and the caller skips the
    file-count guard on the strength of it.
    """
    if target == (root / GUARDRAIL_FILE).resolve():
        return True
    return target.suffix in _GUARDRAIL_SUFFIXES and target.is_relative_to((root / GUARDRAIL_DIR).resolve())


def _generate_target(root: Path) -> Path:
    """Where accepted gates go when `--guardrail-file` names nowhere.

    Whichever shape the repo already keeps: the single file where it has one,
    a file of its own under the directory where it has that instead, rather
    than dropping generated gates into somebody's hand-organized file. A repo
    with neither gets the single file, the same starting shape
    `otari hook setup` scaffolds.
    """
    single = root / GUARDRAIL_FILE
    if single.is_file() or not (root / GUARDRAIL_DIR).is_dir():
        return single
    return root / GUARDRAIL_DIR / "generated.yml"


def _gates_generate_write_checked(gates_file: Path, new_text: str) -> None:
    """Write `new_text` only once `parse_policy` accepts it; refuse, untouched, otherwise."""
    try:
        parse_policy(new_text, source=str(gates_file))
    except PolicyError as exc:
        raise click.ClickException(
            f"Appending to {gates_file} would produce a guardrail that no longer parses ({exc}); left it unchanged."
        ) from exc
    gates_file.write_text(new_text, encoding="utf-8")


def _gates_generate_parse_edit(edited: str | None, *, fallback: dict[str, Any]) -> dict[str, Any]:
    if edited is None:
        click.echo("No changes made.")
        return fallback
    try:
        parsed = yaml.safe_load(edited)
    except yaml.YAMLError as exc:
        click.echo(f"Edited text is not valid YAML ({exc}); keeping the previous version.")
        return fallback
    if not isinstance(parsed, dict):
        click.echo("Edited gate must be a YAML mapping; keeping the previous version.")
        return fallback
    return parsed


@click.group(name="guardrails")
def guardrails() -> None:
    """Work with a repo's guardrail: .otari/guardrails.yml, or the files under .otari/guardrails/."""


@guardrails.command(name="generate")
@click.option(
    "--source",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Doc to read candidate rules from. Defaults to AGENTS.md, then CLAUDE.md, in the repo root.",
)
@click.option(
    "--guardrail-file",
    "guardrail_file_option",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        f"Guardrail file to append accepted gates to. Defaults to {GUARDRAIL_FILE}, or "
        f"{GUARDRAIL_DIR}/generated.yml in a repo that keeps its guardrail as a directory."
    ),
)
@click.option(
    "--cli",
    "cli_override",
    callback=_parse_judge_cli,
    default=None,
    envvar="OTARI_GUARDRAILS_GENERATE_CLI",
    help=(
        "Comma-separated, ordered CLI backend(s) to try (claude, codex). Defaults to claude, "
        "then codex, whichever is found on PATH first."
    ),
)
@click.option(
    "--model",
    default=None,
    envvar="OTARI_GUARDRAILS_GENERATE_MODEL",
    help="Model the resolved CLI uses for its one generation call. Left unset uses that CLI's own default model.",
)
def guardrails_generate(
    source: Path | None,
    guardrail_file_option: Path | None,
    cli_override: tuple[str, ...] | None,
    model: str | None,
) -> None:
    """Propose guardrail gates from a repo's own AGENTS.md/CLAUDE.md, one at a time.

    Resolves a locally installed model CLI (`claude -p` or `codex exec`,
    whichever is found on PATH first; no otari server, no otari credential)
    and asks the user to confirm it before sending the doc anywhere. Walks
    the proposals one at a time, screen cleared between each so only the
    current candidate is on screen: shows it and asks
    `[y]es/[n]o/[e]dit/[q]uit`, a single keypress, no Enter needed. A
    rejected or skipped proposal is never written; a proposal whose id
    already exists in the policy is skipped without asking. Every accepted
    gate (edited or not) is validated the same way a submitted policy is
    (`otari_agent.domain.policy.parse_policy`) before it is appended, so a
    hallucinated field or type is caught here, not the first time the hook
    actually runs. See docs/agent-guardrails.md for the gate schema this asks the
    model to stay inside.
    """
    root = _hook_find_repo_root(Path.cwd())
    if root is None:
        raise click.ClickException("Not inside a Git repository.")

    source_path = _gates_generate_find_source(root, source)
    try:
        source_text = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise click.ClickException(f"Could not read {source_path} as UTF-8 text: {exc}") from exc
    if len(source_text.encode("utf-8")) > _GATES_GENERATE_MAX_SOURCE_BYTES:
        raise click.ClickException(
            f"{source_path} is larger than {_GATES_GENERATE_MAX_SOURCE_BYTES} bytes; "
            "pass --source to point at a smaller/narrower doc."
        )

    # Resolved, not as typed: `--guardrail-file` accepts any spelling, and
    # every check below compares it against paths under the resolved repo
    # root. See `_joins_the_composed_set`.
    target = (guardrail_file_option if guardrail_file_option is not None else _generate_target(root)).resolve()

    # Before the model call, not after: writing the file that takes the set
    # past MAX_POLICY_FILES would report success and leave the repo with a
    # guardrail the hook refuses to compose, which fails open. Every gate
    # already in the set, this one included, would stop being enforced, and
    # the command that did it would have said "Added 1 gate(s)".
    composed = _hook_discover_guardrail_files(root)
    if not target.is_file() and _joins_the_composed_set(target, root) and len(composed) >= MAX_POLICY_FILES:
        raise click.ClickException(
            f"{root} already composes {len(composed)} guardrail files, the most this build reads "
            f"({MAX_POLICY_FILES}). Adding {_guardrails_relative_to(target, root) or target} would make the "
            "whole guardrail unloadable, so every gate in it would stop being enforced. Pass "
            "--guardrail-file to append to one of the existing files instead."
        )

    # Every id already in use anywhere the hook composes, not only in the file
    # being appended to: a gate id is unique across the whole composed
    # guardrail, so a duplicate landing in a second file would not be a
    # shadowed gate but a guardrail that stops loading altogether, and a
    # guardrail that fails to load enforces nothing.
    # Which file each is in, not just that it is taken: with several files in
    # play, "already in the guardrail" leaves a reader opening each one to
    # find out where.
    existing_ids: dict[str, str] = {}
    for path in dict.fromkeys([*composed, target]):
        if not path.is_file():
            continue
        try:
            existing_spec = parse_policy(path.read_text(encoding="utf-8"), source=str(path))
        except (OSError, UnicodeDecodeError, PolicyError) as exc:
            raise click.ClickException(f"{path} does not currently parse: {exc}") from exc
        name = _guardrails_relative_to(path, root) or path.name
        existing_ids.update({gate.id: name for gate in existing_spec.gates})

    candidates = cli_override if cli_override is not None else _GATES_GENERATE_DEFAULT_CLI_ORDER
    resolved = _hook_resolve_judge_cli(candidates)
    if resolved is None:
        raise click.ClickException(f"None of {', '.join(candidates)} was found on PATH. Install one, or pass --cli.")
    cli_name, cli_path = resolved

    try:
        display_source = source_path.relative_to(root)
    except ValueError:
        display_source = source_path

    if (
        _gates_generate_read_choice(
            f"Use {cli_name} ({cli_path}) to read {display_source} and propose gates? [Y/n]: ", "yn", "y"
        )
        == "n"
    ):
        raise click.Abort()

    prompt = _gates_generate_build_prompt(
        doc_name=str(display_source), doc_text=source_text, existing_ids=frozenset(existing_ids)
    )
    click.secho(f"Asking {cli_name} to propose gates from {display_source}...", dim=True)
    raw_output = _gates_generate_call_cli(cli_name, cli_path, model, prompt)
    proposals = _gates_generate_parse_response(raw_output)

    if not proposals:
        click.echo("No gate proposals came back.")
        return

    total = len(proposals)
    accepted = 0
    for index, raw_gate in enumerate(proposals, start=1):
        current: Any = raw_gate
        while True:
            click.clear()
            click.secho(f"Gate proposal {index} of {total}", fg="cyan", bold=True)
            click.echo()

            if not isinstance(current, dict):
                click.secho("Skipping a proposal that is not a mapping.", fg="yellow")
                break
            gate_id = current.get("id")
            if not isinstance(gate_id, str) or not gate_id:
                click.secho("Skipping a proposal with no valid 'id'.", fg="yellow")
                break
            if gate_id in existing_ids:
                click.secho(f"Skipping {gate_id!r}: already in {existing_ids[gate_id]}.", fg="yellow")
                break

            try:
                _gates_generate_validate_gate(current)
            except PolicyError as exc:
                click.echo(_gates_generate_describe_gate(current))
                click.echo()
                click.secho(f"[{gate_id}] does not pass validation: {exc}", fg="red")
                if _gates_generate_read_choice("Edit it and try again? [y/N]: ", "yn", "n") == "y":
                    current = _gates_generate_parse_edit(click.edit(_gates_generate_dump(current)), fallback=current)
                    continue
                break

            click.echo(_gates_generate_describe_gate(current))
            click.echo()
            choice = _gates_generate_read_choice("Add this gate? [y]es/[n]o/[e]dit/[q]uit: ", "yneq", "n")
            if choice == "y":
                _gates_generate_append(target, root.name, current)
                existing_ids[gate_id] = _guardrails_relative_to(target, root) or target.name
                accepted += 1
                click.secho(f"Added {gate_id!r} to {target}.", fg="green")
                break
            if choice == "e":
                current = _gates_generate_parse_edit(click.edit(_gates_generate_dump(current)), fallback=current)
                continue
            if choice == "q":
                click.secho(f"Stopped early. Added {accepted} gate(s) to {target}.", fg="yellow")
                return
            break

    click.secho(f"Added {accepted} gate(s) to {target}.", fg="green" if accepted else None)


# Wide enough for the longest dry-run label ("would run"), so every gate id
# starts in the same column whichever moment is being reported.
_VALIDATE_LABEL_WIDTH = 9


def _declared_in(result: GateResult) -> str:
    """`` in <file>`` for a gate from a composed guardrail, empty for a single-file one."""
    return f" in {result.source}" if result.source else ""


def _guardrails_probe_verifier(repo_root: Path, verifier: str) -> str | None:
    """Report why `otari hook` could not run this verifier gate's script, or None if it could.

    Mirrors `_hook_run_check_verifier`'s own resolution, containment check
    included, so validate and the real run agree on which scripts are
    reachable. The one thing it adds is the executable bit: the real run
    learns that from a failed exec and reports `error`, which blocks a
    required gate, and an author would rather hear it here.
    """
    resolved_root = repo_root.resolve()
    script_path = (repo_root / verifier).resolve()
    try:
        script_path.relative_to(resolved_root)
    except ValueError:
        return f"verifier {verifier!r} resolves outside the repo root, so the hook refuses to run it."
    if not script_path.is_file():
        return f"verifier {verifier!r} does not exist."
    if not os.access(script_path, os.X_OK):
        return f"verifier {verifier!r} is not executable; `chmod +x {verifier}`."
    return None


def _guardrails_relative_to(path: Path, base: Path) -> str | None:
    """``path`` as a repo-relative POSIX string, or ``None`` when it is not under ``base``."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return None


def _guardrails_repo_relative(repo_root: Path, path: str) -> tuple[str, str]:
    """Spell one dry-run path the way `hook` spells it at each moment: (edit target, working tree).

    Both are always present: a path with no spelling inside the repo is
    refused outright above, and every other path has a PreToolUse moment now
    that `hook` submits the lexical spelling as well as the resolved one.

    A gate's globs are repo-relative POSIX, and matching `--path`
    literally would report `quiet` for the absolute or `./`-prefixed spelling a
    person naturally types while a real session matches it: a silent false
    clean, which is the failure this whole command exists to remove.

    The two spellings differ only for a symlink, and differ for a reason.
    `hook`'s PreToolUse branch calls `.resolve()`, which is how an absolute
    `file_path` becomes repo-relative and which follows a link on the way. Its
    Stop branch reads `git status`, which reports the tracked name, link and
    all. Using one spelling for both reports a gate forbidding the link's own
    name as `quiet` at Stop, where a real session fails it.

    Three spellings are tried for the working tree, in order, because a repo
    can be named through an alias: purely lexical, then with the directory
    resolved but the final name kept, then fully resolved. The middle one is
    what an absolute path through an alias needs (`/tmp/repo/CLAUDE.md` where
    the repo is really at `/private/tmp/repo`): without it the lexical attempt
    fails, the fully resolved fallback follows the file's own link too, and
    the Stop spelling silently becomes the link's target again.
    """
    candidate = repo_root / path
    # normpath, not resolve: it collapses `.` and `..` lexically, which is what
    # keeps a link's own name intact for the Stop spelling.
    lexical = Path(os.path.normpath(candidate))
    # The directory resolved, the final name left alone: normalizes the repo's
    # own alias without turning a symlinked file into its target.
    anchored = lexical.parent.resolve() / lexical.name
    resolved = candidate.resolve()
    root_resolved = repo_root.resolve()
    working_tree = (
        _guardrails_relative_to(lexical, repo_root)
        or _guardrails_relative_to(anchored, root_resolved)
        or _guardrails_relative_to(resolved, root_resolved)
    )
    if working_tree is None:
        raise click.ClickException(
            f"--path {path!r} is outside {repo_root}; a guardrail can only name paths inside the repo."
        )
    # Falls back to the working-tree spelling, because `hook`'s own PreToolUse
    # branch no longer abandons a target that resolves out of the repo: it
    # submits the lexical spelling too (see _hook_evidence_paths), so the
    # link's own name really is evaluated and a preview that skipped it would
    # promise silence where a real session refuses. None is now reachable only
    # when neither spelling lands inside the repo, and `working_tree` being
    # non-None means at least one did.
    return _guardrails_relative_to(resolved, root_resolved) or working_tree, working_tree


def _guardrails_dry_run(
    spec: PolicySpec,
    *,
    heading: str,
    paths: list[str],
    path_source: RunsAt | None,
    commands: list[str],
    command_scope: EvidenceScope,
) -> None:
    """Evaluate one hypothetical moment and print what each gate does there.

    The evidence shape mirrors, field for field, what the matching branch of
    `hook` submits for that event, so a gate that fires here fires there.

    Judge and verifier gates at Stop are reported by whether `when_changed`
    selects them, not by an outcome: each needs something actually run (a
    model call, a script) that validate deliberately does not run. Naming
    them anyway is the point, since a judge gate's cost is one model call per
    session it applies to.

    Both of those types carry a per-Stop cap, applied to the gates
    `when_changed` selected, highest `priority` first. The preview applies the
    same caps for the same reason it mirrors the evidence shape: a preview
    that promises six model calls where the hook makes five is wrong about
    exactly the number it exists to report.
    """
    try:
        check = check_policy(
            spec,
            paths=paths,
            commands=commands,
            path_source=path_source,
            command_scope=command_scope,
        )
    except PolicyCheckError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo()
    click.echo(heading)
    changed = tuple(paths)

    # Which gates survive each cap, resolved up front so the loop below can
    # report a gate `when_changed` selected but the cap then dropped. Mirrors
    # _hook_collect_judge_verdicts/_hook_collect_check_verdicts: filter by
    # when_changed, order by priority, then take the first N.
    def applies(gate: JudgeGate | VerifierGate) -> bool:
        return not gate.when_changed or bool(matched_changed_paths(gate.when_changed, changed))

    judges = by_priority([gate for gate in spec.gates if isinstance(gate, JudgeGate) and applies(gate)])
    verifiers = by_priority([gate for gate in spec.gates if isinstance(gate, VerifierGate) and applies(gate)])
    running = {gate.id for gate in judges[:_HOOK_JUDGE_MAX_GATES_PER_RUN]} | {
        gate.id for gate in verifiers[:_HOOK_CHECK_MAX_GATES_PER_RUN]
    }

    elsewhere = 0
    for gate, result in zip(spec.gates, check.results, strict=True):
        if isinstance(gate, JudgeGate | VerifierGate) and command_scope == "session":
            kind = "judge" if isinstance(gate, JudgeGate) else "verifier"
            limit = _HOOK_JUDGE_MAX_GATES_PER_RUN if kind == "judge" else _HOOK_CHECK_MAX_GATES_PER_RUN
            if gate.when_changed and not matched_changed_paths(gate.when_changed, changed):
                click.echo(f"  {'skipped':<{_VALIDATE_LABEL_WIDTH}}  {gate.id} ({kind}, when_changed does not match)")
            elif gate.id not in running:
                click.secho(
                    f"  {'skipped':<{_VALIDATE_LABEL_WIDTH}}  {gate.id} ({kind}, past the {limit}-gate cap "
                    "for one Stop event)",
                    fg="yellow",
                )
            else:
                cost = ", one model call" if kind == "judge" else ""
                click.echo(f"  {'would run':<{_VALIDATE_LABEL_WIDTH}}  {gate.id} ({kind}, {gate.enforcement}{cost})")
        elif result.outcome is Outcome.FAIL:
            click.secho(
                f"  {'fires':<{_VALIDATE_LABEL_WIDTH}}  {gate.id} ({gate.enforcement}){_declared_in(result)}",
                fg="yellow",
            )
        elif result.outcome is Outcome.PASS:
            click.echo(f"  {'quiet':<{_VALIDATE_LABEL_WIDTH}}  {gate.id} ({gate.enforcement})")
        else:
            elsewhere += 1
    if elsewhere:
        click.echo(f"  {elsewhere} gate(s) do not apply here.")


@guardrails.command(name="validate")
@click.option(
    "--guardrail-file",
    "guardrail_file_option",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Check this one file on its own, the way a shared snippet is checked before it is dropped in. "
        "Defaults to everything the repo composes."
    ),
)
@click.option(
    "--command",
    "dry_run_commands",
    multiple=True,
    help="Dry run this shell command against the guardrail. Repeatable.",
)
@click.option(
    "--path",
    "dry_run_paths",
    multiple=True,
    help="Dry run this repo-relative path against the guardrail. Repeatable.",
)
@click.option("--strict", is_flag=True, help="Exit non-zero on a warning too, not only on an error.")
def guardrails_validate(
    guardrail_file_option: Path | None,
    dry_run_commands: tuple[str, ...],
    dry_run_paths: tuple[str, ...],
    strict: bool,
) -> None:
    """Check this repo's guardrail without running it, and try it against a command or a path.

    Checks everything the hook composes: `.otari/guardrails.yml` where a repo
    keeps one file, and every `.yml`/`.yaml` under `.otari/guardrails/`,
    nested ones included. Cross-file problems are found here and only here,
    because no single file can show them: a gate id declared twice across the
    set, or more judge gates in total than one Stop event will run.
    `--guardrail-file` narrows it back to one file, which is how a snippet
    from somewhere else is checked before being dropped in.

    Offline and gateway-free: it parses the guardrail the same way a
    submitted one is parsed, then reports what running it would have taught
    the hard way. An error is a gate that cannot do its job (a missing or
    unrunnable verifier script). A warning is a gate that runs and may not
    mean what its author intended: a `**` glob that cannot reach the
    repository root, a one-token forbidden phrase that also refuses commands
    merely mentioning the word, a path gate blind to every shell write, more
    judge gates than one Stop event evaluates. A warning never fails on its
    own, since each has a legitimate exception; `--strict` is what makes one
    non-zero, for CI.

    `--command` and `--path` answer the other question, "does it say
    what I think it says", by evaluating the guardrail against evidence you
    supply at each moment a real session would offer it: one PreToolUse call
    per command or path, then the Stop event with all of them together. A
    gate firing there is the answer, not a failure, so it does not change the
    exit status.

    See docs/agent-guardrails.md.
    """
    root = _hook_find_repo_root(Path.cwd())
    if root is None:
        raise click.ClickException("Not inside a Git repository.")

    if guardrail_file_option is not None:
        if not guardrail_file_option.is_file():
            raise click.ClickException(
                f"No guardrail file at {guardrail_file_option}. `otari guardrails generate` starts one."
            )
        files = [guardrail_file_option]
    else:
        files = _hook_discover_guardrail_files(root)
        if not files:
            moved = _guardrail_moved_notice(root)
            raise click.ClickException(
                moved
                or (
                    f"No guardrail in {root}: no {GUARDRAIL_FILE} and nothing under {GUARDRAIL_DIR}/. "
                    "`otari hook setup` starts one."
                )
            )

    target = _composed_guardrail_id(files, root) if guardrail_file_option is None else str(guardrail_file_option)
    try:
        spec = _compose_guardrail(_read_guardrail_files(files, root), target)
    except (GuardrailReadError, PolicyError) as exc:
        raise click.ClickException(str(exc)) from exc

    findings = validate_policy(
        spec,
        judge_gate_limit=_HOOK_JUDGE_MAX_GATES_PER_RUN,
        verifier_gate_limit=_HOOK_CHECK_MAX_GATES_PER_RUN,
        probe_verifier=lambda verifier: _guardrails_probe_verifier(root, verifier),
    )
    # `policy_id` is the composed set's own name where several files compose,
    # which is `target` again; one file declares its own, worth showing.
    declared = "" if spec.policy_id == target else f" {spec.policy_id},"
    composed = f" composed from {len(files)} files," if len(files) > 1 else ""
    click.echo(f"{target}:{declared}{composed} {len(spec.gates)} gate(s), schema {spec.schema_version}.")
    for finding in findings:
        in_file = spec.gate_sources.get(finding.gate_id or "")
        where = f"{finding.gate_id}{f' ({in_file})' if in_file else ''}: " if finding.gate_id is not None else ""
        click.secho(
            f"  {finding.severity}: {where}{finding.message}",
            fg="red" if finding.severity == "error" else "yellow",
        )
    errors = sum(1 for finding in findings if finding.severity == "error")
    warnings = len(findings) - errors
    click.echo(f"{errors} error(s), {warnings} warning(s).")

    for command in dry_run_commands:
        _guardrails_dry_run(
            spec,
            heading=f"PreToolUse, Bash: {command}",
            paths=[],
            path_source="pre_tool_use.command",
            commands=[command],
            command_scope="call",
        )
    checks_reads = _policy_checks_reads(spec)
    dry_run_targets = [_guardrails_repo_relative(root, path) for path in dry_run_paths]
    for edit_target, working_tree in dry_run_targets:
        _guardrails_dry_run(
            spec,
            heading=f"PreToolUse, Edit/Write: {edit_target}",
            paths=[edit_target],
            path_source="pre_tool_use.edit_target",
            commands=[],
            command_scope="call",
        )
        # Conditional, unlike the Edit/Write moment above, so a guardrail with
        # no read gate prints exactly what it printed before this moment
        # existed. A path is dry-run at every moment its own guardrail can
        # actually see it, and for a guardrail with no read gate that is not
        # one of them.
        if checks_reads:
            _guardrails_dry_run(
                spec,
                heading=f"PreToolUse, Read: {edit_target}",
                paths=[edit_target],
                path_source="pre_tool_use.read_target",
                commands=[],
                command_scope="call",
            )
    if dry_run_commands or dry_run_paths:
        _guardrails_dry_run(
            spec,
            heading=(
                f"Stop, the finished turn: {len(dry_run_targets)} changed path(s), {len(dry_run_commands)} command(s)"
            ),
            paths=[working_tree for _edit_target, working_tree in dry_run_targets],
            path_source="stop.working_tree",
            commands=list(dry_run_commands),
            command_scope="session",
        )

    if errors or (strict and warnings):
        raise SystemExit(1)
