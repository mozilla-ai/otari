"""Pure gate evaluators. No filesystem, git, network, or clock access.

Each evaluator takes a gate spec and evidence already supplied by the caller
and returns a ``GateResult``. Evidence the caller could not supply is ``None``,
never an empty collection standing in for "nothing changed": an evaluator
cannot tell those apart, and conflating them would let missing evidence read
as a clean result.
"""

from __future__ import annotations

import re
import shlex

from gateway.agent_runtime.domain.types import (
    ChangedPathEvidence,
    ChangedPathGate,
    CommandEvidence,
    CommandMatchGate,
    GateResult,
    Outcome,
)

# Tokens that separate one simple command from the next within a shell
# command line. Only recognized as a *whole token* (see _command_segments);
# _normalize_separators is what guarantees one glued to a word ("a&&b",
# "npm install;") is still its own token by the time this is consulted.
# A subshell's parentheses are separators too: what they enclose is a command
# of its own, and "(npm install)" must reach _contains_subsequence with "npm"
# at its segment's head, not glued to the paren.
_COMMAND_SEPARATORS = frozenset({"&&", "||", ";", "|", "&", "(", ")"})

# Bash's own metacharacter set: the unquoted characters that end a word, so
# a `#` directly after one still starts a comment and a separator directly
# after one is still a separator. Space and tab are handled by str.isspace.
_METACHARACTERS = frozenset("|&;()<>")

# What _normalize_separators pads with whitespace, longest first so "&&" is
# never read as two "&". A case arm's ";;" needs no entry of its own: padding
# each ";" gives two separators and an empty segment, which is dropped.
_PADDED_OPERATORS = ("&&", "||", ";", "|", "&", "(", ")")

# Any character that could begin one of the above, or a bare newline, so
# _normalize_separators can rule out the common case with one C-level scan
# instead of a Python loop over every character.
_SEPARATOR_SCAN = re.compile(r"[\n|&;()]")

# The same padding with no quote awareness at all, for the fallback path in
# _command_segments, which is reached only by a command shlex could not parse
# and is already documented as degraded.
_BLIND_SEPARATOR_PAD = re.compile(r"&&|\|\||;|\||(?<![<>])&(?!>)|\(|\)")


def _basename(token: str) -> str:
    """The final path component of `token`, or `token` itself if that's empty.

    `token.rsplit("/", 1)[-1]` already returns `token` unchanged when it has
    no `/`; the `or token` fallback only matters for the rare token that
    *is* a path but ends in `/` (e.g. a bare `/usr/bin/`), where the split
    would otherwise silently produce `""`.
    """
    return token.rsplit("/", 1)[-1] or token


def _segment_matches(pattern: str, text: str) -> bool:
    """Match `*` (zero or more characters) within one path segment against text.

    Splits on `*` into literal chunks and locates each one with `str.find`/
    `str.startswith`/`str.endswith`, never searching backward: CPython
    implements string search with the two-way algorithm, bounded at
    O(len(haystack) + len(needle)), so one call here costs
    O(len(pattern) + len(text)) regardless of how many `*` the pattern has
    or how close `text` comes to matching without succeeding. A prior
    two-pointer version of this function looked linear but wasn't: on a
    mismatch it rewound to just after the last `*` and rescanned the literal
    that followed from scratch, which is O(len(pattern) * len(text)) in the
    adversarial case (a pattern like `"*" + "a" * 2000 + "b"` against text
    with no trailing `b` measured in the seconds against a handful of paths).
    It also had a correctness bug in the same code path: it consumed a text
    character the instant it saw `*`, so `*a` failed to match the single
    character `a` (a false negative that would have let a forbidden path
    through undetected). Splitting into chunks up front avoids both: each
    chunk is found once, matching zero-or-more is just an empty chunk
    contributing nothing, and nothing is ever rescanned.
    """
    chunks = pattern.split("*")
    if len(chunks) == 1:
        return pattern == text

    first, *middle, last = chunks
    if first and not text.startswith(first):
        return False
    if last and not text.endswith(last):
        return False

    # The room left for the middle chunks once `first` and `last` are
    # reserved. If reserving both leaves no room (or a negative amount),
    # nothing arranges the remaining chunks, even all-empty ones, so a
    # length check up front avoids a `find` call in an invalid range.
    start, end = len(first), len(text) - len(last)
    if start > end:
        return False

    position = start
    for chunk in middle:
        if not chunk:
            continue
        found = text.find(chunk, position, end)
        if found == -1:
            return False
        position = found + len(chunk)
    return True


def _segments_match(pattern_segments: list[str], path_segments: list[str]) -> bool:
    """Match glob path segments against path segments; `**` crosses directories.

    Non-recursive: `domain/policy.py` caps a glob at one standalone `**`
    segment (`_MAX_DOUBLE_STAR_PER_GLOB`), so its position is exactly
    determined by `.index()`, and the segments before and after it match
    the corresponding number of segments at the start and end of the path
    directly, with no need to try more than one split point. Trying every
    split point (recursing once per possible position) is what a second
    `**` in the same glob would require, which is why the parser bounds it
    instead of this function trying to stay safe with more than one.
    """
    if "**" not in pattern_segments:
        return len(pattern_segments) == len(path_segments) and all(
            _segment_matches(p, s) for p, s in zip(pattern_segments, path_segments)
        )

    star_index = pattern_segments.index("**")
    before, after = pattern_segments[:star_index], pattern_segments[star_index + 1 :]
    # ** must consume at least one path segment: a bare directory is never,
    # on its own, a changed path in Git's output.
    if len(path_segments) < len(before) + len(after) + 1:
        return False
    head = path_segments[: len(before)]
    tail = path_segments[len(path_segments) - len(after) :] if after else []
    return all(_segment_matches(p, s) for p, s in zip(before, head)) and all(
        _segment_matches(p, s) for p, s in zip(after, tail)
    )


def _matches_any(path: str, patterns: tuple[str, ...]) -> str | None:
    path_segments = path.split("/")
    for pattern in patterns:
        if _segments_match(pattern.split("/"), path_segments):
            return pattern
    return None


def _strip_shell_comment(command: str) -> str:
    """Remove every shell comment, at a real POSIX word boundary.

    A `#` starts a comment only at the start of a word (the start of the
    command, or right after unquoted whitespace or one of Bash's own
    metacharacters, `_METACHARACTERS`) and only outside any quoting. The
    metacharacters matter as much as the whitespace: `ls;# npm install` is
    entirely a comment to Bash, and treating only whitespace as ending a
    word left the commented-out text to be matched as if it were a real
    command. `shlex`'s own `comments=True` is not used here: it treats *any*
    `#` as starting a comment, even mid-word, which is not what a shell does
    (`echo a#b` prints `a#b`, not `a`) and is not safe for this purpose: a
    URL fragment or a `--flag=value#123` mid-command would silently swallow
    everything after it, including a genuinely separate, later command
    joined by `&&`/`;`/`|`. `git commit -m "fix #123"` (a `#` inside a
    quoted argument) is not a comment either way, POSIX or `comments=True`;
    what differs is exactly this word-boundary rule.

    A comment extends only to the end of its own physical line, not to the
    end of the whole string: a multi-line command's own earlier comment
    (`# a note\nnpm install`) must not swallow a real, later command on the
    next line. The newline itself is kept (shlex already treats it as
    ordinary whitespace), so scanning continues normally right after it,
    including into a comment of its own on that next line.

    A backslash escapes the character right after it wherever it appears
    outside single quotes (unquoted, or inside double quotes), so an
    escaped double quote cannot end the quote it is inside, and an escaped
    `#` cannot start a comment. This is a deliberate over-approximation of
    the narrower real rule for what a backslash escapes inside double
    quotes (`$`, `` ` ``, `"`, `\\`, or a newline): it only matters here for
    whether a character is a real closing `"` or a real comment `#`, and
    treating any other escaped character as "not that" changes nothing.

    Bash's ANSI-C quoting, `$'...'`, is not a plain single-quoted string:
    backslash escapes are active inside it, so an escaped apostrophe (`\\'`)
    is literal content, not the closing quote, unlike a real `'...'`. Left
    to `shlex` (which knows only POSIX quoting), a closing `\\'` reads as
    real, and `shlex.split` then either raises on the now-unbalanced
    trailing quote or, before that, this function treated it the same as a
    plain single quote closing, mistaking whatever followed for a fresh,
    unquoted word, `#` included. Rewritten here into an equivalent plain
    `'...'` shlex can already tokenize (an embedded, escaped apostrophe
    becomes close-quote, escaped-quote, reopen-quote, `'\\''`), so a
    forbidden token inside a `$'...'` argument is found the same as inside
    any other quoting.
    """
    quote: str | None = None
    at_word_start = True
    escaped = False
    result: list[str] = []
    index = 0
    length = len(command)
    while index < length:
        char = command[index]
        if escaped:
            escaped = False
            result.append(char)
            index += 1
            continue
        if quote == "'":
            # Nothing is special inside single quotes, not even backslash.
            result.append(char)
            if char == "'":
                quote = None
            index += 1
            continue
        if quote is None and char == "$" and command[index + 1 : index + 2] == "'":
            result.append("'")
            index += 2
            at_word_start = False
            while index < length:
                inner = command[index]
                if inner == "\\" and index + 1 < length:
                    escaped_char = command[index + 1]
                    result.append("'\\''" if escaped_char == "'" else escaped_char)
                    index += 2
                    continue
                if inner == "'":
                    result.append("'")
                    index += 1
                    break
                result.append(inner)
                index += 1
            continue
        if char == "\\":
            escaped = True
            at_word_start = False
            result.append(char)
            index += 1
            continue
        if quote == '"':
            result.append(char)
            if char == '"':
                quote = None
            index += 1
            continue
        if char in "'\"":
            quote = char
            at_word_start = False
            result.append(char)
            index += 1
            continue
        if char == "#" and at_word_start:
            newline_index = command.find("\n", index)
            if newline_index == -1:
                break
            result.append("\n")
            index = newline_index + 1
            at_word_start = True
            continue
        at_word_start = char.isspace() or char in _METACHARACTERS
        result.append(char)
        index += 1
    return "".join(result)


def _operator_at(command: str, index: int) -> str | None:
    """The separator starting at `index`, or None if no separator starts there.

    Callers must already have established that `index` is outside quoting and
    not escaped; this looks only at the characters themselves.
    """
    if command[index] == "&" and (command[index - 1 : index] in ("<", ">") or command[index + 1 : index + 2] == ">"):
        return None  # Part of a redirect (2>&1, &>out), not a separator.
    return next((operator for operator in _PADDED_OPERATORS if command.startswith(operator, index)), None)


def _normalize_separators(command: str) -> str:
    """Pad every unquoted separator with whitespace, newlines included.

    `_command_segments` only recognizes a separator as a *whole token*, and
    shlex only isolates one when whitespace surrounds it (`"a;b"` stays one
    token, `"a ; b"` becomes three). Without this padding the gate failed
    open on any command whose operator was written without spaces, which is
    how people actually type the commonest of them: `npm install;` tokenized
    to `["npm", "install;"]`, matching no phrase, and `(npm install)` to
    `["(npm", "install)"]`, matching none either.

    A `&` belonging to a redirect (`2>&1`, `&>out`) is left alone: it joins a
    file descriptor to a redirect target rather than ending a command, and
    padding it would split one command into two segments at a point no shell
    does.

    A bare newline becomes a `;` for the same reason. Left as the ordinary
    whitespace shlex treats it as, `git\\npush` (two one-word commands)
    merges into one segment `["git", "push"]` and falsely matches the
    two-word phrase naming one command. It is also what puts a command at
    its own segment's head, where `_contains_subsequence` applies the
    basename equivalence: `echo hi\\n/usr/bin/npm install`, unsplit, has
    `/usr/bin/npm` at a non-zero position, where that equivalence does not.

    Meant to run on `_strip_shell_comment`'s own output, which has already
    rewritten Bash's `$'...'` quoting into plain `'...'`, so this only needs
    to track plain single/double quotes and backslash escaping, not ANSI-C
    quoting a second time. A quoted newline (inside `'...'` or `"..."`) is
    real content, not a boundary, and is left untouched. A backslash before
    a newline, outside single quotes, is a line continuation: a real shell
    deletes both characters, joining the two physical lines with nothing
    between them, so this does too, rather than leaving the pair for shlex.
    Left alone, shlex only treats that escaped newline as "not a word
    break," not as deleted: `shlex.split("git \\\npush --force")` gives
    `["git", "\\npush", "--force"]`, a literal newline still embedded in
    the second token, which then never equals the plain word `push` a
    forbidden phrase names, letting `git push --force` typed across two
    continued lines pass a gate forbidding exactly that.
    """
    if not _SEPARATOR_SCAN.search(command):
        # A command with nothing to pad, cheap to rule out up front: one
        # C-level scan, versus the character-by-character Python loop below
        # running to the end of the string on top of `_strip_shell_comment`'s
        # own pass over it.
        return command
    quote: str | None = None
    result: list[str] = []
    index = 0
    length = len(command)
    while index < length:
        char = command[index]
        if quote == "'":
            result.append(char)
            if char == "'":
                quote = None
            index += 1
            continue
        if char == "\\":
            if command[index + 1 : index + 2] == "\n":
                index += 2  # Delete the line continuation entirely.
                continue
            if index + 1 < length:
                result.append(char)
                result.append(command[index + 1])
                index += 2
                continue
            result.append(char)
            index += 1
            continue
        if quote == '"':
            result.append(char)
            if char == '"':
                quote = None
            index += 1
            continue
        if char in "'\"":
            quote = char
            result.append(char)
            index += 1
            continue
        if char == "\n":
            result.append(" ; ")
            index += 1
            continue
        operator = _operator_at(command, index)
        if operator is not None:
            result.append(f" {operator} ")
            index += len(operator)
            continue
        result.append(char)
        index += 1
    return "".join(result)


def _command_segments(command: str) -> list[list[str]]:
    """Split a command into simple-command segments, each already tokenized.

    Strips a trailing comment, pads every unquoted separator and bare newline
    so shlex will isolate it (`_normalize_separators`), then tokenizes with
    `shlex` (POSIX quoting rules), then splits the resulting token list on
    any token that is exactly one of `_COMMAND_SEPARATORS`: a quoted argument
    that happens to contain that text, like `"a && b"`, survives as a single
    token from shlex and is never mistaken for a separator, since this only
    looks at whole tokens, never substrings of one.

    Each segment's own first token, the command actually being invoked for
    that segment, is later compared by path basename rather than literally
    (`_contains_subsequence`), so a path-qualified invocation of an
    executable matches a phrase naming it unqualified and vice versa. That
    equivalence is applied at comparison time, not by rewriting a token
    here: this function's own output is always the literal tokens a
    caller's command actually contained.

    A command shlex cannot tokenize even after comment-stripping (an
    unbalanced quote outside any comment) falls back to a plain whitespace
    split. Keeping it as one opaque token instead was a silent fail-open:
    `npm install "unterminated` never equals the single-token phrase `npm`,
    so a required gate forbidding it reported `pass`. Reporting the command
    as unreadable instead is worse: a Bash call carrying a heredoc of Python
    or SQL is routinely unparseable to shlex, and an `unknown` there blocks
    every ordinary tool call rather than the forbidden ones.

    The whitespace split is deliberately degraded, not equivalent: it cannot
    tell a quoted argument from a bare word, so a forbidden phrase inside a
    quoted string in an already-unparseable command matches where it would
    not have in a parseable one. That trade is the right way round. It costs
    a false positive on a command that was malformed to begin with, and it
    buys back detection on the shape an evasion would actually take.
    Separator normalization degrades the same way here: `_normalize_separators`
    is quote-aware, but a command that reaches this branch has an unbalanced
    quote, which leaves its state machine believing every character from the
    opening quote onward is still inside it, so it pads none of them. This
    branch instead pads unconditionally (`_BLIND_SEPARATOR_PAD`), on the
    original comment-stripped text rather than that already-attempted,
    possibly-incomplete normalization.

    An empty segment (two separators back to back, or one at either end,
    e.g. `";" * n`) is dropped rather than returned: `_contains_subsequence`
    can never match a non-empty forbidden phrase against it (every phrase is
    non-empty by construction, `domain.policy` rejects one that isn't), so
    keeping it around costs a phrase-count's worth of guaranteed-`False`
    comparisons for nothing. A caller-controlled string built from many
    separators and no real content, ";" * 500 against 500 forbidden
    phrases, measured ~25,000,000 such comparisons and ~1.1s before this.
    """
    stripped = _strip_shell_comment(command)
    try:
        tokens = shlex.split(_normalize_separators(stripped), posix=True)
    except ValueError:
        # Same blind, non-quote-aware treatment as the newline replacement
        # right after it: strip a line continuation first (a backslash
        # right before a newline), or the newline that follows it gets the
        # bare-newline treatment instead and turns one continued line into
        # two segments that were never meant to be split.
        tokens = _BLIND_SEPARATOR_PAD.sub(
            lambda match: f" {match.group()} ", stripped.replace("\\\n", "").replace("\n", " ; ")
        ).split()

    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in _COMMAND_SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def tokenize_phrase(phrase: str) -> list[str]:
    """Tokenize one forbidden phrase, comment-aware and consistent with commands.

    Shared by evaluation (`evaluate_command_match`), parse-time validation
    (`domain.policy`), and the Hook Server route's cost estimate, so a
    phrase is judged the same way everywhere it is tokenized. Raises
    `ValueError` on an unbalanced quote outside any comment, exactly what
    `domain.policy` already rejects at parse time for a gate's own forbidden
    phrases; a caller that has not gone through that validation gets the
    same exception a raw `shlex.split` would.
    """
    return shlex.split(_strip_shell_comment(phrase), posix=True)


def tokenize_phrases(phrases: tuple[str, ...]) -> dict[str, list[str]]:
    """Tokenize every forbidden phrase once, for every gate and the cost estimate to share.

    The phrase-side counterpart to `tokenize_commands`, and shared the same
    way (`phrase_cache`): without it hooks.py tokenizes every phrase for its
    cost estimate, discards the result, and each `evaluate_command_match`
    call then tokenizes that gate's phrases again. Bounded by policy size
    rather than by evidence, so the cost is small either way; sharing it
    keeps the estimate and the evaluation reading from one set of tokens
    instead of two computed the same way in two places.
    """
    return {phrase: tokenize_phrase(phrase) for phrase in phrases}


def tokenize_commands(commands: tuple[str, ...]) -> dict[str, list[list[str]]]:
    """Tokenize every command once, for every command_match gate to share.

    `evaluate_command_match` is called once per command_match gate against
    the same evidence; without this, each call would re-tokenize every
    command from scratch, multiplying `shlex`'s per-character cost (real,
    not negligible: ~100-350ns/char regardless of content) by the number of
    gates. A request well within every per-request budget in hooks.py, since
    none of them accounted for that multiplication, measured several seconds
    of synchronous blocking before this existed. Pass the result to every
    `evaluate_command_match` call for one request via `segment_cache`.
    """
    return {command: _command_segments(command) for command in commands}


def _contains_subsequence(segment: list[str], phrase: list[str]) -> bool:
    """Whether `phrase`'s tokens appear, in order and unbroken, inside `segment`.

    Every position compares literally except one: when a candidate window
    starts at `segment[0]`, that position is the executable actually
    invoked for this segment, so it is compared to `phrase[0]` by path
    basename rather than by literal equality. A policy author writes a
    forbidden phrase against the name they would type (`npm install`), and
    a path-qualified invocation of the same executable (`/usr/bin/npm
    install`) is not a different command; the same equivalence also lets a
    path-qualified *phrase* (`./scripts/release.sh`) keep matching a
    command that invokes that exact path, which comparing only the segment
    side against a literal phrase would silently stop doing the moment the
    phrase's own spelling no longer equals the normalized token. Comparing
    both sides by basename, rather than rewriting either one ahead of time,
    is what keeps every direction working: bare phrase vs. qualified
    command, qualified phrase vs. bare command, and qualified phrase vs. the
    identical qualified command.

    Everywhere else, including `segment[0]` itself when the window instead
    starts later (an earlier phrase token already matched a prefix
    command), a path is real content and compared literally: `sudo
    /usr/bin/npm install` does not match `npm install`, since the actual
    executable sits at `segment[1]`, not `segment[0]`, for that segment; and
    `npm install ./local-package` does not match a phrase naming just
    `local-package`, since an argument's path is not a command position.
    """
    if not phrase or len(phrase) > len(segment):
        return False
    for start in range(len(segment) - len(phrase) + 1):
        head_matches = (
            _basename(segment[0]) == _basename(phrase[0]) if start == 0 else segment[start] == phrase[0]
        )
        if head_matches and segment[start + 1 : start + len(phrase)] == phrase[1:]:
            return True
    return False


def evaluate_command_match(
    gate: CommandMatchGate,
    evidence: CommandEvidence | None,
    *,
    segment_cache: dict[str, list[list[str]]] | None = None,
    phrase_cache: dict[str, list[str]] | None = None,
) -> GateResult:
    """Fail when a submitted command matches one of the gate's forbidden phrases.

    `segment_cache` and `phrase_cache` are both optional and default to
    tokenizing locally, so a caller evaluating a single gate in isolation (a
    unit test, a one-off check) needs nothing extra. A caller evaluating
    several command_match gates against the same evidence, like hooks.py's
    ``check_policy``, should build each cache once (``tokenize_commands``,
    ``tokenize_phrases``) and pass the same dicts to every call, so
    tokenizing costs once per request rather than once per gate.
    """
    if evidence is None:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.UNKNOWN,
            message="Command evidence was not submitted.",
        )

    if not evidence.commands:
        # An explicitly empty commands list is not "checked, none forbidden":
        # a caller submits it for exactly the events that never carry command
        # evidence at all (Claude Code's Stop event, or a PreToolUse call for
        # an edit tool rather than Bash; see docs/agent-gates.md). Reporting
        # PASS there would read as a check that ran and found nothing, when
        # this gate never had anything to check. not_applicable is the
        # non-blocking outcome that says so honestly.
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.NOT_APPLICABLE,
            message="No commands were submitted to check.",
        )

    # A cache built for a different gate is tolerated rather than a KeyError:
    # the parameter is optional and a caller that passes a partial one should
    # get a slower evaluation, not a 500.
    phrases_by_text = phrase_cache if phrase_cache is not None else {}
    forbidden_phrases = [
        phrases_by_text.get(phrase) or tokenize_phrase(phrase) for phrase in gate.forbidden
    ]
    segments_by_command = segment_cache if segment_cache is not None else {}

    matched = sorted(
        command
        for command in evidence.commands
        if any(
            _contains_subsequence(segment, phrase)
            for segment in (segments_by_command.get(command) or _command_segments(command))
            for phrase in forbidden_phrases
        )
    )
    if matched:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.FAIL,
            message=gate.message,
            detail=", ".join(matched),
        )
    return GateResult(
        gate_id=gate.id,
        enforcement=gate.enforcement,
        outcome=Outcome.PASS,
        message="No forbidden commands run.",
    )


def evaluate_changed_path(gate: ChangedPathGate, evidence: ChangedPathEvidence | None) -> GateResult:
    """Fail when a changed path matches one of the gate's forbidden globs."""
    if evidence is None:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.UNKNOWN,
            message="Change evidence was not submitted.",
        )

    if not evidence.changed_paths:
        # Mirrors evaluate_command_match: a caller submits an empty list for
        # exactly the events that carry no path evidence at all (a PreToolUse
        # call for Bash rather than an edit tool), and PASS there reads as a
        # check that ran and found nothing when this gate never had anything
        # to check. Both outcomes are non-blocking, so this changes what is
        # reported rather than what is enforced.
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.NOT_APPLICABLE,
            message="No changed paths were submitted to check.",
        )

    matched = sorted(path for path in evidence.changed_paths if _matches_any(path, gate.forbidden) is not None)
    if matched:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.FAIL,
            message=gate.message,
            detail=", ".join(matched),
        )
    return GateResult(
        gate_id=gate.id,
        enforcement=gate.enforcement,
        outcome=Outcome.PASS,
        message="No forbidden paths changed.",
    )
