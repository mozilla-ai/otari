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
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

from otari_agent.domain.types import (
    CheckEvidence,
    CommandEvidence,
    CommandGate,
    CommandIfChangedGate,
    GateResult,
    JudgeEvidence,
    JudgeGate,
    Outcome,
    PathEvidence,
    PathGate,
    VerifierGate,
)

# Tokens that separate one simple command from the next within a shell
# command line. Only recognized as a *whole token* (see _command_segments);
# _normalize_separators is what guarantees one glued to a word ("a&&b",
# "npm install;") is still its own token by the time this is consulted.
# A subshell's parentheses are separators too: what they enclose is a command
# of its own, and "(npm install)" must reach _contains_subsequence with "npm"
# at its segment's head, not glued to the paren.
# Public because domain/validation.py needs the same set to tell an author
# that a phrase carrying one of these can never match: a phrase matches a
# contiguous token run *within* one segment, and these are what end a segment.
COMMAND_SEPARATORS = frozenset({"&&", "||", ";", "|", "&", "(", ")"})

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

# The unquoted openers of the pairs `_NoRedirectionPairs` tracks, each with its closer, longest first.
_NO_REDIRECTION_OPENERS = (("$((", "))"), ("((", "))"), ("$[", "]"), ("${", "}"), ("=(", ")"))

# Every character that can open or close one of those pairs, so most characters skip the pair check.
_PAIR_CHARACTERS = frozenset("$(=[)]}")

# The characters a backslash escapes inside double quotes, where it is otherwise literal.
_ESCAPED_IN_DOUBLE_QUOTES = frozenset('$`"\\\n')


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


def matched_changed_paths(patterns: tuple[str, ...], changed_paths: tuple[str, ...]) -> tuple[str, ...]:
    """Every one of `changed_paths` that matches any of `patterns`, sorted.

    `patterns` is the same repo-relative POSIX glob grammar `PathGate.forbidden`/
    `CommandIfChangedGate.when_changed` use. Shared, rather than reimplemented, by
    `evaluate_judge`'s own `when_changed` applicability check and by `cli.py`'s local
    judge-gate filtering (skipping a `claude -p` call for a gate that plainly does not
    apply yet), so a path is judged against the same grammar wherever it's checked.
    """
    return tuple(sorted(path for path in changed_paths if _matches_any(path, patterns) is not None))


@dataclass(frozen=True, slots=True)
class _Heredoc:
    """A heredoc that a command line opens, with its body on the lines after it.

    `delimiter` is the terminator word after quote removal.
    `strips_tabs` is true for `<<-`, where leading tabs on the terminator line do not count.
    `joins_lines` is true for an unquoted delimiter, where Bash joins a body line that ends in a backslash to the next.
    """

    delimiter: str
    strips_tabs: bool
    joins_lines: bool


class _NoRedirectionPairs:
    """The matched pairs open at a point in a command, inside which Bash reads `<<` as a shift, not a heredoc.

    These are arithmetic, a parameter expansion, a subscript and a compound array assignment.
    A pair that never closes stays open, so no later `<<` opens a heredoc.
    """

    def __init__(self) -> None:
        self._closers: list[str] = []

    @property
    def any_open(self) -> bool:
        """Whether any pair is open, which rules out a heredoc."""
        return bool(self._closers)

    def advance(self, command: str, index: int, at_word_start: bool) -> int:
        """Open or close a pair at the unquoted `index`, and return how many characters that took, or 0."""
        if self._closers and command.startswith(self._closers[-1], index):
            return len(self._closers.pop())
        for opener, closer in _NO_REDIRECTION_OPENERS:
            if command.startswith(opener, index):
                self._closers.append(closer)
                return len(opener)
        if command[index] == "[" and not at_word_start:
            self._closers.append("]")
            return 1
        if command[index] == "(" and self._closers:
            self._closers.append(")")
            return 1
        return 0


def _read_heredoc_operator(command: str, index: int) -> tuple[_Heredoc, int] | None:
    """Return the heredoc that a `<<` at `index` opens, and the index past its delimiter word.

    Return None when there is no delimiter word, or when the word uses `$` or a backtick outside single quotes.
    Bash gives `$'...'` and `$"..."` delimiters a meaning this reader does not model.

    NOTE: Callers must pass the index of a `<<` that does not begin a `<<<` here-string.
    """
    position = index + 2
    strips_tabs = command.startswith("-", position)
    if strips_tabs:
        position += 1
    while position < len(command) and command[position] in " \t":
        position += 1
    word_start = position
    delimiter: list[str] = []
    quote: str | None = None
    quoted = False
    while position < len(command):
        char = command[position]
        if quote == "'":
            if char == "'":
                quote = None
            else:
                delimiter.append(char)
        elif char in "$`":
            return None
        elif quote == '"':
            if char == '"':
                quote = None
            elif char == "\\" and command[position + 1 : position + 2] in _ESCAPED_IN_DOUBLE_QUOTES:
                position += 1
                delimiter.append(command[position])
            else:
                delimiter.append(char)
        elif char in "'\"":
            quote = char
            quoted = True
        elif char == "\\" and position + 1 < len(command):
            position += 1
            quoted = True
            delimiter.append(command[position])
        elif char.isspace() or char in _METACHARACTERS:
            break
        else:
            delimiter.append(char)
        position += 1
    if quote is not None or position == word_start:
        return None
    return _Heredoc("".join(delimiter), strips_tabs, joins_lines=not quoted), position


class _HeredocTerminators:
    """Find where heredoc bodies end in one command, in time linear in the command's length."""

    def __init__(self, command: str) -> None:
        self._command = command
        self._line_starts: dict[str, list[int]] = {}
        self._tab_stripped_line_starts: dict[str, list[int]] = {}
        self._continuations: list[int] = []
        position = 0
        for line in command.split("\n"):
            self._line_starts.setdefault(line, []).append(position)
            self._tab_stripped_line_starts.setdefault(line.lstrip("\t"), []).append(position)
            if line.endswith("\\"):
                self._continuations.append(position + len(line))
            position += len(line) + 1

    def bodies_end(self, start: int, heredocs: Sequence[_Heredoc]) -> int:
        """Return the index past the bodies of `heredocs`, which follow each other from the line at `start`.

        Stop at the start of a body with no certain terminator line, so that body stays command text.
        A body that Bash would join across a line continuation has no certain terminator line.
        """
        position = start
        for heredoc in heredocs:
            body_end = self._body_end(heredoc, position)
            if body_end is None:
                return position
            position = body_end
        return position

    def _body_end(self, heredoc: _Heredoc, start: int) -> int | None:
        """Return the index past the body of `heredoc` from the line at `start`, or None if its end is not certain."""
        line_starts = (self._tab_stripped_line_starts if heredoc.strips_tabs else self._line_starts).get(
            heredoc.delimiter, []
        )
        found = bisect_left(line_starts, start)
        if found == len(line_starts):
            return None
        terminator_start = line_starts[found]
        if heredoc.joins_lines and bisect_left(self._continuations, start) < bisect_left(
            self._continuations, terminator_start
        ):
            return None
        line_end = self._command.find("\n", terminator_start)
        return len(self._command) if line_end == -1 else line_end + 1


def _strip_comments_and_heredoc_bodies(command: str) -> str:
    """Remove the text Bash does not read as command words: comments and heredoc bodies.

    A `#` starts a comment only where it begins an unquoted word, after whitespace or one of `_METACHARACTERS`.
    A comment ends with its own line.
    A heredoc body is the data its command reads, so it is removed with its terminator line.
    A heredoc with no certain terminator line is kept as command text.
    A `<<` inside arithmetic, a parameter expansion, a subscript or a compound array assignment opens no heredoc.
    A backslash outside single quotes escapes any next character, which is wider than Bash's rule inside double quotes.
    Bash's ANSI-C quoting, `$'...'`, is rewritten into the equivalent plain `'...'`, which shlex can tokenize.

    NOTE: `shlex`'s `comments=True` is not used, because it starts a comment at any `#`, mid-word included.
    """
    quote: str | None = None
    at_word_start = True
    escaped = False
    pending_heredocs: list[_Heredoc] = []
    terminators: _HeredocTerminators | None = None
    pairs = _NoRedirectionPairs()
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
            index = newline_index
            continue
        if char == "\n":
            result.append(char)
            index += 1
            if pending_heredocs:
                if terminators is None:
                    terminators = _HeredocTerminators(command)
                index = terminators.bodies_end(index, pending_heredocs)
                pending_heredocs = []
            at_word_start = True
            continue
        if char in _PAIR_CHARACTERS and (pair_length := pairs.advance(command, index, at_word_start)):
            result.append(command[index : index + pair_length])
            index += pair_length
            # Only an opening parenthesis ends a word, so `$((1))#x` stays one word, as in Bash.
            at_word_start = command[index - 1] == "("
            continue
        if char == "<" and command.startswith("<<", index):
            is_here_string = command.startswith("<<<", index)
            opened = None if is_here_string or pairs.any_open else _read_heredoc_operator(command, index)
            if opened is None:
                # NOTE: The whole operator is consumed, so its second `<` is never read as an operator of its own.
                operator_length = 3 if is_here_string else 2
                result.append(command[index : index + operator_length])
                index += operator_length
                at_word_start = True
                continue
            heredoc, word_end = opened
            pending_heredocs.append(heredoc)
            result.append(command[index:word_end])
            index = word_end
            at_word_start = False
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

    A quoted newline is content and stays.
    A line continuation (a backslash before an unquoted newline) is deleted, as Bash deletes it.
    shlex would otherwise keep that newline inside the next token, which then matches no phrase.

    NOTE: Callers must pass the output of `_strip_comments_and_heredoc_bodies`, because this tracks only plain quotes.
    """
    if not _SEPARATOR_SCAN.search(command):
        # One C-level scan rules out a command with nothing to pad.
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

    Comments and heredoc bodies belong to no segment.
    A separator in `COMMAND_SEPARATORS` or a bare newline ends a segment, and a quoted one such as `"a && b"` does not.

    Each segment's own first token, the command actually being invoked for
    that segment, is later compared by path basename rather than literally
    (`_contains_subsequence`), so a path-qualified invocation of an
    executable matches a phrase naming it unqualified and vice versa. That
    equivalence is applied at comparison time, not by rewriting a token
    here: this function's own output is always the literal tokens a
    caller's command actually contained.

    A command shlex cannot tokenize, such as one with an unbalanced quote, falls back to a whitespace split.
    The split still finds a phrase spelled as bare words, which one opaque token would hide.
    An `unknown` result instead would block every unparseable call, not only the forbidden ones.

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
    stripped = _strip_comments_and_heredoc_bodies(command)
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
        if token in COMMAND_SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def tokenize_phrase(phrase: str) -> list[str]:
    """Tokenize one forbidden phrase, comment-aware and consistent with commands.

    Shared by evaluation (`evaluate_command`), parse-time validation
    (`domain.policy`), and the Hook Server route's cost estimate, so a
    phrase is judged the same way everywhere it is tokenized. Raises
    `ValueError` on an unbalanced quote outside any comment, exactly what
    `domain.policy` already rejects at parse time for a gate's own forbidden
    phrases; a caller that has not gone through that validation gets the
    same exception a raw `shlex.split` would.
    """
    return shlex.split(_strip_comments_and_heredoc_bodies(phrase), posix=True)


def tokenize_phrase_with_separators(phrase: str) -> list[str]:
    """Tokenize one phrase the way a *command* is tokenized, with separators isolated.

    Deliberately not what :func:`tokenize_phrase` does, and the difference is
    the point. Matching leaves a separator glued to its word, because a phrase
    is matched as a contiguous token run *within* one segment and a segment
    never contains a separator to match against. Detecting one needs the
    opposite: the same normalization `_command_segments` applies, so that
    `"npm install;"` yields a `;` token rather than an `install;` token that
    silently equals nothing a command can produce.

    Quote-aware through `_normalize_separators`, so `echo "a && b"` keeps its
    argument whole and is not mistaken for a phrase spanning a boundary.
    """
    return shlex.split(_normalize_separators(_strip_comments_and_heredoc_bodies(phrase)), posix=True)


def tokenize_phrases(phrases: tuple[str, ...]) -> dict[str, list[str]]:
    """Tokenize every forbidden phrase once, for every gate and the cost estimate to share.

    The phrase-side counterpart to `tokenize_commands`, and shared the same
    way (`phrase_cache`): without it hooks.py tokenizes every phrase for its
    cost estimate, discards the result, and each `evaluate_command`
    call then tokenizes that gate's phrases again. Bounded by policy size
    rather than by evidence, so the cost is small either way; sharing it
    keeps the estimate and the evaluation reading from one set of tokens
    instead of two computed the same way in two places.
    """
    return {phrase: tokenize_phrase(phrase) for phrase in phrases}


def tokenize_commands(commands: tuple[str, ...]) -> dict[str, list[list[str]]]:
    """Tokenize every command once, for every command gate to share.

    `evaluate_command` is called once per command gate against
    the same evidence; without this, each call would re-tokenize every
    command from scratch, multiplying `shlex`'s per-character cost (real,
    not negligible: ~100-350ns/char regardless of content) by the number of
    gates. A request well within every per-request budget in hooks.py, since
    none of them accounted for that multiplication, measured several seconds
    of synchronous blocking before this existed. Pass the result to every
    `evaluate_command` call for one request via `segment_cache`.
    """
    return {command: _command_segments(command) for command in commands}


def _cached_segments(command: str, segment_cache: dict[str, list[list[str]]]) -> list[list[str]]:
    """Return `command`'s segments from the cache, tokenizing only on a miss; a command with no segments is a hit."""
    segments = segment_cache.get(command)
    return segments if segments is not None else _command_segments(command)


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
        head_matches = _basename(segment[0]) == _basename(phrase[0]) if start == 0 else segment[start] == phrase[0]
        if head_matches and segment[start + 1 : start + len(phrase)] == phrase[1:]:
            return True
    return False


def evaluate_command(
    gate: CommandGate,
    evidence: CommandEvidence | None,
    *,
    segment_cache: dict[str, list[list[str]]] | None = None,
    phrase_cache: dict[str, list[str]] | None = None,
) -> GateResult:
    """Fail when a submitted command matches one of the gate's forbidden phrases.

    `segment_cache` and `phrase_cache` are both optional and default to
    tokenizing locally, so a caller evaluating a single gate in isolation (a
    unit test, a one-off check) needs nothing extra. A caller evaluating
    several command gates against the same evidence, like hooks.py's
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

    if evidence.scope == "session":
        # A forbidden command is judged where it can still be refused: the
        # call that is about to run it. Session-scoped evidence is the whole
        # transcript, which only ever grows, so matching against it would
        # fail every remaining check of the session over one command already
        # refused (or already run, and by then unrunnable in reverse) with no
        # action left that could clear it. The call-scoped check is not
        # weakened by skipping this: `otari hook setup` puts Bash in the
        # PreToolUse matcher precisely when the policy has a command
        # gate, so every command this would have seen was already judged
        # before it ran.
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.NOT_APPLICABLE,
            message="Forbidden commands are checked per call, not over session history.",
        )

    if not evidence.commands:
        # Call-scoped and empty: a PreToolUse call for an edit tool rather
        # than Bash, which never collects command evidence at all (see
        # docs/agent-guardrails.md). Reporting PASS would read as a check that ran
        # and found nothing, when this gate never had anything to check.
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
    forbidden_phrases = [phrases_by_text.get(phrase) or tokenize_phrase(phrase) for phrase in gate.forbidden]
    segments_by_command = segment_cache if segment_cache is not None else {}

    matched = sorted(
        command
        for command in evidence.commands
        if any(
            _contains_subsequence(segment, phrase)
            for segment in _cached_segments(command, segments_by_command)
            for phrase in forbidden_phrases
        )
    )
    if matched:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.FAIL,
            message=gate.failure_message,
            detail=", ".join(matched),
        )
    return GateResult(
        gate_id=gate.id,
        enforcement=gate.enforcement,
        outcome=Outcome.PASS,
        message="No forbidden commands run.",
    )


def evaluate_command_if_changed(
    gate: CommandIfChangedGate,
    path_evidence: PathEvidence | None,
    command_evidence: CommandEvidence | None,
    *,
    segment_cache: dict[str, list[list[str]]] | None = None,
    phrase_cache: dict[str, list[str]] | None = None,
) -> GateResult:
    """Fail when a changed path matches `when_changed` but no command matches `require`.

    Needs both evidence kinds to resolve for real: changed-path evidence to
    know whether the gate applies at all, command evidence to know whether
    the required command ran. Either being absent (``None``, not merely
    empty) resolves ``unknown``, the same as either evaluator alone treats a
    missing evidence list: a check that could not run must never read as one
    that passed. An empty list is not absent evidence; see ``scope`` below.

    Resolves only against ``session``-scoped command evidence. A
    ``PreToolUse`` call submits its own target as ``paths`` and its
    one command (or none, for an edit tool) as ``call``-scoped evidence,
    before the edit itself has even run: the required command cannot have
    run in response to a change that has not happened yet, so failing there
    would block every attempt to edit a ``when_changed``-matched path
    forever, the edit being the very thing blocked. ``call`` scope
    therefore resolves ``not_applicable``, deferring the gate to the
    ``Stop`` event where `otari hook` submits the session's real command
    history (see docs/agent-guardrails.md).

    Under ``session`` scope an empty command list is a real answer rather
    than a missing one, and resolves ``fail``: a session that changed a
    matched path and ran no commands at all did not run the required one.
    Distinguishing that from "this caller collects no command evidence
    here" is exactly what ``CommandEvidence.scope`` exists for; without it
    both arrive as an empty list and the gate has to read the honest
    failure as non-applicable.

    ``segment_cache`` and ``phrase_cache`` both mirror ``evaluate_command``'s
    own parameters: a caller evaluating several command-evidence gates
    against the same evidence (``hooks.py``'s ``check_policy``) builds each
    once (``tokenize_commands``, ``tokenize_phrases``) and passes the same
    dicts to every call, so tokenizing costs once per request rather than
    once per gate.
    """
    if path_evidence is None or command_evidence is None:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.UNKNOWN,
            message="Change or command evidence was not submitted.",
        )

    matched_paths = sorted(path for path in path_evidence.paths if _matches_any(path, gate.when_changed) is not None)
    if not matched_paths:
        # Nothing this gate cares about changed: there is nothing to require
        # a command for, independent of whether any command ran at all.
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.NOT_APPLICABLE,
            message="No changed path matched this gate's when_changed globs.",
        )

    if command_evidence.scope == "call":
        # One tool call's own command cannot answer "did the required
        # command run at some point", and this gate is evaluated on every
        # PreToolUse call against a matched path, before the edit that would
        # need the command has even happened. Deferred to the Stop event,
        # where session-scoped evidence can answer it.
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.NOT_APPLICABLE,
            message="Required-command checks resolve against session history, not one call.",
        )

    if not command_evidence.commands:
        # Session-scoped and empty is a real answer, not a missing one: the
        # session ran no commands at all, so the required one is among them.
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.FAIL,
            message=gate.failure_message,
            detail=", ".join(matched_paths),
        )

    # A cache built for a different gate is tolerated rather than a KeyError,
    # mirroring evaluate_command's own guard: the parameter is optional
    # and a caller that passes a partial one should get a slower evaluation,
    # not a 500.
    phrases_by_text = phrase_cache if phrase_cache is not None else {}
    required_phrases = [phrases_by_text.get(phrase) or tokenize_phrase(phrase) for phrase in gate.require]
    segments_by_command = segment_cache if segment_cache is not None else {}
    satisfied = any(
        _contains_subsequence(segment, phrase)
        for command in command_evidence.commands
        for segment in _cached_segments(command, segments_by_command)
        for phrase in required_phrases
    )
    if satisfied:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.PASS,
            message="A required command ran.",
        )
    return GateResult(
        gate_id=gate.id,
        enforcement=gate.enforcement,
        outcome=Outcome.FAIL,
        message=gate.failure_message,
        detail=", ".join(matched_paths),
    )


def evaluate_path(gate: PathGate, evidence: PathEvidence | None) -> GateResult:
    """Fail when a submitted path matches one of the gate's forbidden globs.

    "Submitted" rather than "changed" because the evidence's own ``source``
    decides which: a path about to be written, a path about to be read, or one
    `git status` reports afterwards. The match is identical in all three; only
    whether this gate asked to see that source differs.
    """
    if evidence is None:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.UNKNOWN,
            message="Path evidence was not submitted.",
        )

    if evidence.source is not None and evidence.source not in gate.runs:
        # Checked before the empty-list branch below on purpose: both resolve
        # NOT_APPLICABLE, so enforcement is identical either way, but "this
        # gate does not look at this moment's evidence" is the more specific
        # and more debuggable reason than "nothing was submitted". This is the
        # line that keeps a `stop.working_tree`-only gate from reading a
        # PreToolUse call as a clean result, and the line that keeps read
        # evidence off a gate that only ever asked about writes.
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.NOT_APPLICABLE,
            message=f"This gate does not run at {evidence.source}.",
        )

    if not evidence.paths:
        # Mirrors evaluate_command: a caller submits an empty list for
        # exactly the events that carry no path evidence at all (a PreToolUse
        # call for Bash rather than an edit or read tool), and PASS there
        # reads as a check that ran and found nothing when this gate never had
        # anything to check. Both outcomes are non-blocking, so this changes
        # what is reported rather than what is enforced.
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.NOT_APPLICABLE,
            message="No paths were submitted to check.",
        )

    matched = sorted(path for path in evidence.paths if _matches_any(path, gate.forbidden) is not None)
    if matched:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.FAIL,
            message=gate.failure_message,
            detail=", ".join(matched),
        )
    return GateResult(
        gate_id=gate.id,
        enforcement=gate.enforcement,
        outcome=Outcome.PASS,
        message="No forbidden paths were submitted.",
    )


def evaluate_judge(gate: JudgeGate, path_evidence: PathEvidence | None, evidence: JudgeEvidence | None) -> GateResult:
    """Relay the caller's own model verdict for this gate; Otari never calls a model itself.

    ``evidence`` being ``None`` outright, before even checking
    ``gate.when_changed``, resolves ``not_applicable``: this is the one
    caller-observable case a `judge` gate needs that no other gate type
    does, an event kind that never runs judge gates at all (``otari hook``
    on `PreToolUse`, which has neither a finished diff nor a transcript to
    judge yet, unlike `Stop`). Without this, a `PreToolUse` edit to a path a
    `when_changed`-scoped judge gate cares about resolved the same
    ``unknown`` a genuinely missing `Stop`-time verdict does, an advisory
    warning on every single matching edit regardless of how well-behaved
    the session was (confirmed: this repo's own dogfooded judge gate did
    exactly that against itself). A caller that does run judge gates for
    this event (`Stop`) submits ``JudgeEvidence``, empty or not, and the
    checks below are unchanged either way.

    ``gate.when_changed`` narrows applicability the same way
    ``CommandIfChangedGate.when_changed`` narrows its own gate, checked
    next, before looking for a verdict. Empty (the default) means this
    gate always applies. Non-empty needs ``path_evidence`` to resolve
    at all (``unknown`` if it was never submitted, mirroring
    ``evaluate_command_if_changed``'s own applicability check) and resolves
    ``not_applicable`` when nothing the gate cares about changed, the same
    non-blocking "there was nothing to judge" this gate type otherwise has
    no way to express.

    ``evidence.verdicts`` carries one verdict per judge gate the caller
    evaluated (see :class:`JudgeEvidence`); a gate whose id has no matching
    verdict here resolves ``unknown``: unlike ``evidence`` being absent
    outright, this caller did run judge gates for this event and is
    genuinely missing one.

    A ``"not_run"`` verdict resolves ``not_run``, with the caller's reason as its detail.

    The caller's own ``"error"`` outcome (its model call failed or returned
    something unparsable) maps to :class:`Outcome.ERROR`: this is
    ``is_blocking`` like any other unresolved check, but ``gate.enforcement``
    is always ``"advisory"`` (enforced at parse time), so it can only ever
    warn, never block a required gate.
    """
    if evidence is None:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.NOT_APPLICABLE,
            message="This event does not evaluate judge gates.",
        )

    if gate.when_changed:
        if path_evidence is None:
            return GateResult(
                gate_id=gate.id,
                enforcement=gate.enforcement,
                outcome=Outcome.UNKNOWN,
                message="Path evidence was not submitted.",
            )
        if not matched_changed_paths(gate.when_changed, path_evidence.paths):
            return GateResult(
                gate_id=gate.id,
                enforcement=gate.enforcement,
                outcome=Outcome.NOT_APPLICABLE,
                message="No changed path matched this gate's when_changed globs.",
            )

    verdict = next((v for v in evidence.verdicts if v.gate_id == gate.id), None)
    if verdict is None:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.UNKNOWN,
            message="No model verdict was submitted for this gate.",
        )

    if verdict.outcome == "not_run":
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.NOT_RUN,
            message="This judge gate was skipped.",
            detail=verdict.reasoning or None,
        )

    if verdict.outcome == "error":
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.ERROR,
            message="The model verdict could not be produced.",
            detail=verdict.reasoning or None,
        )

    if verdict.outcome == "fail":
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.FAIL,
            message=gate.failure_message,
            detail=verdict.reasoning or None,
        )

    return GateResult(
        gate_id=gate.id,
        enforcement=gate.enforcement,
        outcome=Outcome.PASS,
        message="The model verdict judged this gate's rubric satisfied.",
        detail=verdict.reasoning or None,
    )


def evaluate_verifier(
    gate: VerifierGate, path_evidence: PathEvidence | None, evidence: CheckEvidence | None
) -> GateResult:
    """Relay the caller's own verifier verdict for this gate; Otari never runs a verifier itself.

    Structured exactly like ``evaluate_judge``, including the same
    ``evidence is None`` (this event never runs verifier gates, resolves
    ``not_applicable``) vs. "ran verifier gates but is missing this
    one's verdict" (resolves ``unknown``) distinction; see that function's
    own docstring and docs/agent-guardrails.md for why both matter here too.

    Unlike a judge gate, this gate's outcome can genuinely block a required
    gate: a verifier's exit code is reproducible, not a model's opinion, so
    nothing here restricts ``gate.enforcement`` the way ``domain.policy``
    restricts a judge gate's.
    """
    if evidence is None:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.NOT_APPLICABLE,
            message="This event does not evaluate verifier gates.",
        )

    if gate.when_changed:
        if path_evidence is None:
            return GateResult(
                gate_id=gate.id,
                enforcement=gate.enforcement,
                outcome=Outcome.UNKNOWN,
                message="Path evidence was not submitted.",
            )
        if not matched_changed_paths(gate.when_changed, path_evidence.paths):
            return GateResult(
                gate_id=gate.id,
                enforcement=gate.enforcement,
                outcome=Outcome.NOT_APPLICABLE,
                message="No changed path matched this gate's when_changed globs.",
            )

    verdict = next((v for v in evidence.verdicts if v.gate_id == gate.id), None)
    if verdict is None:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.UNKNOWN,
            message="No verifier result was submitted for this gate.",
        )

    if verdict.outcome == "error":
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.ERROR,
            message="The verifier could not be run.",
            detail=verdict.detail or None,
        )

    if verdict.outcome == "fail":
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.FAIL,
            message=gate.failure_message,
            detail=verdict.detail or None,
        )

    return GateResult(
        gate_id=gate.id,
        enforcement=gate.enforcement,
        outcome=Outcome.PASS,
        message="The verifier reported no violation.",
        detail=verdict.detail or None,
    )
