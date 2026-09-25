"""Parse a submitted guardrail file, and compose several into one policy.

Pure: the caller (an agent hook, eventually the native dispatcher) reads its
own repo's policy file and Git evidence and submits both in one request, per
the production plan's hook-protocol design. This module never touches a
filesystem, a network, or a clock; it only turns already-received YAML text
into a validated :class:`PolicySpec`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import yaml

from otari_agent.domain.evaluators import tokenize_phrase
from otari_agent.domain.types import (
    PATH_EVIDENCE_SOURCES,
    CommandGate,
    CommandIfChangedGate,
    Enforcement,
    GateSpec,
    JudgeGate,
    PathGate,
    PolicySpec,
    RunsAt,
    VerifierGate,
)

# A policy body is a developer-edited text file, not a data export; this bounds
# a pathological input (and an accidental binary) before it ever reaches the
# YAML parser. Production sizing for included/packed policies is AG-005.
# Exported (not module-private) so a caller enforcing its own request-size
# limit, such as the Hook Server route, shares this one number rather than
# duplicating it.
MAX_POLICY_BYTES = 256 * 1024

# Per *file*, not per composed set: a file that parses on its own stays legal
# wherever it is dropped, which is the whole point of composing a directory of
# standalone files. That leaves the composed total unbounded, so this bounds
# the file count instead, which no one reaches by writing guardrails by hand
# and a runaway generator hits as a clear error rather than a slow hook.
MAX_POLICY_FILES = 64

# Exported for the same reason as MAX_POLICY_BYTES: JudgeVerdictRequest.gate_id
# (routes/hooks.py) caps at this same length, since a verdict echoes back the
# gate id the policy itself named. Enforced here too, not only there: without
# it, a policy accepted at parse time (unbounded id length) could name a judge
# gate whose id `otari hook` can never actually submit a verdict for -- the
# whole /hooks/check request 422s on that one field, fail-open, taking every
# other gate in the same policy, mechanical and required ones included, down
# with it. A gate id this build accepts must always be one a verdict can
# round-trip.
MAX_GATE_ID_LENGTH = 200

_SUPPORTED_SCHEMA_VERSIONS = {"1.0"}

# Which `runs` values each gate type may declare. Four of the five admit
# exactly one, because only a path is knowable at more than one moment: about
# to be written, about to be read, or changed by the time the turn ends (see
# PathGate). Requiring the field anyway, rather than defaulting the
# single-choice types, is deliberate: a reader never has to know which types
# have a choice to know when a gate runs, and a wrong value is a parse error
# that says so instead of a gate that quietly never fires.
_LEGAL_RUNS_BY_GATE_TYPE = {
    "path": PATH_EVIDENCE_SOURCES,
    "command": ("pre_tool_use.command",),
    "command_if_changed": ("stop.session",),
    "judge": ("stop.session",),
    "verifier": ("stop.verifier",),
}
_SUPPORTED_GATE_TYPES = {"path", "command", "command_if_changed", "judge", "verifier"}
_SUPPORTED_ENFORCEMENTS = {"required", "advisory"}

# A model's verdict is not reproducible the way a glob or phrase match is, so
# a judge gate may never be the thing that blocks a required gate; see
# JudgeGate's own docstring. Checked here, not left to the caller's own
# discipline, so a mistaken `enforcement: required` is a 422 at policy-load
# time rather than a gate that silently blocks on a model's say-so.
_JUDGE_ENFORCEMENTS = {"advisory"}

# Which locally-installed CLI(s) `otari hook` may use for a judge gate's own
# model call (JudgeGate.judge_cli); see that field's own docstring. Kept as
# its own set, not reused from anywhere `otari hook` itself defines, since
# this module stays dependency-free of that CLI-only concern (subprocess
# names, PATH resolution): a gate author only ever needs to know these two
# names exist, not how either is actually invoked.
_SUPPORTED_JUDGE_CLIS = {"claude", "codex"}

# A `**` in a forbidden glob crosses path segments by recursing over every
# split point in the submitted path (domain/evaluators.py's _segments_match).
# One is what every example in this codebase uses; more than that multiplies
# that recursion's branching and reopens the complexity problem the
# non-backtracking segment matcher exists to close. Not a length or *-count
# limit: those no longer matter for safety once matching stopped using a
# backtracking regex, only how many places the pattern can cross a directory.
_MAX_DOUBLE_STAR_PER_GLOB = 1

_TOP_LEVEL_FIELDS = {"schema_version", "policy", "gates"}
_POLICY_FIELDS = {"id", "description"}
_COMMON_GATE_FIELDS = {"id", "type", "enforcement", "message", "runs"}
# Each gate type accepts only the common fields plus its own: a
# path gate submitting when_changed, or a command_if_changed gate
# submitting forbidden, is an unknown-field error like any other, not a
# silently-ignored one.
_GATE_FIELDS_BY_TYPE = {
    "path": _COMMON_GATE_FIELDS | {"forbidden"},
    "command": _COMMON_GATE_FIELDS | {"forbidden"},
    "command_if_changed": _COMMON_GATE_FIELDS | {"when_changed", "require"},
    "judge": _COMMON_GATE_FIELDS | {"rubric", "when_changed", "judge_cli", "priority"},
    "verifier": _COMMON_GATE_FIELDS | {"verifier", "when_changed", "priority"},
}

# A rubric is prompt text, not a glob or phrase; bounded generously since it
# feeds a model prompt the caller builds, not a matcher whose cost this
# module has to estimate the way it does for _MAX_DOUBLE_STAR_PER_GLOB.
_MAX_RUBRIC_BYTES = 16 * 1024

# A verifier is a repo-relative path, not free text; bounded like a path
# field rather than like rubric's prompt-text allowance.
_MAX_VERIFIER_LENGTH = 4096


class PolicyError(Exception):
    """A submitted policy body is malformed or fails schema validation.

    Always raised rather than returning a partial policy: a policy the parser
    could not fully understand must never silently evaluate as "no gates".
    """


class _DuplicateKeyLoader(yaml.SafeLoader):
    """A SafeLoader that rejects a mapping with a repeated key instead of keeping the last one."""


def _construct_mapping(loader: Any, node: yaml.MappingNode) -> dict[Any, Any]:
    """Build a mapping node, refusing a repeated key.

    ``loader`` is untyped because this one constructor is registered on both
    loaders below, whose classes share no base: PyYAML's C loader is not a
    subclass of its pure-Python one.
    """
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            # YAML permits a sequence or mapping as a key (`? [a, b]\n: c`),
            # which Python cannot hash. Raising here, instead of letting the
            # lookup itself raise TypeError, keeps this an intelligible
            # PolicyError like any other malformed policy rather than an
            # unhandled 500.
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found unhashable key {key!r}",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_DuplicateKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


# The same loader on libyaml, which parses a real policy an order of magnitude
# faster than the pure-Python one. Worth having as its own class rather than
# simply swapping the base above, because the two are not interchangeable on
# the failing path: a libyaml error mark carries no source snippet, so its
# message names a line and column while the pure loader's also prints the
# offending line with a caret under it. A policy is a hand-edited file, and
# that snippet is most of what makes a YAML error in one readable, so
# `_load_policy_document` parses with this and re-reads with the other only
# when something is already wrong. The class is built conditionally because a
# PyYAML installed from an sdist without libyaml present exposes no
# `CSafeLoader` at all, and the light CLI's dependency closure is sdist-only
# (see packaging/homebrew).
_HAS_LIBYAML = hasattr(yaml, "CSafeLoader")

if _HAS_LIBYAML:

    class _FastDuplicateKeyLoader(yaml.CSafeLoader):
        """:class:`_DuplicateKeyLoader` on libyaml; see the comment above."""

    _FastDuplicateKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)
    _fast_loader: Any = _FastDuplicateKeyLoader
else:
    _fast_loader = _DuplicateKeyLoader


def _load_policy_document(raw_yaml: str, source: str) -> Any:
    """Parse ``raw_yaml`` as one YAML document, or raise :class:`PolicyError` naming ``source``."""
    try:
        return yaml.load(raw_yaml, Loader=_fast_loader)
    except yaml.YAMLError as exc:
        error: yaml.YAMLError = exc
        if _HAS_LIBYAML:
            # Only to rebuild the message with a source snippet the C loader's
            # own marks do not carry. If the pure loader somehow accepts what
            # the fast one refused, the fast one's error still stands: the
            # document this build would actually have evaluated is the one it
            # could not read.
            try:
                yaml.load(raw_yaml, Loader=_DuplicateKeyLoader)
            except yaml.YAMLError as readable:
                error = readable
        raise PolicyError(f"{source} is not valid YAML: {error}") from exc


def _require_fields(document: dict[str, Any], known: set[str], where: str) -> None:
    unknown = set(document) - known
    if unknown:
        # A YAML mapping key need not be a string (`42: foo` is a valid,
        # if unusual, key), so this cannot assume every element of `unknown`
        # is one: sorting or joining it directly raises TypeError on a mixed
        # str/int set rather than reporting the error it was building.
        formatted = sorted(repr(key) for key in unknown)
        raise PolicyError(f"Unknown field(s) in {where}: {', '.join(formatted)}.")


def _require_string_list(raw: dict[str, Any], field: str, gate_id: str, gate_type: str) -> list[str]:
    """A non-empty list of non-empty strings under ``field``, deduplicated in first-seen order.

    A duplicate entry matches nothing a single copy wouldn't; collapsing it
    here (once, at parse time) is what keeps a caller who repeats one entry
    many times from multiplying the Hook Server's per-request match work for
    zero effect on the result.
    """
    value = raw.get(field)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise PolicyError(f"Gate {gate_id!r} (type {gate_type!r}) needs a non-empty list of {field!r} entries.")
    return list(dict.fromkeys(value))


def _parse_glob_list(gate_id: str, field: str, globs: list[str]) -> list[str]:
    """Validate every entry as a bounded repo-relative POSIX glob (see ``_MAX_DOUBLE_STAR_PER_GLOB``)."""
    for glob in globs:
        # Only a segment that is exactly "**" crosses directories and
        # recurses in _segments_match; "a****b" is a literal-with-stars
        # pattern the linear intra-segment matcher handles safely, however
        # many '*' it has, and must not be counted here.
        double_star_segments = sum(1 for segment in glob.split("/") if segment == "**")
        if double_star_segments > _MAX_DOUBLE_STAR_PER_GLOB:
            raise PolicyError(
                f"Gate {gate_id!r}: {field} glob {glob!r} uses '**' as its own path "
                f"segment more than {_MAX_DOUBLE_STAR_PER_GLOB} time(s)."
            )
    return globs


def _parse_phrase_list(gate_id: str, field: str, phrases: list[str]) -> list[str]:
    """Validate every entry as a shell phrase the evaluator can tokenize (domain/evaluators.py).

    Validating that it tokenizes, and to at least one token, here rather than
    at evaluation time means a malformed phrase is a 422 at policy-load time,
    not an unhandled error the first time a command happens to be evaluated
    against it.
    """
    for phrase in phrases:
        try:
            phrase_tokens = tokenize_phrase(phrase)
        except ValueError as exc:
            raise PolicyError(
                f"Gate {gate_id!r}: {field} phrase {phrase!r} is not a valid shell phrase: {exc}"
            ) from exc
        if not phrase_tokens:
            raise PolicyError(f"Gate {gate_id!r}: {field} phrase {phrase!r} has no tokens to match.")
    return phrases


def _parse_priority(gate_id: str, raw: dict[str, Any]) -> int:
    """Validate the optional ``priority`` a capped gate type may declare (``domain.types.by_priority``).

    Absent means 0, which is what every gate had before the field existed, so
    a policy that declares it on none of its gates keeps declaration order
    throughout.
    """
    if "priority" not in raw:
        return 0
    value = raw["priority"]
    # `isinstance(True, int)` is True, and `priority: true` is a mistake
    # rather than a request for 1.
    if not isinstance(value, int) or isinstance(value, bool):
        raise PolicyError(f"Gate {gate_id!r}: 'priority' must be an integer, got {value!r}.")
    return value


def _parse_runs(gate_id: str, gate_type: str, raw: dict[str, Any]) -> list[str]:
    """Validate ``runs`` as the moments this gate's own type can actually run at.

    Accepts a bare string as the one-entry case, the same shape ``judge_cli``
    allows, since a gate naming a single moment should not have to spell it as
    a one-item list. Deduplicated first-seen like every other list field, and
    an empty list is rejected for the reason ``when_changed: []`` is: it means
    a gate that can never fire, which is worse than deleting it.
    """
    legal = _LEGAL_RUNS_BY_GATE_TYPE[gate_type]
    value = raw.get("runs")
    if isinstance(value, str):
        candidates = [value] if value else []
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        candidates = list(dict.fromkeys(value))
    else:
        candidates = []
    if not candidates:
        raise PolicyError(
            f"Gate {gate_id!r} (type {gate_type!r}) needs a non-empty 'runs', "
            f"naming when it runs and what it sees there. Legal for this type: {', '.join(legal)}."
        )
    unsupported = [item for item in candidates if item not in legal]
    if unsupported:
        raise PolicyError(
            f"Gate {gate_id!r} (type {gate_type!r}) cannot run at {', '.join(unsupported)}. "
            f"Legal for this type: {', '.join(legal)}."
        )
    return candidates


def _parse_gate(raw: Any) -> GateSpec:
    if not isinstance(raw, dict):
        raise PolicyError(f"Each entry under 'gates' must be a mapping, got {type(raw).__name__}.")

    gate_id = raw.get("id")
    if not isinstance(gate_id, str) or not gate_id:
        raise PolicyError(f"Gate is missing a non-empty 'id': {raw!r}")
    if len(gate_id) > MAX_GATE_ID_LENGTH:
        raise PolicyError(f"Gate id {gate_id!r} is longer than {MAX_GATE_ID_LENGTH} characters.")

    gate_type = raw.get("type")
    # isinstance first: `in` on a set hashes its argument, and a caller can
    # submit an unhashable type (e.g. `type: []`) for any field here. Every
    # membership check below needs the same guard for the same reason.
    if not isinstance(gate_type, str) or gate_type not in _SUPPORTED_GATE_TYPES:
        raise PolicyError(
            f"Gate {gate_id!r} has unsupported type {gate_type!r}. "
            f"Supported in this build: {', '.join(sorted(_SUPPORTED_GATE_TYPES))}."
        )
    # Field set is picked only once `type` itself is known valid, so a gate
    # is judged against the fields its own type actually uses, not some
    # union of every type's fields.
    _require_fields(raw, _GATE_FIELDS_BY_TYPE[gate_type], f"gate {gate_id!r}")

    enforcement = raw.get("enforcement")
    if not isinstance(enforcement, str) or enforcement not in _SUPPORTED_ENFORCEMENTS:
        raise PolicyError(
            f"Gate {gate_id!r} has invalid enforcement {enforcement!r}. "
            f"Must be one of: {', '.join(sorted(_SUPPORTED_ENFORCEMENTS))}."
        )
    if gate_type == "judge" and enforcement not in _JUDGE_ENFORCEMENTS:
        raise PolicyError(
            f"Gate {gate_id!r} (type 'judge') must use enforcement: "
            f"{', '.join(sorted(_JUDGE_ENFORCEMENTS))}. A model verdict can never block a required gate."
        )
    # The isinstance+membership check above is the runtime proof a plain str
    # type can't carry; cast documents that this narrowing is deliberate.
    enforcement_value = cast(Enforcement, enforcement)

    # Optional: a gate's own fields already say what it checks, and each gate
    # type derives a generic line from them (see `failure_message`), so
    # demanding this only forced an author to restate the rule. It was worst on
    # a judge gate, whose rubric states it at length one field above, which is
    # why a generated one kept arriving without it.
    message = raw.get("message", "")
    if not isinstance(message, str):
        raise PolicyError(f"Gate {gate_id!r}: 'message' must be a string.")
    message = message.strip()

    # Parsed once here rather than per branch: every gate type carries it, and
    # the legal set is keyed on the already-validated gate_type.
    runs = tuple(cast(list[RunsAt], _parse_runs(gate_id, gate_type, raw)))

    if gate_type == "path":
        forbidden = _parse_glob_list(gate_id, "forbidden", _require_string_list(raw, "forbidden", gate_id, gate_type))
        return PathGate(
            id=gate_id,
            runs=runs,
            enforcement=enforcement_value,
            forbidden=tuple(forbidden),
            message=message,
        )

    if gate_type == "command":
        forbidden = _parse_phrase_list(gate_id, "forbidden", _require_string_list(raw, "forbidden", gate_id, gate_type))
        return CommandGate(
            id=gate_id,
            runs=runs,
            enforcement=enforcement_value,
            forbidden=tuple(forbidden),
            message=message,
        )

    if gate_type == "command_if_changed":
        # when_changed is the same glob grammar path's forbidden
        # uses; require is the same shell-phrase grammar command's
        # forbidden uses, just under different field names because both
        # evidence kinds apply to the same gate at once.
        when_changed = _parse_glob_list(
            gate_id, "when_changed", _require_string_list(raw, "when_changed", gate_id, gate_type)
        )
        require = _parse_phrase_list(gate_id, "require", _require_string_list(raw, "require", gate_id, gate_type))
        return CommandIfChangedGate(
            id=gate_id,
            runs=runs,
            enforcement=enforcement_value,
            when_changed=tuple(when_changed),
            require=tuple(require),
            message=message,
        )

    if gate_type == "verifier":
        verifier = raw.get("verifier")
        if not isinstance(verifier, str) or not verifier.strip():
            raise PolicyError(f"Gate {gate_id!r} (type 'verifier') needs a non-empty 'verifier'.")
        if len(verifier) > _MAX_VERIFIER_LENGTH:
            raise PolicyError(f"Gate {gate_id!r}: verifier is longer than {_MAX_VERIFIER_LENGTH} characters.")
        if verifier.startswith("/"):
            raise PolicyError(f"Gate {gate_id!r}: verifier {verifier!r} must be a repo-relative path, not absolute.")
        if "\x00" in verifier:
            raise PolicyError(f"Gate {gate_id!r}: verifier {verifier!r} contains a NUL byte.")
        # Optional, like judge's own when_changed: absence means "always
        # applies". See that field's own comment below for why a submitted
        # but empty list is still rejected rather than treated the same way.
        check_when_changed: list[str] = []
        if "when_changed" in raw:
            check_when_changed = _parse_glob_list(
                gate_id, "when_changed", _require_string_list(raw, "when_changed", gate_id, gate_type)
            )
        return VerifierGate(
            id=gate_id,
            runs=runs,
            enforcement=enforcement_value,
            verifier=verifier,
            when_changed=tuple(check_when_changed),
            priority=_parse_priority(gate_id, raw),
            message=message,
        )

    # judge: the only gate type _SUPPORTED_GATE_TYPES admits left once every
    # other branch above has returned.
    rubric = raw.get("rubric")
    if not isinstance(rubric, str) or not rubric.strip():
        raise PolicyError(f"Gate {gate_id!r} (type 'judge') needs a non-empty 'rubric'.")
    if len(rubric.encode("utf-8")) > _MAX_RUBRIC_BYTES:
        raise PolicyError(f"Gate {gate_id!r}: rubric is larger than {_MAX_RUBRIC_BYTES} bytes.")
    # Unlike command_if_changed's when_changed, this one is optional: its
    # absence means "always applies", the only behavior a judge gate had
    # before this field existed. A submitted but empty list is still
    # rejected by _require_string_list, the same as every other gate type's
    # glob/phrase list, rather than silently treated as "always applies" too:
    # an author who writes `when_changed: []` almost certainly meant
    # something, and guessing which is worse than a 422.
    judge_when_changed: list[str] = []
    if "when_changed" in raw:
        judge_when_changed = _parse_glob_list(
            gate_id, "when_changed", _require_string_list(raw, "when_changed", gate_id, gate_type)
        )
    # judge_cli: optional, like when_changed above; absence means "no
    # preference" (see JudgeGate's own docstring), not "always these two".
    # Accepts a bare string as the one-entry case of the list form, not a
    # different shape, since a gate author naming a single required CLI
    # should not have to spell it as a one-item list.
    judge_cli: tuple[str, ...] | None = None
    if "judge_cli" in raw:
        raw_judge_cli = raw["judge_cli"]
        if isinstance(raw_judge_cli, str):
            candidates = [raw_judge_cli] if raw_judge_cli else []
        elif isinstance(raw_judge_cli, list) and all(isinstance(item, str) for item in raw_judge_cli):
            candidates = list(dict.fromkeys(raw_judge_cli))
        else:
            candidates = []
        if not candidates:
            raise PolicyError(
                f"Gate {gate_id!r} (type 'judge'): 'judge_cli' must be a non-empty string or list of strings."
            )
        unsupported = [item for item in candidates if item not in _SUPPORTED_JUDGE_CLIS]
        if unsupported:
            raise PolicyError(
                f"Gate {gate_id!r}: judge_cli entries {unsupported!r} are not supported. "
                f"Supported in this build: {', '.join(sorted(_SUPPORTED_JUDGE_CLIS))}."
            )
        judge_cli = tuple(candidates)
    # enforcement_value is already proven "advisory" by the _JUDGE_ENFORCEMENTS
    # check above; cast documents that narrowing the same way the plain
    # Enforcement cast above documents its own.
    return JudgeGate(
        id=gate_id,
        runs=runs,
        enforcement=cast(Literal["advisory"], enforcement_value),
        rubric=rubric,
        when_changed=tuple(judge_when_changed),
        judge_cli=judge_cli,
        priority=_parse_priority(gate_id, raw),
        message=message,
    )


def parse_policy(raw_yaml: str, *, source: str) -> PolicySpec:
    """Parse and validate a policy body already read by the caller."""
    if len(raw_yaml.encode("utf-8")) > MAX_POLICY_BYTES:
        raise PolicyError(f"{source} is larger than {MAX_POLICY_BYTES} bytes.")

    document = _load_policy_document(raw_yaml, source)
    if not isinstance(document, dict):
        raise PolicyError(f"{source} must contain a YAML mapping at the top level.")
    _require_fields(document, _TOP_LEVEL_FIELDS, source)

    schema_version = document.get("schema_version")
    if not isinstance(schema_version, str) or schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise PolicyError(
            f"{source} has unsupported schema_version {schema_version!r}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_SCHEMA_VERSIONS))}."
        )

    policy_block = document.get("policy")
    if not isinstance(policy_block, dict):
        raise PolicyError(f"{source} is missing a 'policy' mapping.")
    _require_fields(policy_block, _POLICY_FIELDS, f"{source} 'policy' block")
    policy_id = policy_block.get("id")
    if not isinstance(policy_id, str) or not policy_id:
        raise PolicyError(f"{source} 'policy.id' must be a non-empty string.")

    raw_gates = document.get("gates")
    if not isinstance(raw_gates, list) or not raw_gates:
        raise PolicyError(f"{source} must declare at least one gate under 'gates'.")

    gates = [_parse_gate(raw_gate) for raw_gate in raw_gates]
    seen_ids = set()
    for gate in gates:
        if gate.id in seen_ids:
            raise PolicyError(f"{source} declares gate id {gate.id!r} more than once.")
        seen_ids.add(gate.id)

    return PolicySpec(schema_version=schema_version, policy_id=policy_id, gates=tuple(gates))


@dataclass(frozen=True, slots=True)
class PolicyFile:
    """One file of a composed guardrail: the name to report it by, and its body.

    ``name`` is for a reader, not for resolution: nothing here opens a file,
    and a caller that composed a directory passes repo-relative paths so an
    error, and later a failing gate, names something findable.
    """

    name: str
    body: str


def compose_policy(files: Sequence[PolicyFile], *, policy_id: str) -> PolicySpec:
    """Merge several standalone guardrail files into one :class:`PolicySpec`.

    Every file parses on its own, through the same :func:`parse_policy` a
    single-file guardrail goes through, carrying its own ``schema_version``
    and ``policy`` block. That is what makes a shared guardrail a file rather
    than a patch: it can be dropped into a directory, or lifted out of one,
    unchanged. Composition fails if any one file fails.

    Two rules hold across the set rather than within a file, and neither is
    last-one-wins:

    - A gate id is unique across the whole composed guardrail. A collision is
      an error naming both files, matching what :func:`parse_policy` already
      does with a repeated key inside one file.
    - Every file declares the same ``schema_version``. Only one version
      exists today, so this costs nothing now and is what keeps a future
      file dropped into an older set from being silently mixed in.

    Each file's own ``policy.id`` stays its own, and is what
    ``otari guardrails validate`` reports when checking that file alone.
    ``policy_id`` here names the composed set instead, since the set has no
    single declared id and inventing one from the parts would be a label
    nothing declares. Provenance is carried per gate, in
    :attr:`PolicySpec.gate_sources`, which is more use to a reader than any
    composed id would be.
    """
    if not files:
        raise PolicyError(f"{policy_id} composes no guardrail files.")
    if len(files) > MAX_POLICY_FILES:
        raise PolicyError(
            f"{policy_id} composes {len(files)} guardrail files, over the {MAX_POLICY_FILES} limit. "
            "Group related gates into fewer files."
        )

    parsed = [(file, parse_policy(file.body, source=file.name)) for file in files]
    schema_file, first = parsed[0]
    for file, spec in parsed[1:]:
        if spec.schema_version != first.schema_version:
            raise PolicyError(
                f"{file.name} declares schema_version {spec.schema_version!r} but {schema_file.name} "
                f"declares {first.schema_version!r}. Every file of one composed guardrail must agree."
            )

    gates: list[GateSpec] = []
    gate_sources: dict[str, str] = {}
    for file, spec in parsed:
        for gate in spec.gates:
            if gate.id in gate_sources:
                raise PolicyError(
                    f"Gate id {gate.id!r} is declared in both {gate_sources[gate.id]} and {file.name}. "
                    "A gate id is unique across the whole composed guardrail."
                )
            gate_sources[gate.id] = file.name
            gates.append(gate)

    return PolicySpec(
        schema_version=first.schema_version,
        policy_id=policy_id,
        gates=tuple(gates),
        # Left empty for a one-file set: there is nothing to disambiguate, and
        # a caller reporting the one file on every gate is noise (see
        # PolicySpec.gate_sources).
        gate_sources=MappingProxyType(gate_sources if len(files) > 1 else {}),
    )
