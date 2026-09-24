"""Evaluate a submitted policy against submitted evidence.

The orchestration shared by the Hook Server route (``routes/hooks.py``) and
``otari hook``'s own local evaluation: parse the policy, guard the match-cost
budgets below, dispatch every gate to its evaluator, and fold the results
into one pass/fail verdict. Pure, like every other module in this package:
no filesystem, network, subprocess, or clock access. A caller collects its
own evidence; this only ever computes over what it was given.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from otari_agent.domain.evaluators import (
    evaluate_command,
    evaluate_command_if_changed,
    evaluate_judge,
    evaluate_path,
    evaluate_verifier,
    tokenize_commands,
    tokenize_phrases,
)
from otari_agent.domain.policy import PolicyError, parse_policy
from otari_agent.domain.types import (
    PATH_EVIDENCE_SOURCES,
    CheckEvidence,
    CheckVerdict,
    CommandEvidence,
    CommandGate,
    CommandIfChangedGate,
    EvidenceScope,
    GateResult,
    GateSpec,
    JudgeEvidence,
    JudgeGate,
    JudgeVerdict,
    PathEvidence,
    PathGate,
    RunsAt,
    VerifierGate,
)

# A submitted evidence list is caller-observed, not Otari-observed, but it is
# still bounded input: this caps a pathological request/run, not a real repo.
_MAX_PATH_LENGTH = 4096
_MAX_COMMAND_LENGTH = 4096

# A per-match cost bound (domain/evaluators.py) does not bound the total cost
# of one check: this estimates total path-match work as
# pattern_count * total_path_length + path_count * total_pattern_length,
# which is what the matcher's own cost scales with, and rejects a policy/
# evidence combination whose estimate is disproportionate rather than let it
# run. See routes/hooks.py's history (this module inherits its calibration
# unchanged) for the benchmarking behind these numbers.
_MAX_MATCH_WORK = 50_000_000

# A byte-weighted budget alone understates a check built from many *short*
# patterns and paths: each comparison costs a near-constant overhead
# regardless of how few bytes it compares, so a cheap-by-bytes check can
# still mean millions of individual calls. This bounds the raw comparison
# count directly, independent of length.
_MAX_COMPARISONS = 1_000_000

# command's per-(phrase, command) match cost is a product, not a sum,
# so it needs its own bound and its own calibration from
# total_pattern_tokens * total_command_tokens.
_MAX_COMMAND_WORK = 2_000_000

# Independent of token length, for the same reason _MAX_COMPARISONS exists
# alongside _MAX_MATCH_WORK: many short phrases against many near-empty
# commands is still one comparison per pair.
_MAX_COMMAND_COMPARISONS = 500_000

# Tokenizing itself is not free (shlex.split costs real time per character
# regardless of content), so this caps the raw character total *before* any
# command is tokenized, using only len() (uniformly cheap regardless of
# content).
_MAX_TOTAL_COMMAND_CHARS = 2_000_000


class PolicyCheckError(Exception):
    """The policy, or the evidence submitted against it, cannot be evaluated.

    Covers both a malformed policy (``PolicyError``, from ``parse_policy``)
    and evidence whose match cost exceeds this build's budgets, so a caller
    has one exception to catch rather than two.
    """


@dataclass(frozen=True, slots=True)
class PolicyCheckResult:
    """One evaluated policy check: every gate's result, folded into a verdict."""

    policy_id: str
    schema_version: str
    results: tuple[GateResult, ...]
    blocked: bool


def _evaluate_gate(
    gate: GateSpec,
    path_evidence: PathEvidence | None,
    command_evidence: CommandEvidence | None,
    judge_evidence: JudgeEvidence | None,
    check_evidence: CheckEvidence | None,
    segment_cache: dict[str, list[list[str]]] | None,
    phrase_cache: dict[str, list[str]] | None,
) -> GateResult:
    """Dispatch one gate to its evaluator. Extend as a new gate type joins ``GateSpec``."""
    if isinstance(gate, PathGate):
        return evaluate_path(gate, path_evidence)
    if isinstance(gate, CommandGate):
        return evaluate_command(gate, command_evidence, segment_cache=segment_cache, phrase_cache=phrase_cache)
    if isinstance(gate, JudgeGate):
        return evaluate_judge(gate, path_evidence, judge_evidence)
    if isinstance(gate, VerifierGate):
        return evaluate_verifier(gate, path_evidence, check_evidence)
    return evaluate_command_if_changed(
        gate, path_evidence, command_evidence, segment_cache=segment_cache, phrase_cache=phrase_cache
    )


def run_policy_check(
    policy_yaml: str,
    *,
    source: str,
    paths: Sequence[str] | None,
    commands: Sequence[str] | None,
    path_source: RunsAt | None = None,
    command_scope: EvidenceScope = "call",
    judge_results: Sequence[JudgeVerdict] | None = None,
    check_results: Sequence[CheckVerdict] | None = None,
) -> PolicyCheckResult:
    """Parse ``policy_yaml`` and evaluate it against the given evidence.

    ``source`` names ``policy_yaml`` in a raised ``PolicyCheckError`` (e.g.
    "request body", or a guardrail-file path). Every other argument mirrors
    ``routes/hooks.py``'s own ``PolicyCheckRequest`` fields one for one,
    including their tri-state contracts (see that model's own field docs,
    and docs/agent-guardrails.md): ``None`` means a caller that never collects
    that evidence kind at all (resolves ``unknown``/``not_applicable``
    depending on the gate type); ``()``/``[]`` means it collected some and
    there is none (resolves ``not_applicable``).
    """
    try:
        spec = parse_policy(policy_yaml, source=source)
    except PolicyError as exc:
        raise PolicyCheckError(str(exc)) from exc

    for path in paths or ():
        if len(path) > _MAX_PATH_LENGTH:
            raise PolicyCheckError(f"paths entry exceeds {_MAX_PATH_LENGTH} characters.")
    for command in commands or ():
        if len(command) > _MAX_COMMAND_LENGTH:
            raise PolicyCheckError(f"commands entry exceeds {_MAX_COMMAND_LENGTH} characters.")

    path_gates = [gate for gate in spec.gates if isinstance(gate, PathGate)]
    command_gates = [gate for gate in spec.gates if isinstance(gate, CommandGate)]
    command_if_changed_gates = [gate for gate in spec.gates if isinstance(gate, CommandIfChangedGate)]
    judge_gates = [gate for gate in spec.gates if isinstance(gate, JudgeGate)]
    verifier_gates = [gate for gate in spec.gates if isinstance(gate, VerifierGate)]

    # Deduplicated once here and reused below: gate.forbidden/when_changed/
    # require are already deduplicated at parse time (domain.policy), so
    # each estimate and its matching evaluation always agree on the same,
    # cheaper counts. command_if_changed's, judge's, and verifier's own
    # when_changed globs are path-matching work exactly like path's
    # forbidden globs (evaluate_judge/evaluate_verifier both call the
    # same matched_changed_paths), so all four share the same budget.
    # A submitted path list must say which moment it was read at, or a gate
    # cannot tell a PreToolUse call apart from a Stop event on a clean tree and
    # the `runs` declaration means nothing. Refused rather than defaulted:
    # any default silently disables every path gate that named a different one.
    if paths and path_source not in PATH_EVIDENCE_SOURCES:
        # Refuses a missing label and an inapplicable one alike. A path read at
        # `stop.session` or `pre_tool_use.command` is not a moment any path gate
        # can declare, so accepting it would resolve every one of them
        # not_applicable: enforcement lost without a word, which is the failure
        # this whole field exists to remove.
        raise PolicyCheckError(
            f"paths was submitted with path_source={path_source!r}, which no "
            "path gate can be declared to run at, so none of them could resolve against it. Submit one "
            f"of: {', '.join(PATH_EVIDENCE_SOURCES)}."
        )
    path_evidence = PathEvidence(paths=tuple(dict.fromkeys(paths)), source=path_source) if paths is not None else None
    path_list = path_evidence.paths if path_evidence is not None else ()
    path_globs = (
        [glob for gate in path_gates for glob in gate.forbidden]
        + [glob for gate in command_if_changed_gates for glob in gate.when_changed]
        + [glob for gate in judge_gates for glob in gate.when_changed]
        + [glob for gate in verifier_gates for glob in gate.when_changed]
    )
    pattern_count = len(path_globs)
    total_pattern_length = sum(len(glob) for glob in path_globs)
    path_count = len(path_list)
    total_path_length = sum(len(path) for path in path_list)
    estimated_work = pattern_count * total_path_length + path_count * total_pattern_length
    comparisons = pattern_count * path_count
    if estimated_work > _MAX_MATCH_WORK or comparisons > _MAX_COMPARISONS:
        raise PolicyCheckError(
            f"This policy and evidence would take an estimated {estimated_work:,} match operations "
            f"across {comparisons:,} pattern/path comparisons, over this build's limits "
            f"({_MAX_MATCH_WORK:,} and {_MAX_COMPARISONS:,} respectively). Narrow the policy's "
            "forbidden globs or the submitted paths."
        )

    command_evidence = (
        CommandEvidence(commands=tuple(dict.fromkeys(commands)), scope=command_scope) if commands is not None else None
    )
    # Gated on there being a gate that reads command evidence at all, and on
    # evidence actually being present: tokenizing a command is exactly the
    # cost _MAX_TOTAL_COMMAND_CHARS below exists to bound, and a policy or
    # caller with nothing to check against must not pay it.
    segment_cache: dict[str, list[list[str]]] | None = None
    phrase_cache: dict[str, list[str]] | None = None
    if (command_gates or command_if_changed_gates) and command_evidence is not None:
        total_command_chars = sum(len(command) for command in command_evidence.commands)
        if total_command_chars > _MAX_TOTAL_COMMAND_CHARS:
            raise PolicyCheckError(
                f"Submitted commands total {total_command_chars:,} characters, over the "
                f"{_MAX_TOTAL_COMMAND_CHARS:,} limit. Narrow the submitted commands."
            )

        # Tokenized exactly once here and reused for both the estimate below
        # and the real evaluation further down (passed to every
        # evaluate_command/evaluate_command_if_changed call as
        # segment_cache): each is called once per gate against this same
        # evidence, and without sharing this, each call would re-tokenize
        # every command from scratch, multiplying the already-checked cost
        # above by the number of gates.
        segment_cache = tokenize_commands(command_evidence.commands)

        # Policy parsing already proved every forbidden/require phrase
        # tokenizes (domain.policy's own validation), so this cannot raise.
        # command_if_changed's require phrases are command-matching work
        # exactly like command's forbidden phrases, so they share the
        # same budget and cache rather than needing a third one.
        command_phrases = tuple(phrase for gate in command_gates for phrase in gate.forbidden) + tuple(
            phrase for gate in command_if_changed_gates for phrase in gate.require
        )
        phrase_cache = tokenize_phrases(command_phrases)
        # Per gate occurrence, not per distinct phrase text: phrase_cache
        # dedupes identical phrase text across gates so each is tokenized
        # once, but evaluate_command/evaluate_command_if_changed still
        # run the match once per gate that carries it.
        phrase_count = sum(len(gate.forbidden) for gate in command_gates) + sum(
            len(gate.require) for gate in command_if_changed_gates
        )
        total_phrase_tokens = sum(
            len(phrase_cache[phrase]) for gate in command_gates for phrase in gate.forbidden
        ) + sum(len(phrase_cache[phrase]) for gate in command_if_changed_gates for phrase in gate.require)
        command_count = len(command_evidence.commands)
        total_command_tokens = sum(len(segment) for segments in segment_cache.values() for segment in segments)
        estimated_command_work = total_phrase_tokens * total_command_tokens
        command_comparisons = phrase_count * command_count
        if estimated_command_work > _MAX_COMMAND_WORK or command_comparisons > _MAX_COMMAND_COMPARISONS:
            raise PolicyCheckError(
                f"This policy and evidence would take an estimated {estimated_command_work:,} command match "
                f"operations across {command_comparisons:,} phrase/command comparisons, over this build's "
                f"limits ({_MAX_COMMAND_WORK:,} and {_MAX_COMMAND_COMPARISONS:,} respectively). Narrow "
                "the policy's forbidden phrases or the submitted commands."
            )

    judge_evidence = JudgeEvidence(verdicts=tuple(judge_results)) if judge_results is not None else None
    check_evidence = CheckEvidence(verdicts=tuple(check_results)) if check_results is not None else None

    # Evaluated in declaration order (not grouped by type) so a caller
    # reading `results` positionally sees the same order as the policy it
    # submitted.
    results = tuple(
        _evaluate_gate(
            gate, path_evidence, command_evidence, judge_evidence, check_evidence, segment_cache, phrase_cache
        )
        for gate in spec.gates
    )
    blocked = any(result.enforcement == "required" and result.outcome.is_blocking for result in results)

    return PolicyCheckResult(
        policy_id=spec.policy_id,
        schema_version=spec.schema_version,
        results=results,
        blocked=blocked,
    )
