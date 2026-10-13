"""Which built-in tools Otari runs on each candidate a request may be sent to.

A request may declare a built-in tool in a provider's own words (Anthropic's
``web_search_<date>``, OpenAI's ``code_interpreter``) and say who should run it,
in :class:`~gateway.models.tools.CodeExecutor`'s vocabulary: ``otari`` always
runs it here, ``auto`` leaves it with a provider that runs it itself and runs it
here for one that does not, and ``provider`` always forwards it. Whether a
provider runs a declaration itself is the tool's own answer
(``BuiltinTool.runs_natively``), asked here once per candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from gateway.models.tools import CodeExecutor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gateway.services.tools._builtin_tool import BuiltinTool
    from gateway.services.tools._native import Dialect


@dataclass(frozen=True)
class ProviderToolDeclaration:
    """A built-in tool one request declared in a provider's own words, and who it asked to run it.

    ``preference`` is ``None`` where nothing asked, which forwards the declaration to every candidate.
    ``backend_configured`` is whether this deployment can run the tool, without which Otari claims nothing.
    ``uniform`` decides every candidate alike: Otari runs it for all of them where any one needs it.
    ``follow_head`` decides every candidate as the head is decided: Otari runs it for all of them
    where the head needs it, and for none otherwise.
    """

    tool: BuiltinTool
    entry: dict[str, Any]
    preference: CodeExecutor | None
    backend_configured: bool
    uniform: bool = False
    follow_head: bool = False


@dataclass(frozen=True)
class CandidateTools:
    """What one candidate is sent for the declarations a request made in a provider's words.

    ``claimed`` names the tools Otari runs in place of the provider's, and ``native``
    the ones forwarded to a provider that runs them itself. A tool in neither is
    forwarded as written.
    """

    claimed: frozenset[str] = frozenset()
    native: frozenset[str] = frozenset()

    def __or__(self, other: CandidateTools) -> CandidateTools:
        """The same candidate's answers for two sets of declarations, taken together."""
        return CandidateTools(claimed=self.claimed | other.claimed, native=self.native | other.native)


class _Decision(Enum):
    CLAIMED = auto()
    NATIVE = auto()
    FORWARDED = auto()


def _forwarded(declaration: ProviderToolDeclaration, *, provider: str | None, dialect: Dialect) -> _Decision:
    native = declaration.tool.runs_natively(declaration.entry, provider=provider, dialect=dialect)
    return _Decision.NATIVE if native else _Decision.FORWARDED


def _decide(declaration: ProviderToolDeclaration, *, provider: str | None, dialect: Dialect) -> _Decision:
    forwarded = _forwarded(declaration, provider=provider, dialect=dialect)
    native = forwarded is _Decision.NATIVE
    match declaration.preference:
        case CodeExecutor.OTARI:
            return _Decision.CLAIMED if declaration.backend_configured else forwarded
        case CodeExecutor.AUTO:
            return _Decision.CLAIMED if declaration.backend_configured and not native else forwarded
        case _:
            return forwarded


def decide_candidate_tools(
    declarations: Sequence[ProviderToolDeclaration],
    *,
    providers: Sequence[str | None],
    dialect: Dialect,
) -> list[CandidateTools]:
    """What each candidate in ``providers`` (head first) is sent for ``declarations``.

    The one place a request's built-in tools are decided per candidate, so every
    step that builds an attempt reads the same answer.
    """
    decisions = [
        [_decide(declaration, provider=provider, dialect=dialect) for declaration in declarations]
        for provider in providers
    ]
    for index, declaration in enumerate(declarations):
        if declaration.uniform and any(row[index] is _Decision.CLAIMED for row in decisions):
            for row in decisions:
                row[index] = _Decision.CLAIMED
        elif declaration.follow_head and decisions:
            head_claimed = decisions[0][index] is _Decision.CLAIMED
            for provider, row in zip(providers, decisions, strict=True):
                row[index] = (
                    _Decision.CLAIMED if head_claimed else _forwarded(declaration, provider=provider, dialect=dialect)
                )
    return [
        CandidateTools(
            claimed=frozenset(
                declaration.tool.name
                for declaration, decision in zip(declarations, row, strict=True)
                if decision is _Decision.CLAIMED
            ),
            native=frozenset(
                declaration.tool.name
                for declaration, decision in zip(declarations, row, strict=True)
                if decision is _Decision.NATIVE
            ),
        )
        for row in decisions
    ]
