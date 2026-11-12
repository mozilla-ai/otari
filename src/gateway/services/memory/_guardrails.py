"""The guardrails an organization mandates, run over what agents propose before the consolidator decides on it.

An organization's mandated guardrails check what a member sends on a completion request. What an agent proposes
reaches no completion endpoint, yet once kept it reaches other members' prompts through recall, so the
consolidator runs the same mandates over a proposal's text. Memory names only what it needs of them here; the
builder in ``api/deps`` implements both protocols with the guardrails a request runs.
"""

import uuid
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GuardrailOutcome:
    """What the mandates said of one text, as the profiles that said it. Never the text, or why.

    ``blocked`` flagged it under ``block``, which refuses it. ``monitored`` flagged it under ``monitor``, which
    only records it. ``unchecked`` could not run and let it through, as each mandate's ``on_unavailable`` says.
    """

    blocked: tuple[str, ...] = ()
    monitored: tuple[str, ...] = ()
    unchecked: tuple[str, ...] = ()


class GuardrailsUnavailableError(Exception):
    """A mandate that fails closed could not be run, or gave no verdict. The message names no content."""


class MandatedGuardrails(Protocol):
    """The guardrails mandated for one workspace, ready to run."""

    async def check(self, text: str) -> GuardrailOutcome:
        """Run every mandate over ``text``.

        Raises:
            GuardrailsUnavailableError: a mandate that fails closed could not be evaluated.
        """
        ...


class MemoryGuardrails(Protocol):
    """Where the consolidator finds the guardrails an organization mandates for a workspace."""

    async def mandated(self, organization_id: uuid.UUID, workspace_id: uuid.UUID) -> MandatedGuardrails | None:
        """The enabled mandates covering the workspace, or None where there are none, so nothing is called.

        Raises:
            GuardrailsUnavailableError: the mandates could not be read, so none can be run.
        """
        ...
