"""One provider/model attempt, credential-source agnostic.

A routing decision produces an ordered list of these and the executor walks it.
The type lives here, not in ``api/routes``, because the routing services build
attempts and a service may not import the API layer (enforced by
``scripts/check_architecture.py``).

Three fields that look redundant and are not (the distinction
``ResolvedProvider`` already draws, see ``services/provider_kwargs``):

* ``instance`` is otari's own routing key. Pricing rows, budget accounting, and
  the usage log all key on it, so a named instance (``prod-openai``) must reach
  those tables under that name and not under its implementation's name.
* ``provider`` is the any-llm implementation actually dispatched against.
* ``kwargs`` carries credentials and client args as an opaque dict rather than a
  fixed ``api_key`` field, because a locally resolved attempt may have no API key
  at all: the Vertex AI path returns environment setup plus ``client_args``, and
  any instance may carry arbitrary provider-specific keys.

Hybrid-mode attempts are the special case, not the general one: the platform
returns exactly ``api_key`` (+ optional ``api_base``), which is one possible
``kwargs`` shape.
"""

from dataclasses import dataclass, field
from typing import Any

from any_llm import LLMProvider

__all__ = ["Attempt", "SelectionReason"]

# Why this attempt is in the plan, recorded on the usage row so a fallover or a
# tier-down is legible after the fact. ``condition:<key>`` and ``router:<name>``
# carry their source, so the set is open by construction.
SelectionReason = str


@dataclass(frozen=True)
class Attempt:
    """A single candidate the executor may dispatch against."""

    position: int
    """1-based index in the plan. Recorded on the usage row."""

    instance: str
    """Otari-level routing key: pricing / budget / usage-log key prefix."""

    provider: LLMProvider
    """Underlying any-llm implementation to dispatch against."""

    model: str
    """Bare model name, with no instance or provider prefix."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Credentials and client args for the any-llm call."""

    display_model: str | None = None
    """Name to relabel the response ``model`` field to, when the caller reached
    this attempt through a policy or an alias. ``None`` leaves the upstream name
    in place."""

    selection_reason: SelectionReason = "static"
    """Why this candidate is here: ``static``, ``default``, ``condition:<key>``,
    ``router:<name>``, or ``on_failure``."""

    attempt_id: str | None = None
    """Correlation id when an external control plane supplied the attempt.
    ``None`` for locally resolved attempts."""

    @property
    def dispatch_model(self) -> str:
        """The selector to hand to any-llm: ``<implementation>:<model>``."""
        return f"{self.provider.value}:{self.model}"

    def call_kwargs(self, base_request_fields: dict[str, Any]) -> dict[str, Any]:
        """Merge this attempt's credentials with the shared request fields.

        ``model`` is applied last, so the attempt's own selector always wins over
        whatever the caller asked for. Precedence between the other two is the
        same as the hybrid helper's (``default_attempt_kwargs``): request fields
        override credentials by key.

        That ordering is only safe because caller-supplied credentials cannot
        reach here. The request models derive their fields from any-llm's
        ``CompletionParams``, which has no ``api_key`` / ``api_base``, and pydantic
        drops unknown fields, so a body carrying either is stripped at validation
        rather than merged over the resolved credential.
        """
        return {**self.kwargs, **base_request_fields, "model": self.dispatch_model}
