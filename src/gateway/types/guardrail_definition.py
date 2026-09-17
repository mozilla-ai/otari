"""What a guardrail is, to the code that builds one.

A leaf type, so the credential store can produce a definition and the runner can
consume one without either importing the other, and so the request path can name
one later without reaching into the runner it hands them to.

Deliberately not fields on ``GuardrailConfig``. That model is the request body,
so anything on it is something a caller can send, and ``create_kwargs`` is where
a vendor API key and endpoint live: a caller who could set it could point a check
at a server of their own and have this gateway post the prompt there.
``ResolvedOrganizationGuardrail`` keeps a credential beside a config for the same
reason.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GuardrailDefinition:
    """A guardrail this gateway can build: the class, and the arguments for it.

    The mappings are excluded from the generated hash because a ``dict`` cannot be
    hashed, and ``frozen=True`` would otherwise build a ``__hash__`` that raises
    the first time an instance reached a set. Equality stays by value.
    """

    guardrail_name: str
    create_kwargs: Mapping[str, Any] = field(default_factory=dict, hash=False)
    validate_kwargs: Mapping[str, Any] = field(default_factory=dict, hash=False)
