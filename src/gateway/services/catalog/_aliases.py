"""What a caller may be told about the model behind an alias."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # provider_kwargs resolves selectors through this package, so only its type is imported.
    from gateway.services.provider_kwargs import ResolvedProvider


def hide_alias_target(text: str, resolved: ResolvedProvider) -> str:
    """``text`` with the target of the caller's alias replaced by the alias.

    A provider's message often names the model it was asked for, which for an
    aliased request is the target the alias exists to hide. ``text`` is returned
    unchanged when the request named no alias.
    """
    alias = resolved.alias
    if alias is None:
        return text
    targets = {f"{resolved.instance}:{resolved.model}", resolved.dispatch_model, resolved.model}
    # One pass, longest first, so a qualified name is replaced whole and an alias
    # that contains its target is not rewritten again.
    pattern = "|".join(re.escape(target) for target in sorted(targets, key=len, reverse=True))
    return re.sub(pattern, lambda _match: alias, text)
