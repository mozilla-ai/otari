"""What a caller may be told about the model behind an alias."""

import re


def hide_alias_target(text: str, *, alias: str | None, instance: str, model: str, dispatch_model: str) -> str:
    """``text`` with the target of the caller's alias replaced by the alias.

    A provider's message often names the model it was asked for, which for an
    aliased request is the target the alias exists to hide. The target is given
    by its parts, including the ``dispatch_model`` selector sent to the provider
    SDK. ``text`` is returned unchanged when the request named no alias.
    """
    if alias is None:
        return text
    targets = {f"{instance}:{model}", dispatch_model, model}
    # One pass, longest first, so a qualified name is replaced whole and an alias
    # that contains its target is not rewritten again.
    pattern = "|".join(re.escape(target) for target in sorted(targets, key=len, reverse=True))
    return re.sub(pattern, lambda _match: alias, text)
