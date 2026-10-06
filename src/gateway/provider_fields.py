"""Fields a provider returns beyond the OpenAI schema, gathered under ``provider_specific_fields``.

Exa's answer endpoint, for example, puts ``citations`` on the message. The field
stays where the provider put it, and a copy goes under
``message.provider_specific_fields`` (``delta.provider_specific_fields`` on a
stream), the convention LiteLLM established and clients such as Firefox read.
"""

from typing import Any, TypeVar

__all__ = ["PROVIDER_SPECIFIC_FIELDS", "surface_provider_fields"]

PROVIDER_SPECIFIC_FIELDS = "provider_specific_fields"

_T = TypeVar("_T")


def _gather(holder: Any) -> None:
    extras = getattr(holder, "model_extra", None)
    if not extras:
        return
    unknown = {name: value for name, value in extras.items() if name != PROVIDER_SPECIFIC_FIELDS}
    if not unknown:
        return
    existing = extras.get(PROVIDER_SPECIFIC_FIELDS)
    merged = {**unknown, **existing} if isinstance(existing, dict) else unknown
    setattr(holder, PROVIDER_SPECIFIC_FIELDS, merged)


def surface_provider_fields(obj: _T) -> _T:
    """Copy each choice's unknown message (or delta) fields under ``provider_specific_fields``, in place.

    A field the provider already sent under ``provider_specific_fields`` wins over
    a copy of the same name. Objects with no choices are left untouched.
    """
    for choice in getattr(obj, "choices", None) or ():
        for holder in (getattr(choice, "message", None), getattr(choice, "delta", None)):
            if holder is not None:
                _gather(holder)
    return obj
