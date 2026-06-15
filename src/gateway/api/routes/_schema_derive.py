"""Derive request schemas from any-llm's typed ``*Params`` models.

The gateway forwards declared request fields to any-llm's ``acompletion`` /
``aresponses`` / ``amessages`` / ... calls. Hand-maintaining each request schema
let it drift from the param surface any-llm actually accepts, silently dropping
params before the provider call (this is how ``reasoning_effort`` and nine other
chat params ended up being ignored, mozilla-ai/otari#150, #152).

Deriving the request schema's field set from the matching any-llm ``*Params``
model removes that drift structurally: a derived schema cannot omit a param
any-llm understands, and a new any-llm param is picked up automatically.

Usage: build a base with :func:`derive_request_base`, then subclass it to layer
on gateway-internal fields (``mcp_servers``, ``guardrails``, ...), tighten a
field (a validator, ``min_length``), or replace an any-llm annotation that is
unwieldy for a JSON request body (e.g. the Responses ``input`` union, or
``response_format``'s ``type`` member). A subclass field redefinition fully
overrides the derived one, so the OpenAPI schema reflects the override.
"""

from typing import Any

from pydantic import BaseModel, create_model

# any-llm names a couple of params differently from the public wire field the
# gateway exposes; ``CompletionParams`` / ``ImageGenerationParams`` /
# ``AudioSpeechParams`` call the model ``model_id`` while the gateway (and the
# any-llm public functions) take ``model``.
PARAM_FIELD_RENAMES: dict[str, str] = {"model_id": "model"}


def derive_request_base(
    params_model: type[BaseModel],
    *,
    base_name: str | None = None,
    rename: dict[str, str] = PARAM_FIELD_RENAMES,
) -> Any:
    """Build a Pydantic base model mirroring ``params_model``'s fields.

    Each any-llm param becomes a field with the same annotation and default
    (required params stay required), renamed via ``rename``. Subclass the result
    to add gateway-internal fields or override unwieldy annotations.

    The return type is annotated ``Any`` so the result can be used as a base
    class: it is a dynamically-built model, so a static checker cannot otherwise
    treat the variable as a type. Subclasses remain ordinary, statically-known
    classes usable as FastAPI request-body annotations.
    """
    fields: dict[str, Any] = {}
    for field_name, field_info in params_model.model_fields.items():
        target = rename.get(field_name, field_name)
        default = ... if field_info.is_required() else field_info.default
        fields[target] = (field_info.annotation, default)
    return create_model(base_name or f"{params_model.__name__}Base", **fields)
