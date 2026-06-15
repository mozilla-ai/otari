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

from pydantic import BaseModel, Field, create_model

# any-llm names a couple of params differently from the public wire field the
# gateway exposes; ``CompletionParams`` / ``ImageGenerationParams`` /
# ``AudioSpeechParams`` call the model ``model_id`` while the gateway (and the
# any-llm public functions) take ``model``.
PARAM_FIELD_RENAMES: dict[str, str] = {"model_id": "model"}

# any-llm's ``*Params`` are provider-call models, so a future any-llm version
# could add a credential / transport / provider-selection field. Derivation
# picks up new fields automatically (the point for benign params), but exposing
# one of these as a client-settable request field would let a caller override an
# operator-controlled value: the provider-call merge spreads request fields last
# (e.g. ``{**get_provider_kwargs(...), **request_fields}``), so a client value
# would win. The gateway resolves these itself (from ``config`` / the platform
# service), so they must never be derived onto a public request schema, and they
# are also stripped before forwarding (see ``_tools._strip_gateway_fields``) to
# cover schemas that allow extra fields (the Responses request uses
# ``extra="allow"``).
SENSITIVE_PARAM_FIELDS: frozenset[str] = frozenset(
    {
        "api_key",
        "api_base",
        "base_url",
        "provider",
        "organization",
        "api_version",
        "client",
        "credentials",
        "aws_access_key_id",
        "aws_secret_access_key",
    }
)


def derive_request_base(
    params_model: type[BaseModel],
    *,
    base_name: str | None = None,
    rename: dict[str, str] | None = None,
) -> Any:
    """Build a Pydantic base model mirroring ``params_model``'s fields.

    Each any-llm param becomes a field with the same annotation and default
    (required params stay required, ``default_factory`` is preserved), renamed
    via ``rename`` (defaults to :data:`PARAM_FIELD_RENAMES`). Fields in
    :data:`SENSITIVE_PARAM_FIELDS` are never derived. Subclass the result to add
    gateway-internal fields or override unwieldy annotations.

    The return type is annotated ``Any`` so the result can be used as a base
    class: it is a dynamically-built model, so a static checker cannot otherwise
    treat the variable as a type. Subclasses remain ordinary, statically-known
    classes usable as FastAPI request-body annotations.
    """
    rename = PARAM_FIELD_RENAMES if rename is None else rename
    fields: dict[str, Any] = {}
    for field_name, field_info in params_model.model_fields.items():
        if field_name in SENSITIVE_PARAM_FIELDS:
            continue
        target = rename.get(field_name, field_name)
        if field_info.is_required():
            spec: Any = (field_info.annotation, ...)
        elif field_info.default_factory is not None:
            spec = (field_info.annotation, Field(default_factory=field_info.default_factory))
        else:
            spec = (field_info.annotation, field_info.default)
        fields[target] = spec
    return create_model(base_name or f"{params_model.__name__}Base", **fields)
