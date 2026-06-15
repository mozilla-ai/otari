"""Guard that every derivable request schema is derived from any-llm's ``*Params``.

The gateway forwards declared request fields to any-llm; a hand-maintained schema
could omit a param any-llm forwards and silently drop it before the provider call
(this is how ``reasoning_effort`` and nine other chat params were ignored,
mozilla-ai/otari#150, #152). Each inference endpoint with a typed any-llm
``*Params`` source of truth now derives its request schema from that model (see
``gateway.api.routes._schema_derive``), so the drift is impossible by construction.

These tests pin that property:

* every derived schema is a superset of its ``*Params`` (no param dropped, even
  when any-llm adds a new one), and
* the derive-vs-hand-written split stays explicit, so a new inference endpoint
  with a typed source is a conscious decision rather than a silent gap.

Endpoints with no typed any-llm ``*Params`` (embeddings, moderations, rerank,
batches; mozilla-ai/otari#154) take loose ``**kwargs`` upstream, so there is
nothing to derive from or pin against. They stay hand-written and are listed in
``NO_TYPED_PARAMS_SOURCE`` for the record.
"""

import pytest
from any_llm.types.audio import AudioSpeechParams
from any_llm.types.completion import CompletionParams
from any_llm.types.image import ImageGenerationParams
from any_llm.types.messages import MessagesParams
from any_llm.types.responses import ResponsesParams
from pydantic import BaseModel

from gateway.api.routes._schema_derive import PARAM_FIELD_RENAMES
from gateway.api.routes.audio import AudioSpeechRequest
from gateway.api.routes.chat import ChatCompletionRequest
from gateway.api.routes.images import ImageGenerationRequest
from gateway.api.routes.messages import MessagesRequest
from gateway.api.routes.responses import ResponsesRequest

# Inference endpoints whose request schema mirrors an any-llm typed ``*Params``
# model: (gateway request model, any-llm Params model). The request model derives
# its field set from the Params via ``derive_request_base``.
DERIVED_SCHEMAS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "chat": (ChatCompletionRequest, CompletionParams),
    "responses": (ResponsesRequest, ResponsesParams),
    "messages": (MessagesRequest, MessagesParams),
    "images": (ImageGenerationRequest, ImageGenerationParams),
    "audio": (AudioSpeechRequest, AudioSpeechParams),
}

# Inference endpoints with no typed any-llm ``*Params`` source of truth. They take
# loose ``**kwargs`` upstream, so there is nothing to derive from; they stay
# hand-written. Tracked in mozilla-ai/otari#154.
NO_TYPED_PARAMS_SOURCE = {"embeddings", "moderations", "rerank", "batches"}


@pytest.mark.parametrize("endpoint", sorted(DERIVED_SCHEMAS))
def test_request_schema_covers_any_llm_params(endpoint: str) -> None:
    """The derived request schema accepts every param its any-llm ``*Params`` declares."""
    request_model, params_model = DERIVED_SCHEMAS[endpoint]
    expected = {PARAM_FIELD_RENAMES.get(name, name) for name in params_model.model_fields}
    actual = set(request_model.model_fields)

    dropped = expected - actual
    assert not dropped, (
        f"{request_model.__name__} does not accept any-llm {params_model.__name__} params "
        f"{sorted(dropped)}, so they would be silently dropped before the provider call. "
        f"It must subclass the base produced by derive_request_base(...) so the field set "
        f"stays in sync."
    )


@pytest.mark.parametrize("endpoint", sorted(DERIVED_SCHEMAS))
def test_renamed_params_are_exposed_under_the_wire_name(endpoint: str) -> None:
    """A renamed any-llm param (e.g. ``model_id`` -> ``model``) is exposed under the wire name."""
    request_model, params_model = DERIVED_SCHEMAS[endpoint]
    for internal_name, wire_name in PARAM_FIELD_RENAMES.items():
        if internal_name in params_model.model_fields:
            assert wire_name in request_model.model_fields, (
                f"{request_model.__name__} renames any-llm's '{internal_name}' to '{wire_name}', "
                f"but '{wire_name}' is missing from the schema"
            )
            assert internal_name not in request_model.model_fields, (
                f"{request_model.__name__} should expose any-llm's '{internal_name}' as "
                f"'{wire_name}', not leak the internal name"
            )


def test_derived_and_hand_written_endpoints_are_disjoint() -> None:
    """An endpoint is either derived from a typed ``*Params`` or documented as hand-written, not both."""
    overlap = set(DERIVED_SCHEMAS) & NO_TYPED_PARAMS_SOURCE
    assert not overlap, f"endpoints claim both a typed source and none: {sorted(overlap)}"
