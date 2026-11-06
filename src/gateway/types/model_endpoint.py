"""Which Otari routes serve a model, from the protocols models.dev says its host takes.

A path is below the API base: ``/chat/completions``, not ``/api/v1/chat/completions``.
A protocol with no route here names none: a vendor's native protocol (``converse``,
``generate-content``), which any-llm calls behind the OpenAI-shaped routes; a live,
two-way session (``realtime``, ``live``); and ``fim`` and ``ocr``, which Otari has no
route for.
"""

from collections.abc import Iterable
from typing import Literal

ModelEndpoint = Literal[
    "/chat/completions",
    "/responses",
    "/messages",
    "/images/generations",
    "/images/edits",
    "/embeddings",
    "/audio/speech",
    "/audio/transcriptions",
    "/systemone",
    "/rerank",
    "/decisions",
    "/moderations",
    "/videos",
]

# models.dev's v2 ``api`` protocols. ``decisions`` and ``systemone`` are one handler
# under two paths, and each is named, so a client gets the path its SDK expects.
# Both take only a ``decision_providers`` selector, so they name the route a
# protocol maps to, not one every selector of the model is served on. ``images``
# names both image routes: models.dev does not say which image models can edit.
_MODELS_DEV_PROTOCOLS: dict[str, tuple[ModelEndpoint, ...]] = {
    "chat-completions": ("/chat/completions",),
    "responses": ("/responses",),
    "messages": ("/messages",),
    "images": ("/images/generations", "/images/edits"),
    "embeddings": ("/embeddings",),
    "speech": ("/audio/speech",),
    "transcriptions": ("/audio/transcriptions",),
    "systemone": ("/systemone",),
    "rerank": ("/rerank",),
    "decisions": ("/decisions",),
    "moderations": ("/moderations",),
    "videos": ("/videos",),
}


def models_dev_endpoints(protocols: Iterable[str]) -> tuple[ModelEndpoint, ...]:
    """The routes that serve ``protocols``, in their order, each once."""
    return tuple(dict.fromkeys(route for p in protocols for route in _MODELS_DEV_PROTOCOLS.get(p, ())))
