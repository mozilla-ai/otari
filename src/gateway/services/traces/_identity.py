"""Which session a request belongs to, and which agent sent it.

Requests are grouped into a session only when the client says which one, either
on purpose (a header, the ``session_label`` field, a ``session_id`` tag) or
because its agent harness sends an id of its own. The gateway never infers that
two requests belong together from their content.

The session's trace id hashes the workspace and the user in, so two people who
choose the same label get two sessions, and an id a client chose can never land
in another tenant's or another user's trace. A key the client chose on purpose is
also the session's name, where it is identifier-shaped, so a reader can find the
session by the id the client knows it by.
"""

import hashlib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import assert_never

from gateway.core.config import CONVERSATION_HEADER
from gateway.types.traces import ClientLanguage, SessionSource, is_client_version, is_identifier

# The request tag a client can name its session with.
SESSION_TAG = "session_id"

# Headers in which an agent harness sends its own session id, and the harness each
# one belongs to. Each name is the harness's own, so it counts from any client: Claude
# Code sends ``x-claude-code-session-id`` on every request of a session, opencode
# ``x-opencode-session-id``, Amp its thread id, and goose its session id. An opencode or
# goose subagent runs in a session of its own, so it sends its own id.
HARNESS_SESSION_HEADERS: Mapping[str, str] = {
    "x-claude-code-session-id": "claude-code",
    "x-opencode-session-id": "opencode",
    "x-amp-thread-id": "amp",
    "agent-session-id": "goose",
}

# Session headers that name no harness: LiteLLM's convention, which OpenHands sends with
# its conversation id. The harness then comes from ``User-Agent``.
_SHARED_SESSION_HEADERS = ("x-litellm-session-id",)

# Headers a harness sends only on a subagent's requests. A subagent shares its parent's
# session id and works inside the parent's turn, so the task it was handed is no new prompt.
HARNESS_SUBAGENT_HEADERS = frozenset({"x-claude-code-agent-id"})


@dataclass(frozen=True)
class _MarkedHarness:
    """A harness that sends its session id under a generic name, read only on its own requests.

    A request is the harness's when one of ``markers`` (a header and the prefix its value
    starts with) matches. Its session headers are read in order, the first one set winning.
    """

    harness: str
    markers: tuple[tuple[str, str], ...]
    session_headers: tuple[str, ...]


# Harnesses whose session header names are generic, so they count only beside the
# harness's own mark. Codex sends its thread id as ``thread-id`` and its root thread's id
# as ``session-id``; the thread is the session, so a subagent gets one of its own. Kilo
# Code and Crush give a subagent a session of its own too. pi sends ``session-id`` on the
# Responses API and the other names on the APIs where its session affinity is turned on.
# Roo Code sends ``session_id`` only from its OpenAI provider. Mistral Vibe sends its root
# session's id as ``x-affinity``, so a subagent shares it. OpenClaw sends ``session_id`` and
# ``x-session-affinity`` once its session affinity is turned on, and its own User-Agent only
# where it is configured. oh-my-pi sends ``x-claude-code-session-id`` on the Anthropic API,
# which names the session; its mark names the harness.
_MARKED_HARNESSES = (
    _MarkedHarness("codex", (("originator", "codex"), ("user-agent", "codex")), ("thread-id", "session-id")),
    _MarkedHarness("kilo-code", (("user-agent", "kilo-code/"), ("user-agent", "kilo/")), ("x-session-id",)),
    _MarkedHarness("crush", (("user-agent", "charm-crush/"),), ("x-session-id",)),
    _MarkedHarness("pi", (("user-agent", "pi/"),), ("session-id", "session_id", "x-session-affinity", "x-session-id")),
    _MarkedHarness("roo-code", (("originator", "roo-code"), ("user-agent", "roo-code/")), ("session_id",)),
    _MarkedHarness(
        "mistral-vibe",
        (("user-agent", "mistral-vibe/"), ("user-agent", "mistral-client-python/mistral-vibe/")),
        ("x-affinity",),
    ),
    _MarkedHarness(
        "openclaw", (("originator", "openclaw"), ("user-agent", "openclaw/")), ("session_id", "x-session-affinity")
    ),
    _MarkedHarness("oh-my-pi", (("user-agent", "omp/"),), ()),
)

# ``User-Agent`` product names a harness is known by, where it differs from the name shown.
_HARNESS_PRODUCTS: Mapping[str, str] = {
    "claude-cli": "claude-code",
    "githubcopilotchat": "github-copilot",
    "hermesagent": "hermes-agent",
    "kimi-code-cli": "kimi-code",
    "qwencode": "qwen-code",
    "google-genai-sdk": "google-genai",
    "mistralai-azure": "mistralai",
    "mistralai-gcp": "mistralai",
    "ollama-python": "ollama",
}

# The names a language goes by in an SDK's headers or a ``User-Agent``, and the HTTP
# libraries whose default ``User-Agent`` names no language but implies one.
_LANGUAGE_NAMES: Mapping[str, ClientLanguage] = {
    "python": "python",
    "cpython": "python",
    "gl-python": "python",
    "python-httpx": "python",
    "python-requests": "python",
    "python-urllib": "python",
    "python-urllib3": "python",
    "aiohttp": "python",
    "js": "javascript",
    "javascript": "javascript",
    "typescript": "javascript",
    "node": "javascript",
    "gl-node": "javascript",
    "undici": "javascript",
    "node-fetch": "javascript",
    "axios": "javascript",
    "got": "javascript",
    "deno": "javascript",
    "bun": "javascript",
    "go": "go",
    "golang": "go",
    "go-http-client": "go",
    "java": "java",
    "java-http-client": "java",
    "okhttp": "java",
    "apache-httpclient": "java",
    "rust": "rust",
    "ruby": "ruby",
    "faraday": "ruby",
}

# Headers in which an SDK names its own language: Stainless's (OpenAI, Anthropic, Groq and
# others) and Fern's (Cohere).
_LANGUAGE_HEADERS = ("x-stainless-lang", "x-fern-language")

# The languages a Stainless SDK names after its client class, ``OpenAI/Python 1.99``. Only
# languages, never the HTTP libraries ``_LANGUAGE_NAMES`` also holds: ``my-app/okhttp`` is
# no SDK, and the word after it is another product, not a version.
_STAINLESS_LANGUAGES = frozenset({"python", "js", "node", "typescript", "go", "java", "kotlin", "ruby"})

# A Stainless SDK's client classes that are one SDK, by class name less any ``Async``:
# the cloud variants of the OpenAI and Anthropic clients.
_SDK_CLIENT_CLASSES: Mapping[str, str] = {
    "azureopenai": "openai",
    "anthropicaws": "anthropic",
    "anthropicbedrock": "anthropic",
    "anthropicbedrockmantle": "anthropic",
    "anthropicfoundry": "anthropic",
    "anthropicgooglecloud": "anthropic",
    "anthropicvertex": "anthropic",
}

# ``User-Agent`` products whose version is not the client's: Go's is the HTTP version,
# aiohttp's leading ``Python/3.12`` is the interpreter's, and a browser's ``Mozilla/5.0``
# is a compatibility token.
_VERSIONLESS_PRODUCTS = frozenset({"go-http-client", "python", "mozilla"})

# A Speakeasy SDK's User-Agent is ``speakeasy-sdk/<language> <sdk version> <generator
# version> <document version> <package>``: the field each is read from.
_SPEAKEASY_VERSION = 1
_SPEAKEASY_PACKAGE = 4

_MAX_KEY_LENGTH = 512


def _namespace(source: SessionSource) -> str:
    """The namespace a session's key is hashed in.

    An exporter's session id is its harness's own (Claude Code's ``session.id`` is the
    ``x-claude-code-session-id`` it routes with, and ``otari hook`` reports under the
    id its harness hands it), so the two hash alike: a session routed through the
    gateway and the spans exported for it land on one trace. A client's chosen key
    stays apart.
    """
    match source:
        case "harness" | "otlp":
            return "harness"
        case "client" | "none":
            return source
        case _:
            assert_never(source)


@dataclass(frozen=True)
class SessionRef:
    """The signal that named a request's session: where it came from, and the key it carries."""

    source: SessionSource
    key: str

    def trace_id(self, workspace_id: uuid.UUID, user_id: str | None, api_key_id: str | None = None) -> str:
        """The session's trace id, scoped to the workspace and the user, or to the key where it has no user."""
        owner = user_id if user_id is not None else f"key:{api_key_id or ''}"
        material = "\x00".join((str(workspace_id), owner, _namespace(self.source), self.key))
        return "s-" + hashlib.sha256(material.encode()).hexdigest()[:40]

    @property
    def name(self) -> str | None:
        """The session's display name: the key a client chose on purpose, when it is identifier-shaped.

        A harness's or an exporter's key is its own generated id, so it names nothing.
        """
        return self.key if self.source == "client" and is_identifier(self.key) else None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value[:_MAX_KEY_LENGTH] if value else None


def _marked_harness(headers: Mapping[str, str]) -> _MarkedHarness | None:
    for marked in _MARKED_HARNESSES:
        if any((headers.get(header) or "").strip().lower().startswith(prefix) for header, prefix in marked.markers):
            return marked
    return None


def resolve_session(
    headers: Mapping[str, str], *, session_label: str | None, tags: Mapping[str, str] | None
) -> SessionRef | None:
    """The request's session, or None when nothing names one.

    A session the client named on purpose wins over one its harness sends, so a
    caller can group a harness's requests differently when it wants to.
    """
    for client_key in (headers.get(CONVERSATION_HEADER), session_label, (tags or {}).get(SESSION_TAG)):
        if (key := _clean(client_key)) is not None:
            return SessionRef(source="client", key=key)
    for header in HARNESS_SESSION_HEADERS:
        if (key := _clean(headers.get(header))) is not None:
            return SessionRef(source="harness", key=key)
    marked = _marked_harness(headers)
    for header in (*(marked.session_headers if marked else ()), *_SHARED_SESSION_HEADERS):
        if (key := _clean(headers.get(header))) is not None:
            return SessionRef(source="harness", key=key)
    return None


def is_subagent(headers: Mapping[str, str]) -> bool:
    """Whether the request's harness marked it as one of a subagent's."""
    return any(_clean(headers.get(header)) is not None for header in HARNESS_SUBAGENT_HEADERS)


def harness_of(headers: Mapping[str, str]) -> str | None:
    """The agent or client that sent the request, from its ``User-Agent`` product name.

    Only the product name is kept, mapped to the name the harness is known by, and
    only when it is identifier-shaped. A harness's own mark comes first, since one
    harness can send another's session header (oh-my-pi sends Claude Code's); then a
    request carrying a harness's own session header is that harness whatever its
    ``User-Agent`` says.
    """
    if (marked := _marked_harness(headers)) is not None:
        return marked.harness
    for header, harness in HARNESS_SESSION_HEADERS.items():
        if headers.get(header):
            return harness
    return _user_agent_product(headers)


def _user_agent_product(headers: Mapping[str, str]) -> str | None:
    """The client the ``User-Agent`` names by its first product, under the name it is shown by."""
    tokens = (headers.get("user-agent") or "").split()
    if not tokens:
        return None
    segments = tokens[0].split("/")
    product = segments[0].lower()
    if product == "speakeasy-sdk" and len(tokens) > _SPEAKEASY_PACKAGE:
        # A Speakeasy SDK names its package in the fifth field, ``... 1.0.0 mistralai``.
        product = tokens[_SPEAKEASY_PACKAGE].lower()
    elif _is_stainless(segments):
        # A Stainless SDK names itself by its client class, ``AsyncOpenAI/Python 1.99``, so
        # the async twin and the cloud variants are one SDK.
        product = product.removeprefix("async")
        product = _SDK_CLIENT_CLASSES.get(product, product)
    product = _HARNESS_PRODUCTS.get(product, product)
    return product if is_identifier(product) and len(product) <= 64 else None


def _is_stainless(segments: list[str]) -> bool:
    """Whether a ``User-Agent``'s first product is a Stainless SDK's ``Class/Language``."""
    return len(segments) > 1 and segments[1].lower() in _STAINLESS_LANGUAGES


def _user_agent_names(headers: Mapping[str, str], harness: str) -> bool:
    """Whether the ``User-Agent`` is the harness's own, so a version it gives is the harness's."""
    if _user_agent_product(headers) == harness:
        return True
    agent = (headers.get("user-agent") or "").strip().lower()
    return any(
        marked.harness == harness and header == "user-agent" and agent.startswith(prefix)
        for marked in _MARKED_HARNESSES
        for header, prefix in marked.markers
    )


def _client_language_of(headers: Mapping[str, str]) -> ClientLanguage | None:
    """The language the client was written in, or None when nothing says.

    An SDK's own language header comes first. Otherwise the ``User-Agent`` is read
    word by word, each split at ``/``, for a language's name (``OpenAI/Python``,
    ``OpenAPI-Generator/1.0.0/python``, ``node/24.11.0``) or an HTTP library that
    implies one (``python-httpx``, ``Go-http-client``).
    """
    for header in _LANGUAGE_HEADERS:
        if (language := _LANGUAGE_NAMES.get((headers.get(header) or "").strip().lower())) is not None:
            return language
    for token in (headers.get("user-agent") or "").split():
        for segment in token.strip("();,").split("/"):
            if (language := _LANGUAGE_NAMES.get(segment.lower())) is not None:
                return language
    return None


def _client_version_of(headers: Mapping[str, str]) -> str | None:
    """The version the ``User-Agent`` gives its client, or None.

    The version follows the product (``claude-cli/2.1.291``, ``OpenAPI-Generator/1.0.0/python``).
    Only the SDK formats that put it in a word of its own are read there (``OpenAI/Python
    1.99.0``, Speakeasy's), since elsewhere the next word is another product. An SDK's
    package-version header is not read: on a harness built on the SDK it is the SDK's
    version, not the harness's.
    """
    tokens = (headers.get("user-agent") or "").split()
    if not tokens:
        return None
    segments = tokens[0].split("/")
    product = segments[0].lower()
    if product in _VERSIONLESS_PRODUCTS:
        return None
    if product == "speakeasy-sdk":
        candidates = tokens[_SPEAKEASY_VERSION : _SPEAKEASY_VERSION + 1]
    elif _is_stainless(segments):
        candidates = tokens[1:2]
    else:
        candidates = segments[1:]
    for candidate in candidates:
        version = candidate.strip().removeprefix("v")
        if is_client_version(version):
            return version
    return None


@dataclass(frozen=True)
class RequestClient:
    """The client that sent a request, as its headers name it: its harness, language and version."""

    harness: str | None
    language: ClientLanguage | None
    version: str | None

    @classmethod
    def of(cls, headers: Mapping[str, str]) -> "RequestClient":
        """Read the client off a request's headers.

        The version is kept only when the ``User-Agent`` is the harness's own, so a
        proxy's version never lands on the harness its session header names.
        """
        harness = harness_of(headers)
        version = _client_version_of(headers) if harness and _user_agent_names(headers, harness) else None
        return cls(harness, _client_language_of(headers), version)
