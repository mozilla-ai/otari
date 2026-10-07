"""Which session a request belongs to, and which agent sent it.

Requests are grouped into a session only when the client says which one, either
on purpose (a header, the ``session_label`` field, a ``session_id`` tag) or
because its agent harness sends an id of its own. The gateway never infers that
two requests belong together from their content.

The session's trace id hashes the workspace and the user in, so two people who
choose the same label get two sessions, and an id a client chose can never land
in another tenant's or another user's trace.
"""

import hashlib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from gateway.core.config import CONVERSATION_HEADER
from gateway.ports.trace_storage_port import is_identifier

# The request tag a client can name its session with.
SESSION_TAG = "session_id"

# Headers in which an agent harness sends its own session id, and the harness each
# one belongs to. Claude Code sends ``x-claude-code-session-id`` on every request of
# a session. Other harnesses are added as their requests are confirmed.
HARNESS_SESSION_HEADERS: Mapping[str, str] = {"x-claude-code-session-id": "claude-code"}

# ``User-Agent`` product names a harness is known by, where it differs from the name shown.
_HARNESS_PRODUCTS: Mapping[str, str] = {"claude-cli": "claude-code"}

_MAX_KEY_LENGTH = 512


@dataclass(frozen=True)
class SessionRef:
    """The signal that named a request's session: where it came from, and the key it carries."""

    source: str
    key: str

    def trace_id(self, workspace_id: uuid.UUID, user_id: str | None, api_key_id: str | None = None) -> str:
        """The session's trace id, scoped to the workspace and the user, or to the key where it has no user."""
        owner = user_id if user_id is not None else f"key:{api_key_id or ''}"
        material = "\x00".join((str(workspace_id), owner, self.source, self.key))
        return "s-" + hashlib.sha256(material.encode()).hexdigest()[:40]

    def label(self) -> str | None:
        """The key as it may travel or be shown: only when it is an opaque id, never prose."""
        return self.key if is_identifier(self.key) else None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value[:_MAX_KEY_LENGTH] if value else None


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
    return None


def harness_of(headers: Mapping[str, str]) -> str | None:
    """The agent or client that sent the request, from its ``User-Agent`` product name.

    Only the product name is kept, mapped to the name the harness is known by, and
    only when it is identifier-shaped. A request a harness marked with its own
    session header is that harness whatever its ``User-Agent`` says.
    """
    for header, harness in HARNESS_SESSION_HEADERS.items():
        if headers.get(header):
            return harness
    agent = (headers.get("user-agent") or "").strip()
    if not agent:
        return None
    product = agent.split()[0].split("/")[0].lower()
    product = _HARNESS_PRODUCTS.get(product, product)
    return product if is_identifier(product) and len(product) <= 64 else None
