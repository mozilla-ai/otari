"""Which usage rows count, for the readers that must not count all of them.

The column carries provenance: the bare slug ``gateway`` for a request Otari served,
and a source slug (``claude_code``, ``codex``) for usage imported through
``POST /api/v1/usage/external-events``.

Three callers ask this same question, which is why it is here rather than a literal
in each: the usage admin mutations exclude served-here rows (they may only touch
imported usage, and ``counts_toward_budget`` does not tell them apart), the count
behind their "select all N matching" excludes the same rows so the number an operator
confirms is one the mutation can reach, and the activation guide requires a served-here
row (imported usage is somebody else's traffic, so it is never a workspace's first
request to this gateway).

The Playground is a fourth reader and deliberately not a fourth *source*. Its
rows were served here, and they must stay in that set so the imported-usage
mutations keep their hands off them; what tells them apart is the endpoint
label below, which says which surface made the call rather than who served it.
Two questions, two columns, and it lives here beside the other one because both
the route that writes it and the service that filters on it need the name, and a
service may not import the API layer (`scripts/check_architecture.py`).
"""

from typing import Any, cast

from sqlalchemy import ColumnElement, and_, select

from gateway.models.api_keys import APIKey

# The slug on a row this gateway served itself. The ingest reserves it, so an import
# cannot claim to be traffic served here.
SERVED_HERE_SLUG = "gateway"


# The endpoint label on a usage row the dashboard's own Playground produced.
# An identifier, not a URL: it is written to rows and keeps its value if the
# route ever moves, exactly as ``chat.USAGE_ENDPOINT`` does.
PLAYGROUND_USAGE_ENDPOINT = "/v1/playground/chat/completions"


def served_here(column: Any) -> ColumnElement[bool]:
    """Match rows this deployment served itself."""
    return cast("ColumnElement[bool]", column == SERVED_HERE_SLUG)


def not_served_here(column: Any) -> ColumnElement[bool]:
    """Match rows this deployment did not serve: imported usage."""
    return cast("ColumnElement[bool]", column != SERVED_HERE_SLUG)


def is_served_here(source: str) -> bool:
    """The same question about a row already in memory, for a response field."""
    return source == SERVED_HERE_SLUG


def integration_traffic(endpoint: Any, api_key_id: Any) -> ColumnElement[bool]:
    """Match rows made from outside the product, over ``usage_logs``.

    The activation guide's question, and the reason it cannot simply ask
    :func:`served_here`. The guide closes when a workspace first calls this
    gateway *from somebody\'s own code*, which is the milestone it was written to
    celebrate; a message typed into our own Playground is the product being
    demonstrated, not integrated, so closing on one would congratulate somebody
    for something they have not done yet and then never offer the guide again.

    Two columns, because the Playground answers to two shapes and the endpoint
    label only catches one of them. Where this deployment runs the completion
    itself, the row carries :data:`PLAYGROUND_USAGE_ENDPOINT` and nothing else is
    needed. Where it does not, which is a hosted control plane forwarding to its
    data-plane gateway (``services/playground_dispatch``), the row is written from
    the gateway's usage report and is labelled like every other report that
    gateway sends: the surface that made the call is not on the wire, so the
    label cannot carry it. What does carry it is the credential, because the one
    the control plane forwards under is minted by this deployment for exactly
    this purpose and is marked as such.

    Taking both rather than leaving the second to each caller, for the reason the
    module exists: a rule spelled at a call site is a rule the next call site
    spells differently, and half of this one silently counts the Playground as
    an integration.
    """
    return and_(
        endpoint != PLAYGROUND_USAGE_ENDPOINT,
        # A row with no key at all (the standalone Playground writes one, and so
        # does any session-authorized request) matches nothing here and is left to
        # the endpoint half above, which is the half that knows about it.
        ~select(APIKey.id).where(APIKey.id == api_key_id, APIKey.internal_secret.is_not(None)).exists(),
    )
