"""Which usage rows count, for the readers that must not count all of them.

The column carries provenance: the bare slug ``gateway`` for a request Otari served,
and a source slug (``claude_code``, ``codex``) for usage imported through
``POST /api/v1/usage/external-events``. Hosted history adds a third shape, because a row
backfilled from otari.ai keeps its origin under a legacy prefix: traffic that
deployment served itself arrives as ``otari-ai:gateway``, and usage a customer had
imported there as ``otari-ai:claude_code``.

So the question is about the slug behind the prefix, never the prefix itself. A
blanket ``otari-ai:%`` match would be wrong in the direction that breaks a feature:
``otari-ai:claude_code`` is imported usage that happens to have been migrated, and the
operator surface whose whole purpose is repricing imported usage has to keep reaching
it.

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

from sqlalchemy import ColumnElement

# The slug on a row this gateway served itself.
SERVED_HERE_SLUG = "gateway"

# What a hosted-history backfill puts in front of the origin's own slug so a migrated
# row stays distinguishable from one recorded live here (otari-ai#1798). Reserving it
# against an importer claiming it is the ingest's business, not this module's; today
# that lives as a local constant in ``services/external_usage_service.py``.
LEGACY_ORIGIN_PREFIX = "otari-ai:"

# Every spelling of "served here": the bare slug, and the same slug behind the legacy
# prefix. The prefix says a row was migrated; the slug behind it says what kind of row
# it is, and only the slug answers this question.
SERVED_HERE_SOURCES: tuple[str, ...] = (SERVED_HERE_SLUG, f"{LEGACY_ORIGIN_PREFIX}{SERVED_HERE_SLUG}")


# The endpoint label on a usage row the dashboard's own Playground produced.
# An identifier, not a URL: it is written to rows and keeps its value if the
# route ever moves, exactly as ``chat.USAGE_ENDPOINT`` does.
PLAYGROUND_USAGE_ENDPOINT = "/v1/playground/chat/completions"


def served_here(column: Any) -> ColumnElement[bool]:
    """Match rows this deployment served itself, whether recorded live or migrated."""
    return cast("ColumnElement[bool]", column.in_(SERVED_HERE_SOURCES))


def not_served_here(column: Any) -> ColumnElement[bool]:
    """Match rows this deployment did not serve: imported usage, live or migrated."""
    return cast("ColumnElement[bool]", column.not_in(SERVED_HERE_SOURCES))


def is_served_here(source: str) -> bool:
    """The same question about a row already in memory, for a response field."""
    return source in SERVED_HERE_SOURCES


def integration_traffic(column: Any) -> ColumnElement[bool]:
    """Match rows made from outside the product, over ``usage_logs.endpoint``.

    The activation guide's question, and the reason it cannot simply ask
    :func:`served_here`. The guide closes when a workspace first calls this
    gateway *from somebody\'s own code*, which is the milestone it was written to
    celebrate; a message typed into our own Playground is the product being
    demonstrated, not integrated, so closing on one would congratulate somebody
    for something they have not done yet and then never offer the guide again.
    """
    return cast("ColumnElement[bool]", column != PLAYGROUND_USAGE_ENDPOINT)
