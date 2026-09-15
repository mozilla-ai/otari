"""Playground: the dashboard's own chat surface, and what it may remember.

The Playground is the in-product chat page: pick a model this gateway can
route to, talk to it, compare two models side by side, and keep the exchanges
worth keeping. otari-ai#1947 asked for it back after the hosted original was
retired (otari-ai#1920), and otari#663 is the port; njbrake settled that it
lives in the OSS build rather than the overlay.

Five tables, and none of them is on the request path. A completion runs through
the ordinary pipeline and writes the ordinary ``usage_logs`` row; these hold
only what the *page* remembers between visits, for the one identity that wrote
it:

- ``playground_consent``: whether this person has agreed to content retention
  at all, split by what is retained. No row means not agreed, which is why the
  read side defaults rather than provisions.
- ``playground_conversation`` + ``playground_message``: a saved single-panel
  transcript, resumable.
- ``playground_comparison``: one rated A/B exchange, a point-in-time judgment
  rather than something to resume.
- ``playground_favorite_model``: the model keys this person pins to the top of
  every picker, per workspace, so the pin follows them off this browser.

**Owned by the tenancy identity, not by the attribution user.** Every row here
keys on ``user.id`` (the uuid a dashboard session names) and not on
``users.user_id`` (the string spend row a completion is billed to). They are the
same person in this deployment, and deliberately not the same lifecycle: the
spend row is *soft*-deleted so past usage keeps a subject to point at
(``DELETE /api/v1/users`` revokes it and leaves the ledger intact), while this
content has no such duty and should leave with the account. CASCADE from
``user.id`` is what makes deleting an identity actually delete what it wrote.

CASCADE from ``workspace.id`` for the same reason ``models/provider_keys.py``
gives: these are workspace-*owned* rows, not durable request-plane history, so
they have no meaning once the workspace is gone. A transcript is not a usage
record; the usage record is the ``usage_logs`` row the completion already wrote,
and that one still keeps its RESTRICT.

**Content columns are ``Text``, and nothing here is encrypted.** A stored
transcript is the prompt and the reply, which the page already showed the person
who wrote it, and which ``usage_logs`` does not hold. Encrypting it would buy
nothing this deployment's threat model asks for (the reader is the row's own
owner, and every route below carries that predicate) while making the rows
unreadable to the operator who backs them up. Secrets that *are* encrypted here
are credentials, and none of these tables holds one.

Style follows ``models/tenancy.py`` and ``models/provider_keys.py``: SQLModel
rather than `entities.py`'s declarative style, because the ``Public`` schemas
below are the endpoint contracts the generated dashboard client is built from,
and no ``relationship()`` is declared (lazy loading raises ``MissingGreenlet``
on an ``AsyncSession``), so the routes join explicitly.
"""

import uuid
from datetime import datetime
from typing import Annotated, Literal

from sqlalchemy import Column, Index, Text, UniqueConstraint
from sqlmodel import Field, SQLModel

from gateway.models.tenancy import CreatedAtMixin, PrimaryKeyMixin, UpdatedAtMixin

# What one save may carry. Each ceiling is enforced at the request schema, so an
# oversized save is a 422 naming the field rather than a database error, and the
# column widths below are the same numbers.
#
# The transcript limits are generous on purpose: a long tool-using exchange is
# exactly the thing somebody wants to keep, and refusing to save it is worse
# than storing it. What they bound is an unbounded write, not a realistic one.
MAX_CONVERSATION_TITLE_LENGTH = 200
MAX_MESSAGE_CONTENT_LENGTH = 200_000
MAX_MESSAGES_PER_CONVERSATION = 400
MAX_MODEL_KEY_LENGTH = 512
MAX_COMPARISON_TEXT_LENGTH = 200_000

# How many rows one identity keeps per workspace before the oldest are pruned to
# make room. A cap is needed because nothing else bounds these tables, and
# pruning rather than refusing is what the page's Save button means: somebody
# who has saved a hundred conversations wants the hundred-and-first, not an
# error telling them to go delete one first. The list endpoints page at the same
# number, so the cap is also the deepest history the page can show.
MAX_SAVED_CONVERSATIONS = 100
MAX_SAVED_COMPARISONS = 100
# Pinning is cheap and a pin list this long is already unusable as a shortcut.
MAX_FAVORITE_MODELS = 50

ChatRole = Literal["user", "assistant"]
ComparisonPreference = Literal["model_a", "model_b", "tie"]


# ==============================================================================
# Content-retention consent
# ==============================================================================


class PlaygroundConsentPublic(SQLModel):
    """What this identity has agreed the Playground may store.

    Two flags rather than one, matching what the page asks for at the moment it
    asks: saving a transcript and recording a model preference are different
    disclosures (the second stores *both* models' full answers), and the old
    page asked about each separately at the point of use. An identity with no
    stored row reads back as both false.
    """

    store_conversations: bool = False
    store_comparisons: bool = False


class PlaygroundConsentUpdate(SQLModel):
    """A partial update: an omitted flag is left as it was.

    Tri-state on purpose. The page grants one flag at a time, just in time, so a
    request that carried both would silently re-assert the other, which is the
    wrong direction for a consent record to move on its own.
    """

    store_conversations: bool | None = None
    store_comparisons: bool | None = None


class PlaygroundConsent(SQLModel, CreatedAtMixin, UpdatedAtMixin, table=True):
    """One identity's content-retention consent for the Playground.

    ``user_id`` is the primary key: an identity has one consent record or none,
    and there is nothing else to identify a row by. Not per workspace, because
    the disclosure is about what this deployment stores *about the person*, and
    a per-workspace answer would ask them the same question again for reasons
    they never chose.
    """

    __tablename__ = "playground_consent"

    user_id: uuid.UUID = Field(foreign_key="user.id", ondelete="CASCADE", primary_key=True)
    store_conversations: bool = Field(default=False, nullable=False)
    store_comparisons: bool = Field(default=False, nullable=False)

    def to_public(self) -> PlaygroundConsentPublic:
        """The wire shape: the two flags, and nothing about the row itself."""
        return PlaygroundConsentPublic(
            store_conversations=self.store_conversations,
            store_comparisons=self.store_comparisons,
        )


# ==============================================================================
# Saved conversations
# ==============================================================================


class PlaygroundMessageCreate(SQLModel):
    """One turn in a transcript being saved."""

    role: ChatRole
    content: str = Field(max_length=MAX_MESSAGE_CONTENT_LENGTH)
    # Chain-of-thought a model streamed in a field of its own, kept because the
    # page renders it in a collapsed block and a resumed transcript that lost it
    # reads as a different answer than the one that was saved.
    reasoning: str | None = Field(default=None, max_length=MAX_MESSAGE_CONTENT_LENGTH)


class PlaygroundConversationCreate(SQLModel):
    """A transcript to save, whole: there is no append-a-turn endpoint.

    The page saves on an explicit click, with the conversation it currently
    shows, so the write is one row plus its turns and a resave is a new
    conversation rather than a mutation of the old one. That is also what keeps
    the ordering column honest: ``position`` is assigned here, from the list's
    own order, and never negotiated with a client over several requests.
    """

    workspace_id: uuid.UUID
    model: str = Field(max_length=MAX_MODEL_KEY_LENGTH)
    title: str = Field(min_length=1, max_length=MAX_CONVERSATION_TITLE_LENGTH)
    messages: list[PlaygroundMessageCreate] = Field(min_length=1, max_length=MAX_MESSAGES_PER_CONVERSATION)


class PlaygroundConversationSummary(SQLModel):
    """A row in the history list: enough to recognize, not the transcript."""

    id: uuid.UUID
    workspace_id: uuid.UUID
    title: str
    model: str
    message_count: int
    created_at: datetime


class PlaygroundConversationsPublic(SQLModel):
    data: list[PlaygroundConversationSummary]


class PlaygroundMessagePublic(SQLModel):
    """One stored turn, in the order it was saved.

    No usage figures, matching what the save accepts: tokens, cost and timing
    describe the request that ran rather than the conversation, and a resumed
    transcript reporting an old request's latency as this session's would be
    lying. The billing record for that request is its ``usage_logs`` row.
    """

    role: str
    content: str
    reasoning: str | None = None


class PlaygroundMessagesPublic(SQLModel):
    data: list[PlaygroundMessagePublic]


class PlaygroundConversation(SQLModel, PrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, table=True):
    """One saved single-panel transcript, owned by the identity that saved it."""

    __tablename__ = "playground_conversation"
    __table_args__ = (
        # The list query's whole predicate and its sort, in one index: a
        # person's conversations in one workspace, newest first. Without the
        # trailing sort column the page's default ordering is a filesort over
        # every row the predicate matched.
        Index(
            "ix_playground_conversation_owner_recent",
            "user_id",
            "workspace_id",
            "created_at",
        ),
    )

    user_id: uuid.UUID = Field(foreign_key="user.id", ondelete="CASCADE", index=True)
    workspace_id: uuid.UUID = Field(foreign_key="workspace.id", ondelete="CASCADE", index=True)
    title: str = Field(max_length=MAX_CONVERSATION_TITLE_LENGTH)
    # The selector the transcript ran on, kept for display only. Not a foreign
    # key to anything: a model can leave the catalog, and a saved conversation
    # naming a model this deployment no longer routes to is still worth reading.
    model: str = Field(max_length=MAX_MODEL_KEY_LENGTH)


class PlaygroundMessage(SQLModel, PrimaryKeyMixin, CreatedAtMixin, table=True):
    """One turn of a saved transcript."""

    __tablename__ = "playground_message"
    __table_args__ = (
        # Ordering within a transcript is data, not an accident of insertion
        # order, and two turns claiming one position is a transcript that reads
        # differently on every load. The unique index is also the read path:
        # every message query is "this conversation, in order".
        UniqueConstraint("conversation_id", "position", name="uq_playground_message_conversation_position"),
    )

    conversation_id: uuid.UUID = Field(foreign_key="playground_conversation.id", ondelete="CASCADE", index=True)
    position: int = Field(nullable=False)
    role: str = Field(max_length=16)
    content: str = Field(sa_column=Column(Text, nullable=False))
    reasoning: str | None = Field(default=None, sa_column=Column(Text, nullable=True))


# ==============================================================================
# Saved comparisons
# ==============================================================================


class PlaygroundComparisonCreate(SQLModel):
    """One rated A/B exchange.

    Both answers in full, which is the disclosure the comparison consent flag
    covers: a preference with no answers attached is a datum nobody can later
    check, and the page's own history list shows the question and the two model
    ids from these columns.
    """

    workspace_id: uuid.UUID
    user_question: str = Field(min_length=1, max_length=MAX_COMPARISON_TEXT_LENGTH)
    model_a: str = Field(max_length=MAX_MODEL_KEY_LENGTH)
    model_b: str = Field(max_length=MAX_MODEL_KEY_LENGTH)
    model_a_answer: str = Field(max_length=MAX_COMPARISON_TEXT_LENGTH)
    model_b_answer: str = Field(max_length=MAX_COMPARISON_TEXT_LENGTH)
    preference: ComparisonPreference


class PlaygroundComparisonSummary(SQLModel):
    """A row in the comparison history: the question, the pair, the verdict.

    Deliberately without the two answers. The list shows a dozen rows at once
    and none of them renders an answer body, so sending them would move
    megabytes to draw a few lines of text. There is no detail endpoint either,
    because the page has no screen that reads one back: a comparison is a
    judgment that was recorded, not a transcript to resume.
    """

    id: uuid.UUID
    workspace_id: uuid.UUID
    user_question: str
    model_a: str
    model_b: str
    preference: str
    created_at: datetime


class PlaygroundComparisonsPublic(SQLModel):
    data: list[PlaygroundComparisonSummary]


class PlaygroundComparison(SQLModel, PrimaryKeyMixin, CreatedAtMixin, table=True):
    """One recorded model preference, owned by the identity that recorded it."""

    __tablename__ = "playground_comparison"
    __table_args__ = (
        Index(
            "ix_playground_comparison_owner_recent",
            "user_id",
            "workspace_id",
            "created_at",
        ),
    )

    user_id: uuid.UUID = Field(foreign_key="user.id", ondelete="CASCADE", index=True)
    workspace_id: uuid.UUID = Field(foreign_key="workspace.id", ondelete="CASCADE", index=True)
    user_question: str = Field(sa_column=Column(Text, nullable=False))
    model_a: str = Field(max_length=MAX_MODEL_KEY_LENGTH)
    model_b: str = Field(max_length=MAX_MODEL_KEY_LENGTH)
    model_a_answer: str = Field(sa_column=Column(Text, nullable=False))
    model_b_answer: str = Field(sa_column=Column(Text, nullable=False))
    preference: str = Field(max_length=16)

    def to_summary(self) -> PlaygroundComparisonSummary:
        """The list shape: everything but the two answer bodies."""
        return PlaygroundComparisonSummary(
            id=self.id,
            workspace_id=self.workspace_id,
            user_question=self.user_question,
            model_a=self.model_a,
            model_b=self.model_b,
            preference=self.preference,
            created_at=self.created_at,
        )


# ==============================================================================
# Pinned models
# ==============================================================================


class PlaygroundFavoriteModelsUpdate(SQLModel):
    """The whole pin list, replacing whatever was stored.

    A replace rather than a toggle endpoint, because the client already holds
    the list it is rendering and the order is part of it (a newly pinned model
    leads). Two tabs racing therefore resolve to one of the two lists rather
    than to an interleaving neither of them showed.
    """

    # Bounded on the list and on each item, which are different ceilings: the
    # first caps how many pins one person keeps, the second is the column's own
    # width. Refused rather than truncated, because a truncated key is a pin
    # that can never match a model again and nothing says why.
    model_keys: list[Annotated[str, Field(max_length=MAX_MODEL_KEY_LENGTH)]] = Field(
        max_length=MAX_FAVORITE_MODELS
    )


class PlaygroundFavoriteModelsPublic(SQLModel):
    """The pin list, most recently pinned first."""

    model_keys: list[str]


class PlaygroundFavoriteModel(SQLModel, PrimaryKeyMixin, CreatedAtMixin, table=True):
    """One model key an identity has pinned in one workspace."""

    __tablename__ = "playground_favorite_model"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "workspace_id",
            "model_key",
            name="uq_playground_favorite_model_owner_key",
        ),
        # The read is "this person's pins in this workspace, newest first", and
        # ``position`` is what carries the order the client sent rather than the
        # insertion clock, so a replace that reorders without adding a row still
        # reads back the way it was written.
        Index(
            "ix_playground_favorite_model_owner_position",
            "user_id",
            "workspace_id",
            "position",
        ),
    )

    user_id: uuid.UUID = Field(foreign_key="user.id", ondelete="CASCADE", index=True)
    workspace_id: uuid.UUID = Field(foreign_key="workspace.id", ondelete="CASCADE", index=True)
    model_key: str = Field(max_length=MAX_MODEL_KEY_LENGTH)
    position: int = Field(nullable=False)
