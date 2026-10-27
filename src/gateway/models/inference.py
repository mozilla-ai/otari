"""ORM tables for asynchronous batch jobs, idempotent completion requests and the stored country file."""

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Integer, LargeBinary, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime


class BatchRecord(Base):
    """Ownership and accounting record for an asynchronous batch job.

    Written at creation time so results accounting can be made idempotent (bill
    and log once, on the first completed retrieval), the batch cost can be folded
    into ``users.spend``, and ownership can be enforced without depending on the
    provider round-tripping the ``otari_user_id`` metadata marker. Batches created
    before this table existed carry no record and fall back to the
    metadata-anchored ownership path in ``api/routes/batches.py``. ``workspace_id``
    additionally anchors which workspace's organization-scoped provider key
    (otari#643) lifecycle calls should resolve credentials from.
    """

    __tablename__ = "batches"

    # Provider-assigned batch id (globally unique per provider), used as the
    # lookup key on retrieve/cancel/results.
    id: Mapped[str] = mapped_column(primary_key=True)
    # Instance/provider name the batch was created against (echoed to clients).
    provider: Mapped[str] = mapped_column()
    # Billed owner, stamped from the authenticated principal at creation. Non-null:
    # this record is the strict ownership anchor, so it must always name an owner.
    # CASCADE: deleting the user drops the ownership record (the user's keys are
    # gone too, and usage_logs remain the billing history).
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    # SET NULL: a key may be revoked while its batch is still in flight.
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"), index=True)
    # The workspace this batch was CREATED in (otari#643 follow-up), so
    # lifecycle calls (retrieve/cancel/results) can resolve organization-scoped
    # credentials from the batch's own origin rather than the retriever's
    # current workspace: a master-key or legitimately cross-workspace retrieval
    # would otherwise use the wrong organization's key, or find none, exactly
    # the failure `api_key_id` going NULL on key revocation already risks for
    # ownership. Nullable and SET NULL, not RESTRICT: batches created before
    # this column existed carry NULL here and fall back to the caller's own
    # workspace in `api/routes/batches.py`, and a workspace deleted out from
    # under an in-flight batch must not block that delete.
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="SET NULL"), index=True
    )
    model: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    # NULL until the first completed results retrieval accounts the batch; the
    # atomic NULL -> now transition is the idempotency gate for billing/logging.
    results_accounted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


IDEMPOTENCY_TABLE = "idempotency_keys"


class IdempotencyState(StrEnum):
    """Where the request holding an idempotency key has got to."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class IdempotencyRecord(Base):
    """One ``Idempotency-Key`` a caller sent, and the response it produced.

    A row is claimed ``in_progress`` before the provider is called and becomes
    ``completed`` with the response once the request succeeds. A request that
    fails deletes its claim, so a retry runs it again: a failed request is not
    billed, so running it again is not billing twice.
    """

    __tablename__ = IDEMPOTENCY_TABLE

    # Who may reuse the key: ``key:<api key id>``, or ``master:<user id>`` for the
    # master key. Two callers can never see each other's responses.
    scope: Mapped[str] = mapped_column(String(255), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    # SHA-256 of the endpoint and the canonical request body, so a key reused
    # for a different request is refused rather than answered with another result.
    request_hash: Mapped[str] = mapped_column(String(64))
    # Changes on every claim, so a request whose claim was taken over cannot
    # complete or delete the claim that replaced it.
    claim_token: Mapped[str] = mapped_column(String(36))
    state: Mapped[str] = mapped_column(String(16))
    # CASCADE: the scope dies with the user or key it names.
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id", ondelete="CASCADE"), index=True)
    status_code: Mapped[int | None] = mapped_column(Integer)
    response_body: Mapped[str | None] = mapped_column(Text)
    response_headers: Mapped[dict[str, str] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    # Past this, an ``in_progress`` claim is presumed abandoned and a retry may take it over.
    locked_until: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


COUNTRY_DATABASE_TABLE = "country_database"


class StoredCountryDatabase(Base):
    """The MaxMind DB country file an operator uploaded, which every replica reads.

    One row at most, held to ``id = 1``. A table of its own keeps the file's
    bytes out of every other query.
    """

    __tablename__ = COUNTRY_DATABASE_TABLE
    __table_args__ = (CheckConstraint("id = 1", name="ck_country_database_one_row"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False, default=1)
    content: Mapped[bytes] = mapped_column(LargeBinary)
    # The file's own metadata: GeoLite2-Country, DBIP-Country-Lite and so on.
    database_type: Mapped[str] = mapped_column(String(255))
    built_at: Mapped[datetime] = mapped_column(UtcDateTime())
    size_bytes: Mapped[int] = mapped_column(Integer)
    # What a replica compares with the file it has open, so it reads the bytes only when they change.
    sha256: Mapped[str] = mapped_column(String(64))
    uploaded_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
