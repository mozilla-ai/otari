"""Stored guardrail definitions for the dashboard (``/api/v1/guardrail-credentials``).

The write target for the picker at ``GET /api/v1/tool-settings/guardrails/catalog``.
That endpoint lists the guardrails this gateway can run itself and, for each
one, the constructor and per-call arguments it takes; these endpoints store one
of those choices together with the values filled in beside it, so a guardrail is
defined in Otari rather than in a sidecar's YAML.

Startup builds every definition it finds. A write here hands the row it touched
to the same loader, in the background, because the answer to a save is the row
and not a vendor round trip: a client that will not construct must not turn a
committed write into a failed response. Delete forgets the profile and builds
nothing. Re-encryption builds nothing either, and that is not an omission: it
rotates ciphertext and changes no argument, so what is already built is still
correct.

Each worker holds its own, so a write takes effect on the worker that served it
and on the others when they next restart. That is the cross-worker gap the
provider overlay has, and a definition is deployment configuration rather than
per-request policy.

Deliberately the same shape as ``/api/v1/search-tools`` and
``/api/v1/provider-credentials``: rows keyed by name, the credentials encrypted
at rest and never returned, a tri-state PATCH, an optimistic-concurrency check
under a row lock, and a re-encryption endpoint for ``OTARI_SECRET_KEY``
rotation. Operator-gated and never mounted in hybrid, as those two are.

A response carries the *names* of the stored secrets and never a value, which is
where this parts company with its siblings' ``last4``: a guardrail may hold
several credentials, so which ones are set is the useful answer and the last
four characters of a map are not one.
"""

import asyncio
import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, require_deployment_operator
from gateway.exceptions.guardrail_credentials import (
    GuardrailCredentialError,
    GuardrailCredentialExistsError,
    GuardrailCredentialNotFoundError,
)
from gateway.log_config import logger
from gateway.models.guardrails import GuardrailConfig, GuardrailCredential
from gateway.services.guardrail_credential_service import (
    MAX_ENFORCED_GUARDRAILS,
    UNSET,
    create_guardrail_credential,
    definition_from_row,
    delete_guardrail_credential,
    get_guardrail_credential,
    get_guardrail_credential_for_update,
    list_guardrail_credentials,
    reencrypt_guardrail_credentials,
    stored_secret_names,
    update_guardrail_credential,
    workspace_ids_by_credential,
)
from gateway.services.guardrail_loader import apply_stored_guardrail
from gateway.services.guardrail_runner import get_guardrail_runner
from gateway.services.guardrails import GuardrailsNotReachableError
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError

router = APIRouter(
    prefix="/guardrail-credentials",
    tags=["guardrail-credentials"],
    dependencies=[Depends(require_deployment_operator)],
)

# A sample, not a load test. The request path's own limits are the ones that bound
# real traffic; this only keeps a check from being handed a novel.
_MAX_TEST_INPUT = 8000


class StoredGuardrailSchema(BaseModel):
    """A stored guardrail definition. Credentials are never returned, only their names."""

    name: str
    guardrail_name: str
    create_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="The non-secret constructor arguments, as stored."
    )
    create_secrets: dict[str, str] = Field(
        default_factory=dict,
        description="The stored credentials by name, each masked. Empty when the stored map cannot be read.",
    )
    validate_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="The per-call arguments, with credential-shaped entries masked."
    )
    enabled: bool
    mode: Literal["block", "monitor"] = Field(
        description="What happens when this guardrail flags a request: block refuses it, monitor serves it."
    )
    on_unavailable: Literal["block", "allow"] = Field(
        description="What Otari does when the guardrail returns no verdict at all."
    )
    applies_to_all_workspaces: bool
    workspace_ids: list[uuid.UUID] = Field(
        default_factory=list,
        description="The workspaces this definition checks. Empty when it applies to all of them.",
    )
    created_at: str | None = None
    updated_at: str | None = None
    decryptable: bool = Field(
        default=True,
        description=(
            "False when the stored credentials cannot be read with the current OTARI_SECRET_KEY. "
            "The definition is intact; re-enter its credentials or restore the key that wrote them."
        ),
    )
    loaded: bool = Field(
        default=False,
        description=(
            "Whether this worker has the guardrail built and ready. False on a definition that failed "
            "to build, whose checks therefore do not run. Answered by the worker that served the read."
        ),
    )

    @classmethod
    def from_model(
        cls, row: GuardrailCredential, *, workspace_ids: list[uuid.UUID] | None = None
    ) -> "StoredGuardrailSchema":
        names, decryptable = stored_secret_names(row)
        return cls(
            **row.to_public_dict(secret_names=names),
            workspace_ids=[] if row.applies_to_all_workspaces else (workspace_ids or []),
            decryptable=decryptable,
            loaded=get_guardrail_runner().knows(row.name),
        )


class CreateGuardrailCredentialRequest(BaseModel):
    """Store a guardrail definition. Secret arguments are encrypted and never returned."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "prompt-injection",
                "guardrail_name": "lakera_guard",
                "create_kwargs": {"api_key": "lak-...", "endpoint": "https://api.lakera.ai/v2/guard"},
            }
        }
    )

    name: str = Field(
        min_length=1,
        max_length=128,
        # One path segment: the row is addressed as /{name}, and neither a bare
        # slash nor an encoded one reaches that route, so a name carrying either
        # would store a row no read, update or delete could ever name again.
        pattern=r"^[^/]+$",
        description="The profile name a caller sends. One path segment, so it cannot contain '/'.",
    )
    guardrail_name: str = Field(
        description="The guardrail to build, as listed by GET /tool-settings/guardrails/catalog."
    )
    create_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Constructor arguments, secret and plain together. They are split by the catalog's own "
            "secret flag; the secret half is encrypted before it is stored."
        ),
    )
    validate_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Per-call arguments sent with the text on every check."
    )
    enabled: bool = Field(
        default=True,
        description=(
            "A disabled definition is kept but checks nothing. At most "
            f"{MAX_ENFORCED_GUARDRAILS} may be enabled at once."
        ),
    )
    mode: Literal["block", "monitor"] = Field(
        default="block",
        description=(
            "What happens when this guardrail flags a request. 'block' refuses it with a 403 and "
            "never calls the provider; 'monitor' serves it and reports the verdict on the response."
        ),
    )
    on_unavailable: Literal["block", "allow"] = Field(
        default="block",
        description=(
            "What Otari does when the guardrail returns no verdict at all, because the vendor failed, "
            "timed out or answered malformed. 'block' refuses the request, 'allow' serves it. Consulted "
            "only when mode is 'block': a monitoring definition serves the request either way and "
            "reports the missing verdict. Not the same as an inconclusive verdict, which never blocks."
        ),
    )
    applies_to_all_workspaces: bool = Field(
        default=False,
        description=(
            "True checks every workspace, including one created later; false checks only the "
            "workspaces named by workspace_ids."
        ),
    )
    workspace_ids: list[uuid.UUID] = Field(
        default_factory=list,
        description="Workspaces this guardrail checks. Must be empty when applies_to_all_workspaces is true.",
    )

    @model_validator(mode="after")
    def _reject_redundant_scope(self) -> "CreateGuardrailCredentialRequest":
        """Refuse a workspace list alongside ``applies_to_all_workspaces``.

        The two say different things about the same definition and the flag wins
        at resolve time, so accepting both would store a list that never decides
        anything while reading as though it does.
        """
        if self.applies_to_all_workspaces and self.workspace_ids:
            raise ValueError("workspace_ids must be empty when applies_to_all_workspaces is true")
        return self


class UpdateGuardrailCredentialRequest(BaseModel):
    """Update a stored definition. Omitted fields keep their stored value."""

    model_config = ConfigDict(
        json_schema_extra={
            # A partial update, since every field here is optional and a derived
            # example would send placeholders the endpoint refuses. The '***'
            # stands for the stored credential the editor was never shown.
            "example": {
                "create_kwargs": {"api_key": "***", "endpoint": "https://api.lakera.ai/v2/guard"},
                "enabled": False,
            }
        }
    )

    guardrail_name: str | None = None
    create_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Replaces the whole map when sent. A value of '***' keeps the stored credential of that "
            "name, a new value rotates it, and a credential left out is cleared."
        ),
    )
    validate_kwargs: dict[str, Any] | None = None
    enabled: bool | None = None
    mode: Literal["block", "monitor"] | None = None
    on_unavailable: Literal["block", "allow"] | None = None
    applies_to_all_workspaces: bool | None = None
    workspace_ids: list[uuid.UUID] | None = Field(
        default=None, description="Replaces the scope whole when sent; [] clears it."
    )
    expected_updated_at: str | None = Field(
        default=None,
        description="Optimistic concurrency: if set, the update 412s unless it matches the stored updated_at.",
    )


class TestGuardrailRequest(BaseModel):
    """Text to run one stored guardrail against."""

    input_text: str = Field(min_length=1, max_length=_MAX_TEST_INPUT)
    validate_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Merged over the stored per-call arguments, for this call only."
    )


class TestGuardrailResponse(BaseModel):
    """What one guardrail said about the text."""

    ok: bool = Field(description="Whether the guardrail ran at all. False means it could not be evaluated.")
    valid: bool | None = Field(
        default=None,
        description="True when the input passed, false when it was flagged, null when the verdict was inconclusive.",
    )
    explanation: str | None = None
    score: float | None = None
    error: str | None = Field(default=None, description="Why the guardrail could not run, when ok is false.")


class ReencryptGuardrailCredentialsResponse(BaseModel):
    """Result of re-encrypting stored guardrail credentials with the primary secret key."""

    reencrypted: int = Field(description="Number of stored credential maps re-encrypted.")
    unreadable: int = Field(
        description="Number of maps left untouched because they could not be decrypted."
    )


def _bad_request(exc: Exception) -> HTTPException:
    """Every validation refusal answers 400 with the rule that was broken.

    The messages come from the service and name a parameter, a guardrail or a
    requirement, never a submitted value: this router is operator-only, but the
    error-detail boundary holds here as it does everywhere.
    """
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


def _not_found(name: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=str(GuardrailCredentialNotFoundError(name))
    )


def _database_error() -> HTTPException:
    """A refused write is a 500 saying only that, as in the sibling stores.

    The service has already rolled back by the time this is raised, so nothing
    here touches the session.
    """
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error")


# Strong references to the rebuilds in flight, so one is not collected while it is
# still constructing. The pattern, and the reason for it, is ``_pipeline.py``'s
# ``_USAGE_REPORT_TASKS``.
_REBUILD_TASKS: set[asyncio.Task[None]] = set()


def _rebuild(row: GuardrailCredential) -> None:
    """Make the runner agree with the row just written, so the next request finds it ready.

    Without this a write would leave exactly the profile an operator just touched
    as the one nobody has built, while every other one was built at startup.

    In the background, because the answer to a save is the row: a vendor client
    that will not construct must not turn a committed write into a failed response.
    What a row means is the loader's to decide, so that a write and a restart
    cannot disagree about it.
    """
    task = asyncio.create_task(apply_stored_guardrail(row))
    _REBUILD_TASKS.add(task)

    def _finished(done: asyncio.Task[None]) -> None:
        _REBUILD_TASKS.discard(done)
        if done.cancelled():
            return
        if (error := done.exception()) is not None:
            logger.warning("Guardrail '%s' was written but did not build: %s", row.name, error)

    task.add_done_callback(_finished)


@router.get("")
async def list_stored_guardrails(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[StoredGuardrailSchema]:
    """List every stored guardrail definition.

    Credentials are never returned. Each row reports which credentials it holds
    by name and whether they can still be read with the current
    ``OTARI_SECRET_KEY``; a row that cannot be read is listed rather than
    hidden, because the operator is the person who can fix it.
    """
    scoped = await workspace_ids_by_credential(db)
    return [
        StoredGuardrailSchema.from_model(row, workspace_ids=scoped.get(row.name, []))
        for row in await list_guardrail_credentials(db)
    ]


@router.post("/reencrypt")
async def reencrypt_stored_guardrail_credentials(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ReencryptGuardrailCredentialsResponse:
    """Re-encrypt stored guardrail credentials with the primary OTARI_SECRET_KEY.

    The guardrail half of the key-rotation procedure; run it alongside
    ``POST /api/v1/provider-credentials/reencrypt`` and
    ``POST /api/v1/search-tools/reencrypt``. Maps that cannot be decrypted are
    left untouched and must be recovered by re-entering that guardrail's
    credentials.
    """
    try:
        reencrypted, unreadable = await reencrypt_guardrail_credentials(db)
    except SecretBoxUnavailableError as exc:
        await db.rollback()
        raise _bad_request(exc) from None
    except SQLAlchemyError:
        raise _database_error() from None
    return ReencryptGuardrailCredentialsResponse(reencrypted=reencrypted, unreadable=unreadable)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_stored_guardrail(
    request: CreateGuardrailCredentialRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StoredGuardrailSchema:
    """Store a guardrail definition.

    The definition is held to what the catalog says its guardrail accepts: an
    unknown guardrail, an argument it does not take, a live object that cannot
    be written down, and a required argument nothing else supplies are each a
    400 naming the rule. Storing a credential requires ``OTARI_SECRET_KEY``.
    """
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A guardrail name cannot be blank.")
    if await get_guardrail_credential(db, name) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(GuardrailCredentialExistsError(name)))

    try:
        row = await create_guardrail_credential(
            db,
            name=name,
            guardrail_name=request.guardrail_name,
            create_kwargs=request.create_kwargs,
            validate_kwargs=request.validate_kwargs,
            enabled=request.enabled,
            mode=request.mode,
            on_unavailable=request.on_unavailable,
            applies_to_all_workspaces=request.applies_to_all_workspaces,
            workspace_ids=request.workspace_ids,
        )
    except GuardrailCredentialExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from None
    except (GuardrailCredentialError, SecretBoxUnavailableError) as exc:
        await db.rollback()
        raise _bad_request(exc) from None
    except SQLAlchemyError:
        raise _database_error() from None

    _rebuild(row)
    scoped = await workspace_ids_by_credential(db, name=row.name)
    return StoredGuardrailSchema.from_model(row, workspace_ids=scoped.get(row.name, []))


@router.post("/{name}/test")
async def test_stored_guardrail(
    name: str,
    request: TestGuardrailRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TestGuardrailResponse:
    """Run a stored guardrail against some text, so an operator sees it work.

    Builds the definition as it stands right now and checks the text against that,
    changing nothing about what the gateway is enforcing. A disabled definition is
    as testable as any other, since checking one before turning it on is the point,
    and finding out must not be what puts it in front of traffic.

    A guardrail that cannot run answers ``ok: false`` with the reason rather than
    an error status: the question asked was whether this definition works, and one
    shape of answer is easier to act on than two.
    """
    row = await get_guardrail_credential(db, name)
    if row is None:
        raise _not_found(name)

    try:
        definition = definition_from_row(row)
    except (SecretBoxUnavailableError, SecretDecryptionError):
        return TestGuardrailResponse(ok=False, error=f"The stored credentials of '{name}' cannot be decrypted.")

    cfg = GuardrailConfig(profile=name, mode="monitor", validate_kwargs=request.validate_kwargs)
    try:
        result = await get_guardrail_runner().probe(
            definition=definition, cfg=cfg, input_text=request.input_text
        )
    except GuardrailsNotReachableError as exc:
        # The runner's own message, which names types and argument names and never
        # an argument's value. This route is operator-gated.
        logger.info("Test of stored guardrail '%s' could not be evaluated", name)
        return TestGuardrailResponse(ok=False, error=str(exc))

    return TestGuardrailResponse(
        ok=True,
        valid=result.valid,
        explanation=str(result.explanation) if result.explanation is not None else None,
        score=float(result.score) if isinstance(result.score, int | float) else None,
    )


@router.get("/{name}")
async def get_stored_guardrail(
    name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StoredGuardrailSchema:
    """Read one stored guardrail definition. Credentials are returned by name only."""
    row = await get_guardrail_credential(db, name)
    if row is None:
        raise _not_found(name)
    scoped = await workspace_ids_by_credential(db, name=name)
    return StoredGuardrailSchema.from_model(row, workspace_ids=scoped.get(name, []))


@router.patch("/{name}")
async def update_stored_guardrail(
    name: str,
    request: UpdateGuardrailCredentialRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StoredGuardrailSchema:
    """Update a stored guardrail definition. Omitted fields are left as they are.

    The row is locked ``FOR UPDATE`` so the ``expected_updated_at`` check and
    the write it guards are atomic. The definition as it will be *after* the
    update is validated, so a change that would leave it unbuildable (clearing a
    required argument, or moving to a guardrail that does not take an argument
    the row carries) is refused rather than stored.
    """
    row = await get_guardrail_credential_for_update(db, name)
    if row is None:
        raise _not_found(name)
    if request.expected_updated_at is not None:
        current = row.updated_at.isoformat() if row.updated_at else None
        if current != request.expected_updated_at:
            raise HTTPException(
                status_code=status.HTTP_412_PRECONDITION_FAILED,
                detail="This guardrail was modified since you loaded it; reload and retry.",
            )

    sent = request.model_fields_set

    def supplied(field: str) -> Any:
        """The value the caller sent, or UNSET when they sent nothing for this field.

        An omitted field and an explicit null both keep the stored value. None of
        these is nullable, so there is no third state for a null to mean. An empty
        ``workspace_ids`` is a value rather than an omission, and clears the scope.
        """
        value = getattr(request, field)
        return value if field in sent and value is not None else UNSET

    try:
        updated = await update_guardrail_credential(
            db,
            row=row,
            guardrail_name=supplied("guardrail_name"),
            create_kwargs=supplied("create_kwargs"),
            validate_kwargs=supplied("validate_kwargs"),
            enabled=supplied("enabled"),
            mode=supplied("mode"),
            on_unavailable=supplied("on_unavailable"),
            applies_to_all_workspaces=supplied("applies_to_all_workspaces"),
            workspace_ids=supplied("workspace_ids"),
        )
    except (GuardrailCredentialError, SecretBoxUnavailableError, SecretDecryptionError) as exc:
        await db.rollback()
        raise _bad_request(exc) from None
    except SQLAlchemyError:
        raise _database_error() from None

    _rebuild(updated)
    scoped = await workspace_ids_by_credential(db, name=updated.name)
    return StoredGuardrailSchema.from_model(updated, workspace_ids=scoped.get(updated.name, []))


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_stored_guardrail(
    name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a stored guardrail definition, credentials and all."""
    try:
        deleted = await delete_guardrail_credential(db, name)
    except SQLAlchemyError:
        raise _database_error() from None
    if not deleted:
        raise _not_found(name)
    get_guardrail_runner().drop(name)
