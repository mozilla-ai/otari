"""An organization's own definitions of the guardrails this gateway runs itself.

The mandate next door (`organization_guardrail_service`) says *where* a check
runs and how hard it bites. A definition says *what* the check is: which of the
guardrails `services/guardrail_catalog` lists to construct, and the arguments to
construct it with. So an organization can define a check rather than only name a
profile some service already serves, and it does that the way it adds a provider
key: pick from a catalog, fill typed fields, paste the vendor credential.

**Nothing runs these rows yet.** Building the guardrail and putting it on the
request path are later steps. What lands here is the store and its rules.

**Every rule but one is read off the catalog.** A list of guardrails, arguments
or credentials written in this module could only ever drift from the picker it
exists to accept, and the drift would surface as a row the form offered and
nothing can build. So `builtin_guardrail_spec` is the single derivation, and the
one hand-written rule (`_refuse_bedrock_without_keys`) says at its own site why
the catalog cannot give it.

**Secrets are split by the catalog's ``secret`` flag**, not by the shape of a
name: the plain values go to ``create_kwargs`` and every secret into one
encrypted map. The guardrails disagree about how many credentials they need
(Bedrock three, watsonx two, most one, some none), so a column each would chase
every guardrail upstream adds. ``create_kwargs`` comes back in clear on purpose,
the secrets having already been taken out of it by flag, and the form has to
round-trip an endpoint and a project id.

**This service holds no session.** Its queries are the repository's, its commits
are the Unit of Work's blocks, and the one thing it reaches for by session is
the role gate, which it receives already built. `OrganizationService` is still
in the old shape, and that seam is where it shows.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any

from any_guardrail.base import GuardrailName
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic.json_schema import SkipJsonSchema

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.guardrails import OrganizationGuardrailDefinition
from gateway.models.secret_fields import REDACTED_VALUE, restore_redacted_values
from gateway.models.tenancy import User
from gateway.repositories.tenancy import OrganizationGuardrailDefinitionRepository
from gateway.services.guardrail_catalog import (
    BuiltInGuardrailSpec,
    builtin_guardrail_spec,
    definable_by_an_organization,
)
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.tenancy.errors import (
    OrganizationGuardrailDefinitionAlreadyExistsError,
    OrganizationGuardrailDefinitionArgumentsError,
    OrganizationGuardrailDefinitionLimitReachedError,
    OrganizationGuardrailDefinitionNotFoundError,
    OrganizationGuardrailNotBuildableError,
    OrganizationGuardrailNotDefinableError,
    SecretBoxUnavailableTenancyError,
)
from gateway.services.tenancy.organization_service import OrganizationService

# What one organization may define. A different bound from
# `MAX_GUARDRAILS_PER_ORGANIZATION` next door and bounding something else: that
# one counts sequential calls a request waits on, while each enabled definition
# here becomes a built vendor client held in memory by every worker. An
# organization can mandate the same definition under several profiles, so
# neither number constrains the other.
#
# Ten to start, and check-then-act like its neighbor. #1411 covers both.
MAX_DEFINITIONS_PER_ORGANIZATION = 10

_MAX_LIST_LIMIT = 1000

# The one rule below that the catalog cannot give us. Bedrock marks both AWS
# keys optional because boto3 has somewhere else to look: with neither supplied
# it falls back to the host's instance role, which belongs to the operator, so an
# organization's definition would quietly run on the deployment's own AWS
# identity. Upstream cannot mark them required without refusing every caller who
# means that fallback, and the catalog states what upstream declares, so the rule
# lives at the store that has an opinion about whose credentials are being spent.
_BEDROCK_KEY_PAIR = ("aws_access_key_id", "aws_secret_access_key")


class OrganizationGuardrailDefinitionCreate(BaseModel):
    """Request body for defining a guardrail Otari will build and call itself.

    ``create_kwargs`` carries both halves of the form: the plain arguments and
    the vendor credentials. Which is which is the catalog's answer, not this
    schema's, so a credential lands in the encrypted map whatever it is called
    and a caller cannot move one into the plain column by naming it oddly.

    Per-call arguments are not offered. This row says what a guardrail is built
    with; what travels with each check is the mandate's ``validate_kwargs``.
    """

    # A body the server would actually accept, which the derived one is not: the
    # generator fills every string with "string", and ``guardrail_name`` has to
    # name a guardrail the catalog lists. A sample that fails validation teaches
    # the wrong shape, and it reaches a reader through the OpenAPI spec and the
    # Postman collection (`OrganizationGuardrailCreate` carries one for the same
    # reason).
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "prod-lakera",
                "guardrail_name": "lakera_guard",
                "create_kwargs": {"api_key": "lakera-...", "endpoint": "https://api.lakera.ai"},
                "enabled": True,
            }
        }
    )

    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)] = Field(
        description="The organization's own label for this definition, unique within the organization",
    )
    guardrail_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)] = Field(
        description="The any-guardrail class to build, as the built-in guardrail catalog names it",
    )
    create_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Constructor arguments for the guardrail. Every argument the catalog marks secret is "
            "encrypted at rest and never returned; the rest are stored and returned as sent"
        ),
    )
    enabled: bool = Field(default=True, description="False stops the guardrail everywhere it is mandated")


class OrganizationGuardrailDefinitionUpdate(BaseModel):
    """Partial update. Only the fields the caller sets are applied.

    ``create_kwargs`` replaces the arguments whole when sent, and an argument the
    catalog marks secret keeps its stored value where the caller echoes back the
    ``***`` a read gave them. Omitting it leaves both columns untouched *and
    reads neither*, which is what lets an admin on a deployment whose
    ``OTARI_SECRET_KEY`` has moved still flip ``enabled`` and repair the row by
    typing the credential again.

    Changing ``guardrail_name`` without sending ``create_kwargs`` re-splits the
    stored arguments under the new class, because the plain/secret split is the
    old class's answer and would otherwise go stale.
    """

    # Same reason as the create body: the derived `"guardrail_name": "string"`
    # sample would be refused.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "create_kwargs": {"api_key": "***", "endpoint": "https://eu.api.lakera.ai"},
                "enabled": False,
            }
        }
    )

    # ``SkipJsonSchema[None]`` on the fields backing NOT NULL columns, as
    # `OrganizationGuardrailUpdate` explains: ``None`` is this schema's "not
    # sent" marker and the validator below refuses it as a value, so the
    # published schema must not advertise ``null`` as accepted.
    name: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)] | SkipJsonSchema[None]
    ) = None
    guardrail_name: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)] | SkipJsonSchema[None]
    ) = None
    create_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Replaces the constructor arguments whole. An argument sent as *** keeps the value "
            "stored under that name; a secret left out is cleared. Omit the field to leave the "
            "stored arguments alone, and {} to clear them"
        ),
    )
    enabled: bool | SkipJsonSchema[None] = None

    @model_validator(mode="after")
    def _reject_explicit_nulls(self) -> OrganizationGuardrailDefinitionUpdate:
        """Refuse an explicit ``null`` for a column that cannot hold one.

        Caught here for the reason `OrganizationGuardrailUpdate` catches it: a
        ``NOT NULL`` violation arrives as the same ``IntegrityError`` the unique
        index raises, so the service would report a name collision that never
        happened.
        """
        nulled = [
            field
            for field in ("name", "guardrail_name", "enabled")
            if field in self.model_fields_set and getattr(self, field) is None
        ]
        if nulled:
            raise ValueError(f"{', '.join(nulled)} cannot be null; omit the field to leave it unchanged")
        return self


class OrganizationGuardrailDefinitionPublic(BaseModel):
    """The API-facing shape. Carries the names of the stored secrets and none of their values."""

    id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    guardrail_name: str
    create_kwargs: dict[str, Any] = Field(
        description=(
            "The constructor arguments the catalog does not mark secret, as they were stored. "
            "Returned in clear: the secrets were taken out of this map by flag, and a form has to "
            "round-trip an endpoint or a project id"
        ),
    )
    create_secrets: dict[str, str] = Field(
        description=(
            "The secret constructor arguments this definition holds, each as ***. Sending one back "
            "unchanged keeps the stored value; sending a new one rotates it, and leaving one out "
            "clears it"
        ),
    )
    secrets_decryptable: bool = Field(
        description=(
            "False when the stored secrets cannot be read with the current OTARI_SECRET_KEY, in "
            "which case create_secrets is empty and the definition needs its credentials sent again"
        ),
    )
    enabled: bool
    created_at: str
    updated_at: str

    @classmethod
    def from_model(
        cls,
        definition: OrganizationGuardrailDefinition,
        *,
        secret_names: list[str],
        secrets_decryptable: bool,
    ) -> OrganizationGuardrailDefinitionPublic:
        return cls(
            id=definition.id,
            organization_id=definition.organization_id,
            name=definition.name,
            guardrail_name=definition.guardrail_name,
            create_kwargs=dict(definition.create_kwargs),
            create_secrets=dict.fromkeys(secret_names, REDACTED_VALUE),
            secrets_decryptable=secrets_decryptable,
            enabled=definition.enabled,
            created_at=definition.created_at.isoformat(),
            updated_at=definition.updated_at.isoformat(),
        )


class OrganizationGuardrailDefinitionsPublic(BaseModel):
    data: list[OrganizationGuardrailDefinitionPublic]
    count: int


def _names(values: list[str]) -> str:
    """Render parameter names for a refusal message, in a stable order."""
    return ", ".join(sorted(values))


def _definable_spec(guardrail_name: str) -> BuiltInGuardrailSpec:
    """The catalog row for a guardrail this organization may define, or a refusal.

    Three questions in order, because each has a different answer for the caller:
    what this gateway can construct, whose money a construction spends, and
    whether a stored row can say enough to construct it at all.
    """
    spec = builtin_guardrail_spec(guardrail_name)
    if spec is None:
        raise OrganizationGuardrailNotBuildableError(guardrail_name)
    if not definable_by_an_organization(guardrail_name):
        raise OrganizationGuardrailNotDefinableError(guardrail_name)
    _refuse_unconfigurable(spec)
    return spec


def _refuse_unconfigurable(spec: BuiltInGuardrailSpec) -> None:
    """Refuse a guardrail that no stored row could describe, whatever it supplied.

    Both arms are empty against the installed any-guardrail, which is the point:
    each closes a shape a future release could arrive in rather than describing
    one that is here. The requirement-group rule below takes the same posture,
    every group upstream declares today carrying environment variables of its
    own.

    A required constructor argument that cannot be written down has no field to
    arrive in. A required *per-call* argument has no column at all: this row says
    what a guardrail is built with, and the mandate is what carries arguments to
    each check, so a guardrail that cannot run without one is not a guardrail a
    definition alone describes. ``any_llm.policy`` is exactly that shape and is
    already refused a step earlier, for spending the deployment's own money.
    """
    unwritable = [
        parameter.name
        for parameter in spec.create_parameters
        if parameter.required and not parameter.env_var and not parameter.storable
    ]
    if unwritable:
        raise OrganizationGuardrailDefinitionArgumentsError(
            f"'{spec.guardrail_name}' cannot be defined here: {_names(unwritable)} must be supplied and "
            "cannot be stored, being a live object rather than configuration"
        )
    per_call = [
        parameter.name for parameter in spec.validate_parameters if parameter.required and not parameter.env_var
    ]
    if per_call:
        raise OrganizationGuardrailDefinitionArgumentsError(
            f"'{spec.guardrail_name}' cannot be defined here: {_names(per_call)} must be sent with every "
            "check, and a definition stores only what a guardrail is built with"
        )


def _storable_alternatives(spec: BuiltInGuardrailSpec, parameter_name: str) -> list[str]:
    """What a caller can supply in place of an argument that cannot be stored.

    Read off the catalog two ways rather than written out per guardrail. Where the
    refused argument belongs to requirement groups, those groups already name its
    alternatives: watsonx offers ``api_client`` as the substitute for ``api_key``,
    ``url`` and the project or space, so the storable members of its groups are
    the path that works. Where it belongs to no group, as Bedrock's
    ``boto3_session`` does, the guardrail's own storable secrets are that path,
    the session existing to carry exactly those credentials.
    """
    storable = {parameter.name for parameter in spec.create_parameters if parameter.storable}
    grouped = {
        member
        for group in spec.requirement_groups
        if parameter_name in group.parameters
        for member in group.parameters
        if member in storable
    }
    if grouped:
        return sorted(grouped)
    return sorted(parameter.name for parameter in spec.create_parameters if parameter.secret and parameter.storable)


def _validate_arguments(spec: BuiltInGuardrailSpec, arguments: dict[str, Any]) -> None:
    """Refuse build arguments the guardrail cannot be constructed from.

    Run *before* the split, and that order is load-bearing: an argument no spec
    declares carries no ``secret`` flag, so validating afterwards would have
    already dropped a credential under an unexpected name into the plain column.
    """
    declared = {parameter.name: parameter for parameter in spec.create_parameters}

    # No guardrail takes extra keyword arguments, so an unknown name would fail
    # at build time anyway. Refusing it here is what keeps an unrecognized name
    # from being stored in the first place.
    unknown = [name for name in arguments if name not in declared]
    if unknown:
        raise OrganizationGuardrailDefinitionArgumentsError(
            f"'{spec.guardrail_name}' takes no build argument named {_names(unknown)}"
        )

    # Not covered by the check above, because a live-object parameter *is*
    # declared. Without this it reads as an ordinary secret, serializes into the
    # encrypted map, saves, and fails much later inside a vendor SDK.
    unstorable = sorted(name for name in arguments if not declared[name].storable)
    if unstorable:
        detail = "; ".join(
            f"{name} cannot be stored, being a live object rather than configuration: "
            f"supply {_names(_storable_alternatives(spec, name))} instead"
            for name in unstorable
        )
        raise OrganizationGuardrailDefinitionArgumentsError(f"'{spec.guardrail_name}': {detail}")

    # Only the ones no environment variable can supply. The carve-out is
    # load-bearing: `required` folds in upstream's effectively-required flag, so
    # `lakera_guard.api_key` reads required although `LAKERA_API_KEY` supplies
    # it, and demanding it would refuse a legitimate row.
    missing = [
        parameter.name
        for parameter in declared.values()
        if parameter.required and not parameter.env_var and parameter.storable and parameter.name not in arguments
    ]
    if missing:
        raise OrganizationGuardrailDefinitionArgumentsError(
            f"'{spec.guardrail_name}' needs {_names(missing)}, which no environment variable supplies"
        )

    for group in spec.requirement_groups:
        # A group whose members an environment variable can satisfy is left
        # alone, for the same reason a required env-backed parameter is: whether
        # the variable is set belongs to the process that builds the guardrail,
        # not to this one.
        if group.env_vars:
            continue
        if not set(group.parameters) & set(arguments):
            raise OrganizationGuardrailDefinitionArgumentsError(f"'{spec.guardrail_name}': {group.description}")

    _refuse_bedrock_without_keys(spec, arguments)


def _refuse_bedrock_without_keys(spec: BuiltInGuardrailSpec, arguments: dict[str, Any]) -> None:
    """Refuse a Bedrock definition that would run on the host's own AWS identity.

    See ``_BEDROCK_KEY_PAIR`` for why this one rule is written out. Both keys,
    not either: boto3 falls back to the instance role unless it has the pair.
    """
    if spec.guardrail_name != GuardrailName.BEDROCK_GUARDRAILS.value:
        return
    missing = [name for name in _BEDROCK_KEY_PAIR if not arguments.get(name)]
    if missing:
        raise OrganizationGuardrailDefinitionArgumentsError(
            f"'{spec.guardrail_name}' needs {_names(list(_BEDROCK_KEY_PAIR))}; without both it would "
            "use the credentials of the host this gateway runs on"
        )


def _split_by_secret_flag(
    spec: BuiltInGuardrailSpec, arguments: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate the arguments into the plain column and the encrypted map.

    By the catalog's ``secret`` flag rather than by the shape of a name, which is
    what `models/secret_fields` has to fall back on for a free-form settings
    dict: here the guardrail's own schema says which argument is a credential, so
    a vendor key cannot reach the plain column by being named unusually.
    """
    secrets = {parameter.name for parameter in spec.create_parameters if parameter.secret}
    plain = {name: value for name, value in arguments.items() if name not in secrets}
    hidden = {name: value for name, value in arguments.items() if name in secrets}
    return plain, hidden


def _encrypted_secrets(secrets: dict[str, Any]) -> str | None:
    """One ciphertext for the whole map, or None when there is nothing to keep.

    ``sort_keys`` so reordering a form does not produce different ciphertext for
    the same credentials.
    """
    if not secrets:
        return None
    try:
        return encrypt_secret(json.dumps(secrets, sort_keys=True))
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError("organization guardrail definitions") from None


def _stored_secrets(definition: OrganizationGuardrailDefinition) -> dict[str, Any]:
    """The definition's decrypted secret arguments.

    Raises:
        SecretBoxUnavailableError: no ``OTARI_SECRET_KEY`` is configured.
        SecretDecryptionError: the key configured cannot read this ciphertext.
        json.JSONDecodeError: the plaintext is not the map this service wrote.
    """
    if not definition.encrypted_create_secrets:
        return {}
    decoded = json.loads(decrypt_secret(definition.encrypted_create_secrets))
    return dict(decoded)


def _readable_secrets(definition: OrganizationGuardrailDefinition) -> dict[str, Any]:
    """The decrypted secrets, translating an unreadable map into an answer a caller can act on.

    A rotated or lost ``OTARI_SECRET_KEY`` is the caller's problem to fix here and
    not a 500: they fix it by sending the credentials again, which the message
    says. A *missing* key is the deployment's problem, so it keeps its own 500.
    """
    try:
        return _stored_secrets(definition)
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError("organization guardrail definitions") from None
    except (SecretDecryptionError, json.JSONDecodeError) as exc:
        raise OrganizationGuardrailDefinitionArgumentsError(
            f"The stored secrets of '{definition.name}' cannot be read with the current OTARI_SECRET_KEY, "
            "so this change needs create_kwargs sent with the credentials in it"
        ) from exc


def _arguments_after(
    definition: OrganizationGuardrailDefinition,
    request: OrganizationGuardrailDefinitionUpdate,
    guardrail_name: str,
) -> dict[str, Any] | None:
    """The build arguments this request leaves the definition with, or None to keep both columns.

    Four cases, and which of them decrypts is the whole point:

    - Nothing sent and the same guardrail: keep both columns and read neither. An
      admin whose ``OTARI_SECRET_KEY`` has moved can still rename the row or turn
      it off.
    - Nothing sent and a new guardrail: the stored split is the old class's
      answer, so the stored arguments are decrypted, merged and re-split.
      Changing the class is the one edit that cannot leave them alone.
    - Sent with no ``***`` in it: taken as sent, and still no decryption. This is
      how a caller repairs a row whose secrets cannot be read.
    - Sent with a ``***``: the stored value under that name is kept, a new value
      rotates it, and a secret left out is cleared.
    """
    if request.create_kwargs is None:
        if guardrail_name == definition.guardrail_name:
            return None
        return {**definition.create_kwargs, **_readable_secrets(definition)}

    if not any(value == REDACTED_VALUE for value in request.create_kwargs.values()):
        return dict(request.create_kwargs)

    stored = {**definition.create_kwargs, **_readable_secrets(definition)}
    return restore_redacted_values(request.create_kwargs, stored) or {}


def _public(definition: OrganizationGuardrailDefinition) -> OrganizationGuardrailDefinitionPublic:
    """The read shape, reporting an unreadable secret map rather than failing the listing.

    The `routes/providers.py::_is_decryptable` posture: one row nobody can
    decrypt must not cost an organization the page that would let them fix it.
    """
    try:
        secrets = _stored_secrets(definition)
    except (SecretBoxUnavailableError, SecretDecryptionError, json.JSONDecodeError):
        return OrganizationGuardrailDefinitionPublic.from_model(definition, secret_names=[], secrets_decryptable=False)
    return OrganizationGuardrailDefinitionPublic.from_model(
        definition, secret_names=sorted(secrets), secrets_decryptable=True
    )


class OrganizationGuardrailDefinitionService:
    """CRUD for the caller's organization's guardrail definitions.

    Every method runs its database work in one Unit of Work block, so a write
    that refuses halfway leaves nothing behind.
    """

    def __init__(
        self,
        *,
        definitions: OrganizationGuardrailDefinitionRepository,
        organizations: OrganizationService,
        uow: UnitOfWork,
    ) -> None:
        self._definitions = definitions
        self._organizations = organizations
        self._uow = uow

    async def _manageable_organization_id(self, user: User) -> uuid.UUID:
        """The caller's organization, having checked they may manage its guardrails.

        One gate for reads and writes alike, as the mandates next door have:
        these rows name the vendors an organization has accounts with and say
        which of those accounts have credentials stored here.
        """
        organization = await self._organizations.get_active_organization_for_user(user)
        await self._organizations.require_active_organization_management_access(
            user=user,
            organization=organization,
        )
        return organization.id

    async def list_definitions(
        self,
        *,
        user: User,
        skip: int = 0,
        limit: int = 100,
    ) -> OrganizationGuardrailDefinitionsPublic:
        """One page of the organization's definitions, and the total."""
        organization_id = await self._manageable_organization_id(user)
        async with self._uow:
            total = await self._definitions.count_in_organization(organization_id)
            rows = await self._definitions.list_in_organization(
                organization_id, skip=skip, limit=min(limit, _MAX_LIST_LIMIT)
            )
            # Serialized inside the block. The commit that ends it expires every
            # instance in the session, and reading an expired column afterwards
            # is a lazy load in a place that cannot await one.
            data = [_public(row) for row in rows]
        return OrganizationGuardrailDefinitionsPublic(data=data, count=total)

    async def create_definition(
        self, *, user: User, request: OrganizationGuardrailDefinitionCreate
    ) -> OrganizationGuardrailDefinitionPublic:
        """Define a guardrail, encrypting its vendor credentials before they are stored."""
        organization_id = await self._manageable_organization_id(user)
        spec = _definable_spec(request.guardrail_name)
        # A create has nothing stored to restore from, so a ``***`` here is a
        # literal the caller typed and stays one.
        _validate_arguments(spec, request.create_kwargs)
        plain, secrets = _split_by_secret_flag(spec, request.create_kwargs)

        async with self._uow:
            if await self._definitions.count_in_organization(organization_id) >= MAX_DEFINITIONS_PER_ORGANIZATION:
                raise OrganizationGuardrailDefinitionLimitReachedError(MAX_DEFINITIONS_PER_ORGANIZATION)

            definition = OrganizationGuardrailDefinition(
                organization_id=organization_id,
                name=request.name,
                guardrail_name=request.guardrail_name,
                create_kwargs=plain,
                encrypted_create_secrets=_encrypted_secrets(secrets),
                enabled=request.enabled,
            )
            if not await self._definitions.add_unless_name_taken(definition):
                raise OrganizationGuardrailDefinitionAlreadyExistsError(request.name)
            return _public(definition)

    async def update_definition(
        self, *, user: User, definition_id: uuid.UUID, request: OrganizationGuardrailDefinitionUpdate
    ) -> OrganizationGuardrailDefinitionPublic:
        """Apply the fields this request set, leaving the rest as they were."""
        organization_id = await self._manageable_organization_id(user)

        async with self._uow:
            definition = await self._definitions.get_in_organization(definition_id, organization_id)
            if definition is None:
                raise OrganizationGuardrailDefinitionNotFoundError(definition_id)

            guardrail_name = request.guardrail_name if request.guardrail_name is not None else definition.guardrail_name
            arguments = _arguments_after(definition, request, guardrail_name)
            if arguments is not None:
                spec = _definable_spec(guardrail_name)
                _validate_arguments(spec, arguments)
                plain, secrets = _split_by_secret_flag(spec, arguments)
                definition.create_kwargs = plain
                definition.encrypted_create_secrets = _encrypted_secrets(secrets)

            if request.name is not None:
                definition.name = request.name
            if request.guardrail_name is not None:
                definition.guardrail_name = guardrail_name
            if request.enabled is not None:
                definition.enabled = request.enabled

            # Read off the row before the flush. A refused name rolls the block
            # back, which expires every instance in the session, so reading it
            # afterwards would be a lazy load where none can be awaited.
            attempted_name = definition.name
            if not await self._definitions.flush_unless_name_taken():
                raise OrganizationGuardrailDefinitionAlreadyExistsError(attempted_name)
            return _public(definition)

    async def delete_definition(self, *, user: User, definition_id: uuid.UUID) -> None:
        """Drop a definition and the credentials it holds.

        Use ``enabled: false`` instead to stop the guardrail while keeping both.

        A mandate that named this definition would make the database refuse the
        delete, the link being ``RESTRICT`` so that dropping a definition cannot
        silently stop a guardrail running. Nothing can name one yet, and the
        refusal a caller is owed belongs with the write that makes the link.
        """
        organization_id = await self._manageable_organization_id(user)
        async with self._uow:
            definition = await self._definitions.get_in_organization(definition_id, organization_id)
            if definition is None:
                raise OrganizationGuardrailDefinitionNotFoundError(definition_id)
            await self._definitions.delete(definition)
