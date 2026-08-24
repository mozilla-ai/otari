"""Domain errors for the tenancy services, and the HTTP status each carries.

The platform maps roughly 25 exception modules to statuses in one central table
(`otari-ai` `backend/app/api/exception_handlers.py`). The gateway has no such
table and its routes raise ``HTTPException`` directly, which would mean a
try/except around every one of the tenancy handlers. Instead each error names
its own status here and one handler, registered in `gateway.main`, renders it as
FastAPI's own ``{"detail": ...}`` body, so a rehomed service keeps raising
domain errors and the routes stay thin.

The status is on the class rather than at the raise site because it is a
property of the condition, not of the endpoint that hit it: a workspace the
caller may not see is a 404 whichever route asked for it.
"""

from fastapi import status


class TenancyError(Exception):
    """Base class for a tenancy operation that cannot be completed."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class TenancyNotFoundError(TenancyError):
    """A resource does not exist, or does not exist *for this caller*.

    The two are deliberately one status. A workspace in another organization
    must not be distinguishable from a workspace that was never created, or the
    404 becomes an existence oracle for other tenants' data.
    """

    status_code = status.HTTP_404_NOT_FOUND


class TenancyForbiddenError(TenancyError):
    """The caller is known, the resource is visible, and the action is not theirs."""

    status_code = status.HTTP_403_FORBIDDEN


class TenancyConflictError(TenancyError):
    """The request collides with something that already exists."""

    status_code = status.HTTP_409_CONFLICT


class TenancyValidationError(TenancyError):
    """The request is well-formed but would leave the tenancy model inconsistent."""

    status_code = status.HTTP_400_BAD_REQUEST


class OrganizationNotFoundError(TenancyNotFoundError):
    def __init__(self, organization_id: object):
        super().__init__(f"Organization {organization_id} not found")


class OrganizationMemberNotFoundError(TenancyNotFoundError):
    def __init__(self, organization_member_id: object):
        super().__init__(f"Organization member {organization_member_id} not found")


class WorkspaceNotFoundError(TenancyNotFoundError):
    def __init__(self, workspace_id: object):
        super().__init__(f"Workspace {workspace_id} not found")


class WorkspaceMemberNotFoundError(TenancyNotFoundError):
    def __init__(self, workspace_id: object, user_id: object):
        super().__init__(f"User {user_id} is not a member of workspace {workspace_id}")


class WorkspaceAlreadyExistsError(TenancyConflictError):
    def __init__(self, name: str):
        super().__init__(f"A workspace named '{name}' already exists in this organization")


class WorkspaceMemberAlreadyExistsError(TenancyConflictError):
    def __init__(self, user_id: object):
        super().__init__(f"User {user_id} is already a member of this workspace")


class NotAuthorizedError(TenancyForbiddenError):
    def __init__(self, message: str = "Not enough privileges to perform this action"):
        super().__init__(message)


class MembershipUpdateError(TenancyValidationError):
    """A membership change the organization's own rules refuse (e.g. the last owner)."""


class NotAnOrganizationMemberError(TenancyValidationError):
    def __init__(self, user_id: object):
        super().__init__(f"User {user_id} is not an active member of this organization")


class InvalidRoleError(TenancyValidationError):
    def __init__(self, role: str, allowed: set[str]):
        super().__init__(f"Invalid role '{role}'; expected one of {', '.join(sorted(allowed))}")


class OrganizationMemberAlreadyExistsError(TenancyConflictError):
    def __init__(self, identifier: object):
        super().__init__(f"{identifier} is already an active member of this organization")


class InvitationAlreadyPendingError(TenancyConflictError):
    """The address already has a live, unexpired invitation, so a fresh one is refused rather than piled on.

    Distinct from ``OrganizationMemberAlreadyExistsError``: that message says
    "already an active member", which is false for an address that is only
    ``invited``. Resending is revoke (which cancels the pending invitation and
    suspends the membership) followed by a fresh invite; once the existing
    invitation's own expiry has passed, a fresh invite supersedes it directly
    instead of raising this.
    """

    def __init__(self, identifier: object):
        super().__init__(f"{identifier} already has a pending invitation")


class InvalidCredentialsError(TenancyError):
    """A password sign-in that did not succeed, without saying which part failed.

    401 rather than 403: nothing is known about the caller, so the answer is
    "authenticate", not "you may not". Declared on this class rather than on a
    shared 401 base, because it is the only 401 the tenancy family raises today
    and a base class with one subclass is an abstraction with no second user.

    **A deliberate departure from the platform's port.** ``user_service`` there
    distinguishes ``UserNotFoundError``, ``OAuthAccountPasswordLoginError``,
    ``IncorrectPasswordError`` and ``EmailNotVerifiedError`` at the sign-in
    endpoint. Each of those answers "does this address hold an account here, and
    how does it sign in", which is a question an unauthenticated caller may ask
    an unlimited number of times. A self-hosted deployment's roster is small
    enough that enumerating it is worth doing, so the four collapse into one
    message here, and `gateway.services.password_service` makes them cost the
    same wall-clock time as well.

    The distinctions survive where a caller has already authenticated:
    ``CurrentPasswordIncorrectError`` and ``PasswordNotSetError`` below say
    exactly what went wrong, because by then the caller is not being told
    anything about somebody else's account.
    """

    status_code = status.HTTP_401_UNAUTHORIZED

    def __init__(self) -> None:
        super().__init__("Incorrect email or password")


class PasskeysNotConfiguredError(TenancyValidationError):
    """This deployment has no relying-party ID, so it cannot run a ceremony.

    503 rather than the 400 its base carries: nothing is wrong with the request,
    the deployment is not set up to answer it, and that is the same shape
    `api.routes.mail` gives an unconfigured mailer. The message names the
    setting, because an operator who reached this endpoint meant to offer
    passkeys and needs to know which line is missing rather than that something
    was refused.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    def __init__(self) -> None:
        super().__init__(
            "Passkeys are unavailable on this deployment: it does not know its own address. "
            "Set public_base_url (or webauthn_rp_id) and restart."
        )


class PasskeyNotFoundError(TenancyNotFoundError):
    def __init__(self, credential_id: object):
        super().__init__(f"Passkey {credential_id} not found")


class PasskeyNameTakenError(TenancyConflictError):
    def __init__(self, name: str):
        super().__init__(f"You already have a passkey named '{name}'")


class PasskeyAlreadyRegisteredError(TenancyConflictError):
    """This authenticator already has a row, possibly on another identity.

    Said plainly rather than hidden, and the wording does not reveal *whose*.
    A caller performing this ceremony is signed in and holds the authenticator
    that just answered, so telling them it is already known here costs nothing;
    telling them which identity holds it would be somebody else's business.
    """

    def __init__(self) -> None:
        super().__init__("That passkey is already registered on this deployment")


class PasskeyLimitReachedError(TenancyValidationError):
    """This identity already holds as many passkeys as it may.

    A ceiling on a table an authenticated caller writes in a loop they control,
    not a policy about how many devices a person should have; see
    ``MAX_PASSKEYS_PER_IDENTITY``. The message says the number, because the only
    useful action is to delete one and the caller cannot count what they cannot
    see.
    """

    def __init__(self, limit: int):
        super().__init__(
            f"You already have {limit} passkeys, which is the most one identity may hold. Delete one first."
        )


class PasskeyCeremonyError(TenancyValidationError):
    """A registration or authentication ceremony did not verify.

    One error for every way the ceremony can fail (an unknown or expired
    challenge, a mismatched origin or relying-party ID, a signature that does
    not check out, an authenticator answering somebody else's challenge),
    carrying the library's reason in the log and a fixed sentence to the caller.

    Undifferentiated on purpose, for the reason ``InvalidCredentialsError``
    gives: the sign-in half of this is reachable unauthenticated, and each
    distinct refusal would answer a question about which credentials this
    deployment holds. The registration half is authenticated and could afford
    to say more, but a caller there cannot act on the distinction either: every
    branch means "try the ceremony again".
    """

    def __init__(self) -> None:
        super().__init__("That passkey could not be verified. Try again.")


class PasskeySignInFailedError(TenancyError):
    """A passkey sign-in that did not succeed, without saying which part failed.

    401 for the reason ``InvalidCredentialsError`` is: nothing is known about
    the caller. Separate from that class only because the message names the
    credential the caller actually used, and being told to check an email and
    password after tapping a passkey is a dead end.
    """

    status_code = status.HTTP_401_UNAUTHORIZED

    def __init__(self) -> None:
        super().__init__("That passkey did not sign you in")


class CurrentPasswordIncorrectError(TenancyValidationError):
    """The current password given with a password change does not match.

    400 and not 401, as the platform's own note says: a 401 on this route reads
    to a browser client as "your session died", and it would sign the caller out
    of a form they filled in correctly except for one field.
    """

    def __init__(self) -> None:
        super().__init__("Current password is incorrect")


class PasswordNotSetError(TenancyValidationError):
    """A password change on an identity that has no password to change."""

    def __init__(self) -> None:
        super().__init__("This identity has no password set; set one instead of changing it")


class UnmodifiedPasswordError(TenancyValidationError):
    """The new password is the one already stored."""

    def __init__(self) -> None:
        super().__init__("The new password cannot be the same as the current one")


class CurrentPasswordRequiredError(TenancyValidationError):
    """A password change from a session, with no current password supplied."""

    def __init__(self) -> None:
        super().__init__("The current password is required to change it")


class EmailChangeNotSupportedError(TenancyValidationError):
    """An address change attempted through the password endpoint.

    Supplying an address is part of claiming an identity that has none. Changing
    one that already exists is a different operation with its own requirements
    (it invalidates a sign-in handle, and the new address has to be verified),
    and it belongs to the verification flow rather than being smuggled in
    alongside a password.
    """

    def __init__(self) -> None:
        super().__init__("This identity already has an email address; changing it is not supported yet")


class PasswordPolicyError(TenancyValidationError):
    """A password that is too short, or longer than bcrypt will hash."""


class SignInAddressRequiredError(TenancyValidationError):
    """Setting a password on an identity that has no address to sign in with.

    The operator identity first boot provisions is a label rather than a sign-in
    address (`gateway.services.tenancy.provisioning_service`), so claiming it
    supplies one. Refused rather than defaulted: a synthesized address would be
    a credential handle nobody knows.
    """

    def __init__(self) -> None:
        super().__init__("This identity has no email address; supply one to sign in with a password")


class EmailAlreadyInUseError(TenancyConflictError):
    """Another identity already holds the address being claimed."""

    def __init__(self, email: str) -> None:
        super().__init__(f"'{email}' already belongs to another identity")


class InvalidEmailError(TenancyValidationError):
    """An address that could not be a claim handle.

    Deliberately a shape check and nothing more. The address is not delivered to
    by this edition, and ownership of it is proven by the claim flow that
    arrives with sign-in, so anything stricter here would be theater.
    """

    def __init__(self, email: str):
        super().__init__(f"'{email}' is not a valid email address")


class WorkspaceNameRequiredError(TenancyValidationError):
    """A workspace name that is absent, null, or blank once trimmed.

    ``Workspace.name`` is NOT NULL and carries no minimum length, and SQLModel
    skips validation when constructing a table instance, so without this a
    ``{"name": null}`` update reaches the column as a NOT NULL violation and a
    ``{"name": ""}`` create stores a nameless workspace.
    """

    def __init__(self) -> None:
        super().__init__("A workspace name is required")


class ForeignTenancyError(TenancyError):
    """The database holds organizations this deployment did not provision.

    A 500 rather than a client error, because nothing the caller sent is wrong:
    the deployment is pointed at a database it cannot serve, and that is an
    operator's problem to fix before any request can succeed.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class OrganizationNameRequiredError(TenancyValidationError):
    """An organization name that is absent, null, or blank once trimmed.

    The request's ``min_length=1`` admits a single space, so this is what stops a
    whitespace-only rename from reaching the column, in the same way
    ``WorkspaceNameRequiredError`` does for a workspace.
    """

    def __init__(self) -> None:
        super().__init__("An organization name is required")


class WorkspaceInUseError(TenancyConflictError):
    """A workspace still holds request-plane rows, which are ON DELETE RESTRICT.

    Keys, usage, aliases and policies are restricted rather than cascaded on
    purpose: a workspace is a billing scope, and deleting one should not take
    the record of what was spent in it with it. So this is a real refusal with a
    real reason, not the integrity error escaping as a 500.
    """

    status_code = status.HTTP_409_CONFLICT

    def __init__(self) -> None:
        super().__init__(
            "This workspace still holds API keys, usage, aliases or routing policies. "
            "Move or delete those first; they are kept rather than cascaded so a "
            "workspace's spend history survives it."
        )


class LastWorkspaceError(TenancyValidationError):
    """Deleting this workspace would leave the organization without one."""

    def __init__(self) -> None:
        super().__init__("An organization keeps at least one workspace; create another before deleting this one")


class OrgProviderKeyNotFoundError(TenancyNotFoundError):
    def __init__(self, key_id: object):
        super().__init__(f"Provider key {key_id} not found")


class OrgProviderKeyNameRequiredError(TenancyValidationError):
    """A key name that is absent, null, or blank once trimmed.

    ``OrgProviderKey.name`` is NOT NULL, and ``OrgProviderKeyUpdateRequest``
    types it as nullable so a client can send an explicit ``null``, the same
    shape ``WorkspaceNameRequiredError`` guards against for a workspace. Left
    unguarded, an explicit ``null`` reaches the database as a NOT NULL
    violation, which the surrounding duplicate-name handling then reports as a
    409 naming a key called "None" rather than the 400 this is.
    """

    def __init__(self) -> None:
        super().__init__("A provider key name is required")


class OrgProviderKeyUnknownProviderError(TenancyValidationError):
    """A ``provider`` that is blank, or does not resolve to a known any-llm implementation.

    ``OrgProviderKey.provider`` is stored verbatim and is exactly the string
    ``cached_org_provider_kwargs`` keys its cache on, matched against a
    resolved selector's ``LLMProvider.value`` at dispatch (see
    ``org_provider_key_service.refresh_org_provider_cache``). Left unguarded,
    a typo, unexpected casing, or an unaliased value (``"OpenAI"``,
    ``"azure-openai"``, trailing whitespace) is accepted with a 201 and then
    never resolves at dispatch, with no error at either point. Mirrors
    ``/v1/provider-credentials``'s ``_validate_instance`` provider_type guard,
    including ``PROVIDER_TYPE_ALIASES`` so an aliased name still resolves.
    """

    def __init__(self, provider: str) -> None:
        if not provider:
            super().__init__("A provider is required")
        else:
            super().__init__(f"'{provider}' is not a known provider implementation")


class OrgProviderKeyUnsafeApiBaseError(TenancyValidationError):
    """An ``api_base`` that resolves to an internal address, gated off.

    Wraps ``services.url_safety.UnsafeURLError`` as a tenancy error so the
    route stays thin; the message is that function's own, which already
    carries no more than the host it refused.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class OrgProviderKeyAlreadyExistsError(TenancyConflictError):
    def __init__(self, provider: str, name: str):
        super().__init__(f"A '{provider}' key named '{name}' already exists in this organization")


class OrgProviderKeyArchivedError(TenancyValidationError):
    """The key is archived, which refuses every mutation except restore."""

    def __init__(self, key_id: object):
        super().__init__(f"Provider key {key_id} is archived; restore it before changing it")


class OrgProviderKeyNotArchivedError(TenancyValidationError):
    """Deletion requires archiving first, the same two-step every irreversible action here takes."""

    def __init__(self, key_id: object):
        super().__init__(f"Provider key {key_id} must be archived before it can be deleted")


class OrgDefaultProviderKeyConflictError(TenancyConflictError):
    """Two concurrent 'set default' calls raced for the same (organization, provider).

    The partial unique index (``uq_org_provider_keys_org_default``) is the
    actual arbiter; this is what the loser's ``IntegrityError`` is mapped to.
    """

    def __init__(self, provider: str) -> None:
        super().__init__(f"Another request just changed the default '{provider}' key; retry")


class OrgProviderKeyDisabledForWorkspaceError(TenancyValidationError):
    """A model restriction was requested for a key this workspace has disabled.

    Refused rather than stored: a restriction on a key the workspace cannot
    use anyway would resurface with a stale list if the key were re-enabled
    later, which `set_workspace_override_for_user` already deletes for the
    opposite transition (see the repository docstring on the cascade).
    """

    def __init__(self) -> None:
        super().__init__("This provider key is disabled for the workspace; enable it before restricting its models")


class WorkspaceProviderKeyOverrideConflictError(TenancyValidationError):
    """A caller asked to pin and disable the same key in the same request.

    Sending one flag lets the other auto-resolve (pinning re-enables a
    disabled key, disabling un-pins a pinned one); sending both explicitly
    true is a contradiction with no safe default to pick.
    """

    def __init__(self) -> None:
        super().__init__("A provider key override cannot be both pinned as default and disabled")


class SecretBoxUnavailableTenancyError(TenancyError):
    """`OTARI_SECRET_KEY` is not configured, so a secret cannot be stored.

    Wraps `services.secret_box.SecretBoxUnavailableError` as a tenancy error so
    the route stays thin (see the module docstring): the underlying error
    carries no key material, and neither does this one. A 500, not the 400 a
    `TenancyValidationError` would carry: the caller sent a well-formed
    request, and a missing secret key is a deployment configuration gap the
    caller cannot fix. Blaming the client here would also keep the condition
    out of 5xx error-rate alerting, which is exactly the audience that can.

    ``stored`` names what could not be stored, so the message points at the
    surface the caller was using. It defaults to the provider credentials this
    error was written for; workspace MCP servers pass their own.
    """

    def __init__(self, stored: str = "provider credentials") -> None:
        super().__init__(f"OTARI_SECRET_KEY is not set; it is required to store {stored}")


# The two below are pricing errors in a tenancy module, because the status
# mapping is what decides where an error class lives here: one handler is
# registered for ``TenancyError`` (see `gateway.main`), so an organization-scoped
# error that wants a status rather than a 500 has to descend from it. Naming them
# for what they are keeps that visible.
class OrganizationPricingNotFoundError(TenancyNotFoundError):
    """No pricing override with this id in the caller's organization.

    One status for "never existed" and "belongs to another organization", the same
    rule the base class states: a distinguishable 404 would tell one tenant which
    ids exist in another.
    """

    def __init__(self, pricing_id: object):
        super().__init__(f"Pricing override {pricing_id} not found")


class OrganizationPricingOverlapError(TenancyConflictError):
    """A period that would overlap one this organization already priced.

    ``model_pricing`` lets a later row shadow an earlier one, because a catalog is
    re-imported wholesale. An override is a commitment for a period, so two
    periods covering one instant is an unanswerable question rather than a newest
    wins rule, and it is refused with both periods named.
    """

    def __init__(self, model_key: str, existing_period: str):
        super().__init__(
            f"An override for '{model_key}' already covers part of that period ({existing_period}). "
            "Change this period, or edit the existing override instead."
        )
class InvitationNotFoundError(TenancyNotFoundError):
    """No invitation matches the token or id given.

    One status for "wrong token", "unknown id", and "someone else's
    invitation", for the same reason ``TenancyNotFoundError`` gives every
    cross-tenant lookup one status: distinguishing them would let a caller
    probe for which is true.
    """

    def __init__(self, identifier: object = "invitation") -> None:
        super().__init__(f"{identifier} not found or already used")


class InvitationExpiredError(TenancyValidationError):
    """The invitation's ``expires_at`` has passed."""

    def __init__(self) -> None:
        super().__init__("This invitation has expired")


class InvitationAlreadyUsedError(TenancyValidationError):
    """The invitation is not ``pending`` (already accepted, cancelled, or expired)."""

    def __init__(self) -> None:
        super().__init__("This invitation has already been used or is no longer valid")


class VerificationTokenInvalidError(TenancyValidationError):
    """A verification token that is unknown, expired, or already consumed.

    One message for all three, the same reasoning ``InvitationNotFoundError``
    gives an unknown-or-foreign invitation: distinguishing "expired" from
    "already used" from "never existed" would let a caller narrow down which
    is true of a token they do not hold.
    """

    def __init__(self) -> None:
        super().__init__("This verification link is invalid, expired, or already used")


class ResetTokenInvalidError(TenancyValidationError):
    """A password-reset token that is unknown, expired, or already consumed.

    Same collapse as ``VerificationTokenInvalidError``, for the same reason.
    """

    def __init__(self) -> None:
        super().__init__("This password reset link is invalid, expired, or already used")


class EmailNotVerifiedError(TenancyForbiddenError):
    """A password sign-in on an identity that has not verified its address.

    Raised only after the password itself has already checked out, which is
    why it is allowed to say what is actually wrong: the distinction the
    module docstring on ``authenticate`` promises survives once a caller has
    proven something, the same way ``CurrentPasswordIncorrectError`` and
    ``PasswordNotSetError`` do.
    """

    def __init__(self) -> None:
        super().__init__("Verify your email before signing in; request a new verification email if yours expired")


class WorkspaceBudgetDefaultNotFoundError(TenancyNotFoundError):
    def __init__(self, default_id: object):
        super().__init__(f"Workspace budget default {default_id} not found")


class WorkspaceBudgetDefaultBudgetNotFoundError(TenancyNotFoundError):
    """The default names a budget that does not exist.

    Only reachable on the way in, when a caller assigns a budget by id. A stored
    default cannot reach it: ``budget_id`` is NOT NULL and the foreign key is
    ``RESTRICT``, so the budget it names cannot be deleted out from under it.
    """

    def __init__(self, budget_id: object):
        super().__init__(f"Budget {budget_id} not found")


class WorkspaceBudgetDefaultAlreadyExistsError(TenancyConflictError):
    def __init__(self, workspace_id: object, provider_key_id: object):
        scope = "every provider" if provider_key_id is None else f"provider '{provider_key_id}'"
        super().__init__(f"Workspace {workspace_id} already has a budget default for {scope}")


class WorkspaceActivationUnavailableError(TenancyConflictError):
    """The first-request setup guide is not on offer for this workspace.

    The deployment turned it off, the workspace is classified out of it, or
    someone dismissed it. A conflict rather than a 403: the caller is allowed to
    manage this workspace, the flow they are asking to act on is simply retired,
    and the message says which of the three it was.
    """


class WorkspaceAlreadyActivatedError(TenancyConflictError):
    """A request in this workspace has already succeeded, so the guide is finished.

    Reached by a browser tab left open across the first successful request, which
    is the one caller likely to ask a retired guide for a credential.
    """

    def __init__(self) -> None:
        super().__init__("This workspace has already served a successful request")


class WorkspaceMcpServerNotFoundError(TenancyNotFoundError):
    def __init__(self, mcp_server_id: object):
        super().__init__(f"MCP server {mcp_server_id} not found")


class WorkspaceMcpServerAlreadyExistsError(TenancyConflictError):
    """A workspace already has an MCP server under this name.

    Refused rather than collapsed onto the existing row: the name is what the
    tool loop labels a server's tools with, so silently reusing it would point
    a caller's request at a different endpoint than the one they just
    configured.
    """

    def __init__(self, workspace_id: object, name: object):
        super().__init__(f"Workspace {workspace_id} already has an MCP server named '{name}'")


class WorkspaceMcpServerUnsafeUrlError(TenancyValidationError):
    """The URL failed the same SSRF and TLS checks a request-body MCP server faces.

    Carries the reason from `services.url_safety.UnsafeURLError` verbatim: it
    names the host and the range it resolved into, which is what an operator
    needs to fix the entry, and it is the operator's own URL either way (this
    surface is management-gated, not a caller-supplied endpoint).
    """

    def __init__(self, reason: str):
        super().__init__(reason)


class WorkspaceMcpServerLimitReachedError(TenancyValidationError):
    """The workspace already holds as many MCP servers as it may.

    A resolved request opens a session to every server it names, so the cap
    bounds the fan-out one workspace can ask a gateway process for.
    """

    def __init__(self, workspace_id: object, limit: int):
        super().__init__(f"Workspace {workspace_id} already has the maximum of {limit} MCP servers")


class WorkspaceWebSearchDomainsExcludedError(TenancyForbiddenError):
    """A request's search allow-list shares no domain with its workspace's.

    The two lists are intersected rather than overridden, so this is the empty
    intersection: every domain the request asked for is one the workspace does
    not permit. Refused rather than run, because an empty effective allow-list
    is read by ``_build_web_search_backend`` as *no* allow-list (an empty list
    is falsy), which would turn the narrowest possible policy into no policy at
    all.
    """

    def __init__(self) -> None:
        super().__init__("The requested search domains are not permitted for this workspace")


__all__ = [
    "CurrentPasswordIncorrectError",
    "CurrentPasswordRequiredError",
    "EmailAlreadyInUseError",
    "EmailChangeNotSupportedError",
    "EmailNotVerifiedError",
    "ForeignTenancyError",
    "InvalidCredentialsError",
    "InvalidEmailError",
    "InvalidRoleError",
    "InvitationAlreadyPendingError",
    "InvitationAlreadyUsedError",
    "InvitationExpiredError",
    "InvitationNotFoundError",
    "LastWorkspaceError",
    "MembershipUpdateError",
    "NotAnOrganizationMemberError",
    "NotAuthorizedError",
    "OrgDefaultProviderKeyConflictError",
    "OrgProviderKeyAlreadyExistsError",
    "OrgProviderKeyArchivedError",
    "OrgProviderKeyDisabledForWorkspaceError",
    "OrgProviderKeyNameRequiredError",
    "OrgProviderKeyNotArchivedError",
    "OrgProviderKeyNotFoundError",
    "OrgProviderKeyUnknownProviderError",
    "OrgProviderKeyUnsafeApiBaseError",
    "OrganizationMemberAlreadyExistsError",
    "OrganizationMemberNotFoundError",
    "OrganizationNameRequiredError",
    "OrganizationNotFoundError",
    "OrganizationPricingNotFoundError",
    "OrganizationPricingOverlapError",
    "PasskeyAlreadyRegisteredError",
    "PasskeyCeremonyError",
    "PasskeyLimitReachedError",
    "PasskeyNameTakenError",
    "PasskeyNotFoundError",
    "PasskeySignInFailedError",
    "PasskeysNotConfiguredError",
    "PasswordNotSetError",
    "PasswordPolicyError",
    "ResetTokenInvalidError",
    "SecretBoxUnavailableTenancyError",
    "SignInAddressRequiredError",
    "TenancyConflictError",
    "TenancyError",
    "TenancyForbiddenError",
    "TenancyNotFoundError",
    "TenancyValidationError",
    "UnmodifiedPasswordError",
    "VerificationTokenInvalidError",
    "WorkspaceAlreadyExistsError",
    "WorkspaceActivationUnavailableError",
    "WorkspaceAlreadyActivatedError",
    "WorkspaceBudgetDefaultAlreadyExistsError",
    "WorkspaceBudgetDefaultBudgetNotFoundError",
    "WorkspaceBudgetDefaultNotFoundError",
    "WorkspaceInUseError",
    "WorkspaceMcpServerAlreadyExistsError",
    "WorkspaceMcpServerLimitReachedError",
    "WorkspaceMcpServerNotFoundError",
    "WorkspaceMcpServerUnsafeUrlError",
    "WorkspaceMemberAlreadyExistsError",
    "WorkspaceMemberNotFoundError",
    "WorkspaceNameRequiredError",
    "WorkspaceNotFoundError",
    "WorkspaceProviderKeyOverrideConflictError",
    "WorkspaceWebSearchDomainsExcludedError",
]
