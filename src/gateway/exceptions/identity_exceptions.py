"""Errors the sign-in and account surfaces raise, and the HTTP status each carries."""

from fastapi import status

from gateway.exceptions import (
    TenancyConflictError,
    TenancyError,
    TenancyForbiddenError,
    TenancyNotFoundError,
    TenancyValidationError,
)


class DeploymentAdministrationUnavailableError(TenancyNotFoundError):
    """The caller is not an operator of this deployment, so the surface is not there.

    A 404 and not the 403 the condition really is, because a surface that answers
    403 confirms it exists to anyone who asks, which for a deployment-wide
    account list is a thing worth not confirming. A caller who is not an operator
    gets the answer an unentitled deployment gets, which is the one that tells
    them least. The hosted backends' own admin surface refuses the same way
    (mozilla-ai/otari-ai#1842).
    """

    def __init__(self) -> None:
        super().__init__("Not found")


class DeploymentUserNotFoundError(TenancyNotFoundError):
    def __init__(self, user_id: object):
        super().__init__(f"User {user_id} not found")


class DeploymentUserSelfChangeError(TenancyValidationError):
    """An operator aiming one of these flags at their own account.

    Deactivating yourself ends the session you are holding, and clearing your own
    superuser flag takes away the page you did it from. Neither is recoverable
    from the dashboard, so both are refused rather than confirmed: an operator
    who really means to stand down has another operator do it, or the deployment
    still has its bootstrap identity.
    """


class BootstrapOperatorProtectedError(TenancyValidationError):
    """A change aimed at the identity ``tenancy_bootstrap_user_id`` names.

    That identity is what master-key sign-in mints a session for, and
    ``resolve_dashboard_session`` refuses a deactivated one, so deactivating it
    turns the deployment's fallback credential into a session that dies on
    arrival. Its superuser flag is protected for the matching reason: the marker
    is the gate that survives a cleared flag, and the two together are what keep
    a deployment from being locked out of its own administration.
    """


class EmptyDeploymentUserUpdateError(TenancyValidationError):
    """A change request that names neither flag.

    Refused rather than answered as a no-op: the two fields are optional so that
    deactivating an account and changing what it may administer stay separate
    decisions, and a body that set neither is a decision that got lost on the way.
    """

    def __init__(self) -> None:
        super().__init__("Set is_active or is_superuser; a change naming neither does nothing")


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


class OAuthNotConfiguredError(TenancyValidationError):
    """This deployment configures no client credentials for the named provider.

    503 rather than the 400 its base carries, for the reason
    ``PasskeysNotConfiguredError`` gives: nothing is wrong with the request, the
    deployment is not set up to answer it. The message names the settings,
    because the only caller who reaches this meant to offer that provider and
    needs to know which two lines are missing.

    Reachable at all only because the dashboard hides a provider this deployment
    does not configure: the affordance is absent rather than disabled
    (``GET /v1/bootstrap``'s ``oauth_providers``), so this answers a bookmark or
    a hand-made request rather than a button somebody was offered.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"{provider} sign-in is not configured on this deployment. "
            f"Set oauth_{provider}_client_id and oauth_{provider}_client_secret, and public_base_url, then restart."
        )


class OAuthExchangeError(TenancyValidationError):
    """The authorization code did not exchange for an identity.

    One error for every way the exchange can fail (a code already spent, a code
    that expired, a redirect URI the provider does not recognize, the provider
    being unreachable), carrying a fixed sentence to the caller.

    The provider's own reason is deliberately not interpolated into the message.
    apron-auth's exchange errors carry the provider's RFC 6749 ``error`` and
    ``error_description`` verbatim, and this message reaches both the HTTP
    client and the log aggregator (CWE-532). Chaining keeps that payload on the
    traceback, which is where debugging needs it and where the error-detail
    boundary in ``gateway.main`` leaves it.
    """

    def __init__(self, provider: str) -> None:
        super().__init__(f"{provider} did not complete the sign-in. Try again.")


class OAuthStateError(TenancyValidationError):
    """The callback did not carry a ``state`` this deployment is still waiting on.

    Covers every way that can be true (never minted here, already spent,
    expired, or minted for a different provider), because they are one answer to
    the caller: start the sign-in again. Which of them it was stays in the log.

    Deliberately indistinguishable from the outside. Telling an attacker that a
    state existed but had expired, as against never existing at all, tells them
    their guess was in the right shape.
    """

    def __init__(self) -> None:
        super().__init__("That sign-in did not start here, or it expired. Start again.")


class OAuthEmailNotVerifiedError(TenancyError):
    """The provider returned an address it will not vouch for.

    401, and named rather than collapsed into the refusal below, for the reason
    ``InvalidCredentialsError``'s own docstring carves out: the distinctions
    survive where a caller has already authenticated. By the time this is
    raised the provider has confirmed the person at the browser holds that
    account, so naming the reason tells them about themselves and nobody else,
    and it is the one refusal here they can act on without an operator.

    An address the provider simply did not speak to (apron-auth reports
    ``email_verified`` as tri-state, and silence is not an assertion) lands
    here too. That is the point: an unasserted address is treated as
    unverified rather than laundered into a verified identity.
    """

    status_code = status.HTTP_401_UNAUTHORIZED

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"{provider} did not confirm that address is yours, so it cannot sign you in here. "
            f"Verify your email address with {provider} and try again."
        )


class OAuthIdentityUnknownError(TenancyError):
    """No account on this deployment signs in as that external identity.

    401, and the message says what to do, because the caller has already proven
    to the provider that they hold the address: the enumeration risk
    ``InvalidCredentialsError`` exists to close is a caller asking about
    *other people's* addresses, and nobody can complete a consent screen for an
    address they do not control.

    A deactivated identity is refused here rather than in an error of its own.
    Deactivating somebody has to close every road in, and "your account is
    switched off" and "there is no account" are the same instruction to the same
    person: talk to whoever administers this gateway. Saying which would let
    somebody an operator has already shut out keep confirming their account is
    still on file.

    This is the base build's roster policy speaking, not a property of OAuth.
    Otari's signup claims an identity an operator already added and never
    creates one from nothing (``user_service.create_user_for_signup``), and
    social sign-in does not get to be the exception that lets any holder of a
    Google account in. An overlay that provisions on first sight binds its own
    ``IdentityProviderPort`` adapter and never raises this.
    """

    status_code = status.HTTP_401_UNAUTHORIZED

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"That {provider} account is not registered on this gateway. "
            "Ask whoever administers it to add your email address, then sign in again."
        )


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


__all__ = [
    "BootstrapOperatorProtectedError",
    "CurrentPasswordIncorrectError",
    "CurrentPasswordRequiredError",
    "DeploymentAdministrationUnavailableError",
    "DeploymentUserNotFoundError",
    "DeploymentUserSelfChangeError",
    "EmailAlreadyInUseError",
    "EmailChangeNotSupportedError",
    "EmailNotVerifiedError",
    "EmptyDeploymentUserUpdateError",
    "InvalidCredentialsError",
    "InvalidEmailError",
    "OAuthEmailNotVerifiedError",
    "OAuthExchangeError",
    "OAuthIdentityUnknownError",
    "OAuthNotConfiguredError",
    "OAuthStateError",
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
    "SignInAddressRequiredError",
    "UnmodifiedPasswordError",
    "VerificationTokenInvalidError",
]
