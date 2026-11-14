import secrets
import unicodedata
import uuid
from collections.abc import Collection
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

import idna

from gateway.core.email_domains import blocklist_entry, blocklist_keys
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    BlockedEmailEntryNotFoundError,
    BootstrapOperatorProtectedError,
    DeploymentUserNotFoundError,
    DeploymentUserSelfChangeError,
    InvalidBlockedEmailEntryError,
    OAuthAccountLinkedElsewhereError,
    OAuthEmailNotVerifiedError,
    OAuthIdentityUnknownError,
    PasswordLinkUnavailableError,
    SignInBlockedError,
    SsoIdentityNotAdmittedError,
    SsoSignInRequiredError,
    TermsNotAcceptedError,
)
from gateway.log_config import logger
from gateway.models.identity import (
    BlockedEmail,
    EmailAssertion,
    MailedLinkStatus,
    OIDCConnection,
    ProviderIdentity,
    is_organization_connection_provider,
)
from gateway.models.tenancy import Organization, OrganizationDomainRole, User
from gateway.ports.identity_provider_port import IdentityProviderPort, IdentityResolution
from gateway.repositories.identity import IdentityRepositories
from gateway.services.dashboard_session_service import hash_session_token
from gateway.services.identity._blocklist import EmailBlocklist
from gateway.services.identity._enforcement import (
    has_link_to_organization,
    sso_enforcement_refuses_every_membership,
)
from gateway.services.identity._membership import VerifiedDomainClaims, organization_for_sso_identity
from gateway.services.mail import Mailer
from gateway.services.oauth_service import provider_label
from gateway.services.tenancy import (
    OrganizationService,
    generate_shared_token,
    generate_token,
    hash_token,
    render_password_reset_email,
    sso_blocks_membership,
    unverified_identity_is_claimable,
)

_SESSION_TOKEN_PREFIX = "otari-sess-"


class SignInListener(Protocol):
    """Reacts to a sign-in inside the transaction that records it.

    NOTE: An implementation must not commit or roll back.
    The caller owns the transaction, so a refused sign-in takes the listener's writes back with it.
    """

    async def account_signed_in(self, account: User) -> None:
        """React to ``account`` signing in."""
        ...


@dataclass(frozen=True)
class SignIn:
    """An account that signed in, and the dashboard session the sign-in started.

    ``active_organization_id`` is the organization the session acts in, or ``None`` for an account with no
    live membership. ``new_account`` is true when the sign-in created the account, or is the first way in to
    one nobody had claimed. ``password_removed`` is true when the sign-in removed a password the account held.
    """

    account: User
    active_organization_id: uuid.UUID | None
    new_account: bool
    session_token: str
    session_expires_at: datetime
    password_removed: bool = False


@dataclass(frozen=True)
class EmailBlock:
    """A new blocklist entry, and how many existing accounts it matched."""

    entry: BlockedEmail
    blocked_account_count: int


@dataclass(frozen=True)
class PasswordLink:
    """A link that sets an account's password, and how it reaches the person.

    ``link`` is set only when the link was not mailed, for whoever asked to pass it on.
    """

    mail_sent: bool
    link: str | None
    expires_at: datetime


class IdentityService:
    """Decide which account a person signs in as, and start their dashboard session.

    Each method runs in a Unit of Work block, so what it and its collaborators stage commits together.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: IdentityRepositories,
        *,
        identity_provider: IdentityProviderPort,
        domain_claims: VerifiedDomainClaims,
        organizations: OrganizationService,
        sign_in_listener: SignInListener,
        session_ttl_hours: int,
        terms_url: str | None,
        mailer: Mailer,
        open_signup: bool,
        password_reset_expiry_hours: int,
    ) -> None:
        """``terms_url`` is the terms the deployment publishes, or ``None`` where it publishes none.

        ``domain_claims`` looks up a verified email-domain claim, so an SSO identity's organization can
        be resolved without this service importing another domain's repository.
        """
        self._uow = uow
        self._blocked_emails = repositories.blocked_emails
        self._blocklist = EmailBlocklist(uow, repositories.blocked_emails)
        self._connections = repositories.connections
        self._links = repositories.links
        self._passkeys = repositories.passkeys
        self._sessions = repositories.sessions
        self._users = repositories.users
        self._identity_provider = identity_provider
        self._domain_claims = domain_claims
        self._organizations = organizations
        self._sign_in_listener = sign_in_listener
        self._session_ttl_hours = session_ttl_hours
        self._terms_url = terms_url
        self._mailer = mailer
        self._open_signup = open_signup
        self._password_reset_expiry_hours = password_reset_expiry_hours

    async def sign_in_with_oauth(self, external: ProviderIdentity, *, terms_accepted: bool = False) -> SignIn:
        """Sign in the account that an identity a provider has proven belongs to.

        An identity linked to an account signs in as that account, and its password and verification stay as they are,
        unless its address is only vouched for: the sign-in then proves it, as a first sign-in does.
        An identity with no link signs in as the account the identity provider policy returns, and is linked to it,
        except an identity from one organization's own OIDC connection, which instead signs in to the one account
        that organization's proven domain and connection admit, provisioned fresh on a first sign-in, and never
        reaches the identity provider policy.
        Such an identity is held to that admission on every sign-in, so a link does not outlive the proof or the
        connection it was admitted on.
        An account holds at most one link for each provider.
        Where the deployment publishes terms, a sign-in that creates the account, or is the first way in to one
        nobody had claimed, needs ``terms_accepted`` and records it, as the signup form does.
        The account's changes, what the sign-in listener stages, its active organization, and the session commit
        together.

        Raises:
            OAuthEmailNotVerifiedError: If the provider says the address is not verified.
            OAuthAccountLinkedElsewhereError: If the account the policy returns is linked to another identity
                at the same provider.
            OAuthIdentityUnknownError: If the account is deactivated or gone.
            SsoIdentityNotAdmittedError: If no organization both proves an OIDC identity's email domain and
                holds a connection to the issuer that authenticated it.
            OAuthIdentityLinkConflictError: If a concurrent sign-in linked the identity to another account first.
            TermsNotAcceptedError: If the sign-in would create or first claim an account without the published
                terms. Nothing the sign-in staged is kept, including an organization the policy provisioned.
            SignInBlockedError: If the deployment's email blocklist matches the account's address, with the
                same rollback.
            TenancyError: If the identity provider policy refuses the identity.

        """
        label = provider_label(external.provider)
        if external.email_verified is EmailAssertion.UNVERIFIED:
            raise OAuthEmailNotVerifiedError(label)
        async with self._uow:
            resolution = await self._account_for(external, label)
            if resolution.new_account:
                self._record_terms(resolution.identity, terms_accepted=terms_accepted)
            signed_in = await self.sign_in(resolution.identity, asserted_email=external.email)
        return replace(signed_in, new_account=resolution.new_account, password_removed=resolution.password_removed)

    async def sign_in(self, account: User, *, asserted_email: str | None) -> SignIn:
        """Sign in an account that has proven who it is, and start its dashboard session.

        ``asserted_email`` is the address a provider asserted for this sign-in, carried to the session
        for the blocklist, and ``None`` where no provider asserted one.

        What the sign-in listener stages and the session commit together.

        Raises:
            SignInBlockedError: If the deployment's email blocklist matches either address. Nothing the
                caller staged in the same block is kept, an account a first provider sign-in created included.

        """
        async with self._uow:
            await self._sign_in_listener.account_signed_in(account)
            return await self.start_session(account, asserted_email=asserted_email)

    async def start_session(self, account: User, *, asserted_email: str | None) -> SignIn:
        """Start a dashboard session for ``account`` without running the sign-in listener.

        The session counts as the account's latest sign-in, and acts in the organization the account can act in.

        ``asserted_email`` is the address a provider asserted for this sign-in, and ``None`` where no
        provider asserted one. It is the only address an account reached through an organization's own
        connection has, so the blocklist is checked against it as well as the account's own.

        Raises:
            SignInBlockedError: If the deployment's email blocklist matches either address. The check holds
                the blocklist lock until the session commits, so a block cannot commit between the two.

        """

        # Resolved once: whether the deployment admits this caller does not vary by organization.
        is_operator = await self._organizations.is_deployment_operator(account)

        async def sso_enforcement_blocks(organization: Organization) -> bool:
            membership = await self._organizations.members.get_active_by_organization_and_user(
                organization.id, account.id
            )
            if membership is None:
                return False
            linked = await has_link_to_organization(
                organization, account.id, connections=self._connections, links=self._links
            )
            return sso_blocks_membership(
                organization=organization,
                membership=membership,
                is_linked=linked,
                is_deployment_operator=is_operator,
            )

        async with self._uow:
            # The shared lock before any row lock, so a block holding it exclusive cannot deadlock against this.
            await self._blocked_emails.lock_shared()
            destination = await self._organizations.sign_in_destination(account, blocked=sso_enforcement_blocks)
            if destination.refused_everywhere:
                logger.info(
                    "Refused a sign-in to account %s, which every organization it belongs to admits "
                    "only through that organization's own connection",
                    account.id,
                )
                raise SsoSignInRequiredError
            active_organization_id = destination.organization_id
            now = datetime.now(UTC)
            # A write before the check, so on SQLite, where the lock is a no-op, this transaction holds the one
            # write lock when it reads the blocklist.
            await self._sessions.prune_expired(now)
            presented = {email for email in (account.email, asserted_email) if email is not None}
            if presented and await self.blocked_among(presented):
                logger.info("Refused a sign-in to account %s, whose address is blocked", account.id)
                raise SignInBlockedError
            await self._users.record_sign_in(account.id, at=now)
            token = f"{_SESSION_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
            expires_at = now + timedelta(hours=self._session_ttl_hours)
            await self._sessions.add(
                token_hash=hash_session_token(token), user_id=account.id, created_at=now, expires_at=expires_at
            )
        return SignIn(
            account=account,
            active_organization_id=active_organization_id,
            new_account=False,
            session_token=token,
            session_expires_at=expires_at,
        )

    async def blocked_among(self, emails: Collection[str]) -> set[str]:
        """Return the members of ``emails`` that the deployment's email blocklist matches."""
        return await self._blocklist.blocked_among(emails)

    async def list_blocked_emails(self, *, skip: int, limit: int) -> tuple[list[BlockedEmail], int]:
        """Return a page of the email blocklist, newest entry first, and how many entries there are."""
        async with self._uow:
            return await self._blocked_emails.list_page(skip=skip, limit=limit)

    async def block_email(self, *, actor: User, value: str, reason: str | None) -> EmailBlock:
        """Block an address, or a domain and its subdomains, and end the sessions of every account it matches.

        Raises:
            InvalidBlockedEmailEntryError: If ``value`` is neither an address nor a domain.
            DeploymentUserSelfChangeError: If the entry matches ``actor``'s own address.
            BootstrapOperatorProtectedError: If it matches the bootstrap operator's, whose session master-key sign-in
                mints.
            BlockedEmailEntryExistsError: If the entry is already blocked.

        """
        pattern = blocklist_entry(value)
        if pattern is None:
            raise InvalidBlockedEmailEntryError(value)
        async with self._uow:
            # Before matching, so an account a sign-in in flight creates is either matched here or refused there.
            await self._blocked_emails.lock_exclusive()
            matched = await self._accounts_matching(pattern)
            if actor.id in matched:
                raise DeploymentUserSelfChangeError("You cannot block your own email address")
            bootstrap_id = await self._organizations.bootstrap_operator_id()
            if bootstrap_id is not None and bootstrap_id in matched:
                raise BootstrapOperatorProtectedError("The deployment's bootstrap operator's address cannot be blocked")
            entry = await self._blocked_emails.add(
                pattern=pattern, reason=(reason or "").strip() or None, created_by_user_id=actor.id
            )
            await self._sessions.delete_for_users(matched)
        logger.warning(
            "Email blocklist entry added by operator: actor=%s entry=%s matched_accounts=%s",
            actor.id,
            entry.id,
            len(matched),
        )
        return EmailBlock(entry=entry, blocked_account_count=len(matched))

    async def unblock_email(self, *, actor: User, entry_id: uuid.UUID) -> None:
        """Remove a blocklist entry, so the accounts it matched may sign in again.

        Raises:
            BlockedEmailEntryNotFoundError: If there is no such entry.

        """
        async with self._uow:
            entry = await self._blocked_emails.get(entry_id)
            if entry is None:
                raise BlockedEmailEntryNotFoundError(entry_id)
            await self._blocked_emails.remove(entry)
        logger.warning("Email blocklist entry removed by operator: actor=%s entry=%s", actor.id, entry_id)

    async def _accounts_matching(self, pattern: str) -> set[uuid.UUID]:
        """The ids of every account whose address the entry ``pattern`` blocks.

        An address reaches an account either as the account's own or as the one a provider asserts for it, and
        both are matched. Queried by every spelling of the domain a stored address may carry, then matched
        exactly by ``blocklist_keys``.
        """
        is_domain = "@" not in pattern
        domain = pattern if is_domain else pattern.rsplit("@", 1)[1]
        spellings = {domain}
        try:
            decoded = idna.decode(domain)
        except (idna.IDNAError, UnicodeError):
            decoded = domain
        # A stored address may carry its domain composed or decomposed; ``blocklist_keys`` reads both alike.
        spellings |= {unicodedata.normalize("NFC", decoded), unicodedata.normalize("NFD", decoded)}
        candidates = await self._users.list_with_email_at(spellings, include_subdomains=is_domain)
        # An account reached through an organization's own connection holds no address, so the only
        # address that can match it is the one its provider asserted.
        candidates += await self._links.list_with_email_at(spellings, include_subdomains=is_domain)
        return {user_id for user_id, email in candidates if pattern in blocklist_keys(email)}

    async def issue_password_link(self, account_id: uuid.UUID) -> PasswordLink:
        """Give an account a single-use link that sets its password, for someone who operates the deployment.

        Where this deployment can mail links, the link is mailed to the account's address and not returned.
        Otherwise, or when the send fails, a shared link is returned for the caller to pass on; redeeming it
        records the address as vouched for rather than proven, since the mailbox never saw it.
        A new link replaces any earlier reset link.

        Raises:
            DeploymentUserNotFoundError: If there is no such account.
            PasswordLinkUnavailableError: If the account is deactivated, has no address, its address is blocked, is
                verified and has no password, or is unverified and ``unverified_identity_is_claimable`` refuses it,
                which only happens where ``open_signup`` is off.

        """
        mailed = self._mailer.can_send_links
        async with self._uow:
            account = await self._users.get_locked(account_id)
            if account is None:
                raise DeploymentUserNotFoundError(account_id)
            await self._require_password_link_allowed(account)
            token = generate_token() if mailed else generate_shared_token()
            expires_at = self._stage_reset_token(account, token)
        assert account.email is not None  # guaranteed by _require_password_link_allowed

        if mailed:
            delivery = await self._mailer.send(
                to=account.email,
                message=render_password_reset_email(
                    reset_link=self._mailer.link(_reset_path(token)),
                    expiry_hours=self._password_reset_expiry_hours,
                    asset_root=self._mailer.asset_root,
                ),
            )
            if delivery.delivered:
                logger.info("Mailed a password link for account %s", account.id)
                return PasswordLink(mail_sent=True, link=None, expires_at=expires_at)
            # The mailed token is replaced, so the link handed back is the only one that works.
            async with self._uow:
                token = generate_shared_token()
                expires_at = self._stage_reset_token(account, token)
        logger.info("Issued a password link to pass on for account %s", account.id)
        return PasswordLink(mail_sent=False, link=self._mailer.link(_reset_path(token)), expires_at=expires_at)

    async def _require_password_link_allowed(self, account: User) -> None:
        if not account.is_active:
            raise PasswordLinkUnavailableError("This account is deactivated. Reactivate it first.")
        if account.email is None:
            raise PasswordLinkUnavailableError("This account has no email address to sign in with.")
        if await self.blocked_among([account.email]):
            raise PasswordLinkUnavailableError("This account's address is blocked. Unblock it first.")
        if account.hashed_password is None and account.email_verified_at is not None:
            raise PasswordLinkUnavailableError(
                "This account has no password and signs in with a provider or a passkey. "
                "Once signed in, its owner can add a password from Account settings."
            )
        if not await unverified_identity_is_claimable(self._organizations, account, open_signup=self._open_signup):
            raise PasswordLinkUnavailableError(
                "No organization still has this unverified address as a member or invitee. "
                "Add or invite it again first."
            )
        if await sso_enforcement_refuses_every_membership(
            account, organizations=self._organizations, connections=self._connections, links=self._links
        ):
            raise PasswordLinkUnavailableError(
                "Every organization this account belongs to requires sign-in through its own connection."
            )

    def _stage_reset_token(self, account: User, token: str) -> datetime:
        expires_at = datetime.now(UTC) + timedelta(hours=self._password_reset_expiry_hours)
        account.password_reset_token_hash = hash_token(token)
        account.password_reset_token_expires_at = expires_at
        self._users.stage(account)
        return expires_at

    async def reset_link_status(self, token: str) -> MailedLinkStatus:
        """Say whether a password reset link can still be used, without using it."""
        async with self._uow:
            account = await self._users.get_by_reset_token_hash(hash_token(token))
        return _link_status(account, None if account is None else account.password_reset_token_expires_at)

    async def verification_link_status(self, token: str) -> MailedLinkStatus:
        """Say whether an email verification link can still be used, without using it."""
        async with self._uow:
            account = await self._users.get_by_verification_token_hash(hash_token(token))
        return _link_status(account, None if account is None else account.email_verification_token_expires_at)

    async def _account_for(self, external: ProviderIdentity, label: str) -> IdentityResolution:
        link = await self._links.get_by_identity(provider=external.provider, subject=external.subject)
        if link is not None:
            if is_organization_connection_provider(external.provider):
                await self._require_admitting_connection(external)
            account = await self._active_account(link.user_id, label)
            password_removed = await self._prove_vouched_address(account)
            self._take_display_name_from_sso(account, external)
            self._links.record_asserted_address(link, external)
            return IdentityResolution(identity=account, new_account=False, password_removed=password_removed)

        if is_organization_connection_provider(external.provider):
            return await self._account_for_sso(external)

        resolution = await self._identity_provider.resolve(external)
        account = await self._active_account(resolution.identity.id, label)
        # A concurrent first sign-in can link this identity, or this account at the provider, after that lookup.
        existing = await self._links.get_for_user(provider=external.provider, user_id=account.id)
        if existing is None:
            await self._links.add(external, user_id=account.id)
        elif existing.subject != external.subject:
            logger.warning(
                "Refused a %s sign-in to account %s, which is linked to another identity at that provider",
                external.provider,
                account.id,
            )
            raise OAuthAccountLinkedElsewhereError(label)
        return replace(resolution, identity=account)

    async def _account_for_sso(self, external: ProviderIdentity) -> IdentityResolution:
        """Resolve a first sign-in from one organization's own OIDC connection, bypassing the identity provider port.

        What makes the claimed address usable here is the organization's DNS proof of its domain, not the
        issuer's own assertion about it. A tenant-supplied issuer can assert any address it likes, so the
        assertion is not read, and ``trust_email_verified`` stays off for every connection.

        Raises:
            SsoIdentityNotAdmittedError: If no organization both proves the identity's email domain and
                holds a connection to the issuer that authenticated it.

        """
        connection = await self._require_admitting_connection(external)
        account = await self._organizations.provision_sso_tenancy(
            organization_id=connection.organization_id,
            full_name=external.full_name,
            # NOTE: The column is a plain string. Every writer validates it against this Literal first.
            default_role=cast(OrganizationDomainRole, connection.default_role),
        )
        account.oauth_provider = external.provider
        self._users.stage(account)
        await self._links.add(external, user_id=account.id)
        return IdentityResolution(identity=account, new_account=True)

    async def _require_admitting_connection(self, external: ProviderIdentity) -> OIDCConnection:
        """The connection admitting ``external``, asked on every sign-in rather than once.

        Raises:
            SsoIdentityNotAdmittedError: If no organization both proves the identity's email domain and
                holds an enabled connection to the issuer that authenticated it.

        """
        connection = await organization_for_sso_identity(
            external, connections=self._connections, domains=self._domain_claims, now=datetime.now(UTC)
        )
        if connection is None:
            raise SsoIdentityNotAdmittedError
        return connection

    def _take_display_name_from_sso(self, account: User, external: ProviderIdentity) -> None:
        """Take the display name an organization's own provider reports, for an account only it names.

        Such an account holds no address and no surface here sets its name, so the provider is where
        the name comes from and a later sign-in is when a change to it arrives. A provider reporting
        none leaves the name alone rather than clearing it.
        """
        if not is_organization_connection_provider(external.provider):
            return
        if external.full_name is None or external.full_name == account.full_name:
            return
        account.full_name = external.full_name
        self._users.stage(account)

    async def _prove_vouched_address(self, account: User) -> bool:
        """Treat a sign-in through a provider link as proof of a vouched address, and say if a password went.

        The link was made when the provider verified this account's address, so its holder owns the account.
        A vouched address had a password chosen through a link its mailbox never saw, so that password, the
        reset and verification links, the sessions and the passkeys set up since go, as on a first sign-in.
        """
        if account.email_vouched_at is None:
            return False
        password_removed = account.hashed_password is not None
        account.email_vouched_at = None
        account.hashed_password = None
        account.email_verification_token_hash = None
        account.email_verification_token_expires_at = None
        account.password_reset_token_hash = None
        account.password_reset_token_expires_at = None
        self._users.stage(account)
        await self._sessions.delete_for_user(account.id)
        await self._passkeys.delete_for_user(account.id)
        return password_removed

    def _record_terms(self, account: User, *, terms_accepted: bool) -> None:
        """Stage the acceptance of the published terms on a new account, or refuse one that has none.

        Here rather than in the identity provider policy, so an adapter an overlay binds cannot skip it.
        """
        if self._terms_url is None or account.terms_accepted_at is not None:
            return
        if not terms_accepted:
            raise TermsNotAcceptedError
        account.terms_accepted_at = datetime.now(UTC)
        self._users.stage(account)

    async def _active_account(self, user_id: uuid.UUID, label: str) -> User:
        """Return the account read under a lock, so a concurrent deactivation is seen before the sign-in completes."""
        account = await self._users.get_locked(user_id)
        if account is None or not account.is_active:
            raise OAuthIdentityUnknownError(label)
        return account


def _reset_path(token: str) -> str:
    return f"/#/reset-password?token={token}"


def _link_status(account: User | None, expires_at: datetime | None) -> MailedLinkStatus:
    """Judge a mailed link the way redeeming it would, short of the checks that need its password or the roster."""
    if account is None or not account.is_active or expires_at is None:
        return MailedLinkStatus.INVALID
    return MailedLinkStatus.EXPIRED if expires_at < datetime.now(UTC) else MailedLinkStatus.VALID
