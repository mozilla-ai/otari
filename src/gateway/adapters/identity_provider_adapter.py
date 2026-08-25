"""Identity adapter enforcing the base build's roster policy on an OAuth sign-in.

Satisfies :class:`gateway.ports.identity_provider_port.IdentityProviderPort` with
the policy Otari's base build already applies to every other way in: an account
exists here because an operator put it here. A social identity signs in as an
account already on the roster and never creates one, so enabling Google or GitHub
sign-in widens *how* a member authenticates, never *who* may.

This is a real implementation and not a Null Object, per ``ARCHITECTURE.md``'s
cardinal property. There is a live decision behind the port (link, refuse, and
whether the provider's assertion is enough to lift the local verification gate),
and it is the decision an overlay is most likely to want to replace: a hosted
edition provisions on first sight, and an enterprise edition maps a directory
connection onto an organization. Both bind here without editing this tree.
"""

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tenancy import User
from gateway.repositories.tenancy import UserRepository
from gateway.services.tenancy.email_address import validated_email
from gateway.services.tenancy.errors import (
    InvalidEmailError,
    OAuthEmailNotVerifiedError,
    OAuthIdentityUnknownError,
)


class RosterIdentityProviderAdapter:
    """Resolves an OAuth identity onto an account an operator already added.

    Holds the request's session because resolving is a database read and, on the
    two paths below, a write. It writes but never commits: the sign-in route owns
    the transaction, so the link and the verification stamp land with the session
    row that sign-in mints, or with neither.

    Constructible with no session, like every other core adapter, because the
    container builds one per request whether or not the request will use it. It
    is ``None`` only in hybrid mode, which mounts no sign-in route at all
    (``api.main.register_routers``), so nothing ever resolves an identity there.
    """

    def __init__(self, session: AsyncSession | None) -> None:
        self._session = session

    async def resolve(
        self,
        *,
        provider: str,
        email: str | None,
        full_name: str | None,
        email_verified: bool,
    ) -> User:
        """Return the roster identity this OAuth identity signs in as.

        The provider's assertion is what this trusts, and it must be an
        assertion: an address the provider returned but would not vouch for is
        refused before any lookup, so an unverified or merely unasserted address
        never selects a row. Refusing first also keeps the refusal from
        depending on whether that address happens to be on the roster.

        Two writes, both conditional and neither committed here:

        - The provider is recorded on an identity that had none, which is what
          "linking" means on this schema. An identity that already names a
          different provider is *not* rewritten: a second provider vouching for
          the same address still signs the same person in, and the column keeps
          the provider that arrived first.

          "First" is enforced rather than hoped for. Both writes below are
          read-then-write on a row with no unique index to lose to, so two
          sign-ins racing on one identity with two *different* providers could
          otherwise both see NULL and the later commit would win. The row is
          locked before either is decided, which is the shape otari#729 settled
          for verification and reset redemption on this same table.

          PostgreSQL only, per ``UserRepository.lock``: ``FOR UPDATE`` is a
          no-op on SQLite. The consequence there is which of two provider names
          lands in a column this edition never reads, so it is a documented
          limit rather than a reason to serialize differently.
        - A provider-verified address stamps ``email_verified_at`` if it is
          unset, which lifts the local sign-in gate the password path enforces
          (``user_service.authenticate``). That is not a shortcut around
          verification, it is a stronger proof of the same fact than this
          gateway's own mail loop produces, and it is the one way a deployment
          that cannot send mail can still let a member in.

        Raises:
            OAuthEmailNotVerifiedError: If the provider returned no address, or
                one it does not affirmatively vouch for.
            OAuthIdentityUnknownError: If no active identity here holds that
                address.

        """
        assert self._session is not None, "resolving an identity needs a database session"
        if not email_verified or not email:
            raise OAuthEmailNotVerifiedError(provider)
        # Normalized the way every other address on this deployment is, so a
        # provider returning a differently-cased address still finds its row.
        # A provider that returns something this gateway would never have
        # stored is refused as unknown rather than as malformed: it is not the
        # caller's typo, and it names no account here either way.
        try:
            address = validated_email(email)
        except InvalidEmailError as error:
            raise OAuthIdentityUnknownError(provider) from error

        users = UserRepository(self._session)
        identity = await users.get_by_email(address)
        # Deactivated collapses into "unknown" deliberately; see
        # ``OAuthIdentityUnknownError``.
        if identity is None or not identity.is_active:
            raise OAuthIdentityUnknownError(provider)

        # Locked before anything below is decided, so the reads the two writes
        # branch on are still true when they land. Re-read through the lock for
        # the same reason otari#729 re-resolves after taking it: the row this
        # transaction is holding may not be the row it first read.
        await users.lock(identity.id)
        await self._session.refresh(identity)

        if identity.oauth_provider is None:
            identity.oauth_provider = provider
        if identity.email_verified_at is None:
            identity.email_verified_at = datetime.now(UTC)
        # Only if the roster row has none: a display name an operator or the
        # person themselves set here is theirs, and a provider profile does not
        # get to overwrite it on every sign-in.
        if not identity.full_name and full_name:
            identity.full_name = full_name
        self._session.add(identity)
        return identity


__all__ = ["RosterIdentityProviderAdapter"]
