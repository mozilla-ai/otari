"""Email-domain claims, their DNS proof, and the sign-in that acts on them.

Rehomed from the platform's ``OrganizationDomainService`` plus the auto-join
half of ``OrganizationMembershipService``, converted to async. It is one module
rather than two because the two halves only make sense against each other: the
verification rule exists to decide what auto-join is allowed to act on, and
reading the claim surface without the sign-in it feeds makes the DNS proof look
like decoration.

**What the DNS proof is for.** Anyone may claim any domain here, so the claim
alone is worth nothing: without a proof, an organization could name a
competitor's domain and collect everyone who signs in with an address there.
Publishing a TXT record at the domain's apex is the check that the claimant
controls it, and until that lands, ``verified_at`` is null and auto-join skips
the row entirely.

**Why auto-join runs at sign-in rather than at signup.** Verifying a domain has
to sweep in the accounts that already existed, which is the ordinary case: an
admin claims their company's domain on a deployment their colleagues already
use. Running on every sign-in gets that for free and stays correct when a claim
is enabled, disabled or re-verified later, at the cost of one indexed lookup per
sign-in.
"""

import secrets
import uuid
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.email_domains import (
    email_domain,
    is_public_email_domain,
    is_registrable_domain,
    normalized_domain,
)
from gateway.log_config import logger
from gateway.models.tenancy import (
    DOMAIN_VERIFICATION_TXT_PREFIX,
    OrganizationDomain,
    OrganizationDomainCreateRequest,
    OrganizationDomainPublic,
    OrganizationDomainsPublic,
    OrganizationDomainUpdateRequest,
    OrganizationMember,
    User,
)
from gateway.repositories.tenancy.organization_domain_repository import OrganizationDomainRepository
from gateway.repositories.tenancy.organization_member_repository import OrganizationMemberRepository
from gateway.services.tenancy.domain_verification import resolve_txt_records
from gateway.services.tenancy.errors import (
    OrganizationDomainAlreadyClaimedError,
    OrganizationDomainNotFoundError,
    OrganizationDomainNotVerifiedError,
    PublicEmailDomainError,
    UnregistrableDomainError,
)
from gateway.services.tenancy.organization_service import OrganizationService

# How long a verification token is, in bytes of entropy before hex encoding.
# The token is the whole secret: anyone who can read it can publish the record
# and complete someone else's claim, and it is published in DNS where it is
# world-readable, so it has to be unguessable rather than merely unique.
_VERIFICATION_TOKEN_BYTES = 24


class OrganizationDomainService:
    """Business logic for the organization email-domain surface."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.domains = OrganizationDomainRepository(db)
        self.members = OrganizationMemberRepository(db)
        self.organizations = OrganizationService(db)

    # ------------------------------------------------------------------
    # The claim surface
    # ------------------------------------------------------------------

    async def list_domains_for_user(self, *, user: User) -> OrganizationDomainsPublic:
        """List the caller's organization's domain claims. Owners and admins only.

        Management-gated like the rest of this surface: a row carries the
        verification token in the record an admin has to publish, and that token
        is what completes a claim.
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)

        rows = await self.domains.list_by_organization(organization.id)
        return OrganizationDomainsPublic(data=[self._to_public(row) for row in rows], count=len(rows))

    async def create_domain_for_user(
        self,
        *,
        user: User,
        request: OrganizationDomainCreateRequest,
    ) -> OrganizationDomainPublic:
        """Claim a domain for the caller's organization. Owners and admins only.

        The claim lands unverified and inert; ``verify_domain_for_user`` is what
        makes it act on anyone.
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)

        domain = self._validated_claim(request.domain)

        # The pre-check answers the ordinary case (a domain another organization
        # already holds) with a clean conflict; the caught IntegrityError closes
        # the race where two admins claim the same domain at once. Both are
        # needed: the first alone loses the race, the second alone would make a
        # routine conflict indistinguishable from a bug.
        if await self.domains.get_by_domain(domain) is not None:
            raise OrganizationDomainAlreadyClaimedError(domain)

        try:
            async with self.db.begin_nested():
                row = await self.domains.create_domain(
                    organization_id=organization.id,
                    domain=domain,
                    default_role=request.default_role,
                    enabled=request.enabled,
                    verification_token=secrets.token_hex(_VERIFICATION_TOKEN_BYTES),
                )
        except IntegrityError as exc:
            raise OrganizationDomainAlreadyClaimedError(domain) from exc

        await self.db.commit()
        await self.db.refresh(row)
        return self._to_public(row)

    async def update_domain_for_user(
        self,
        *,
        user: User,
        organization_domain_id: uuid.UUID,
        request: OrganizationDomainUpdateRequest,
    ) -> OrganizationDomainPublic:
        """Change a claim's role or enabled flag. Owners and admins only.

        Neither the domain nor the verification state can be edited here. A
        different domain is a different claim and needs its own proof, and
        letting ``verified_at`` be set directly would make the DNS check
        optional for anyone who can reach this endpoint.
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)

        row = await self._require_domain(organization_domain_id, organization.id)

        # ``exclude_unset`` distinguishes "not mentioned" from "sent as null",
        # and the null is then dropped as well: both columns are NOT NULL, so a
        # PATCH naming one explicitly as null would otherwise fail on the
        # constraint rather than being the no-op the caller meant.
        update_data = {
            key: value for key, value in request.model_dump(exclude_unset=True).items() if value is not None
        }
        updated = await self.domains.update_domain(row, update_data)
        await self.db.commit()
        await self.db.refresh(updated)
        return self._to_public(updated)

    async def verify_domain_for_user(
        self,
        *,
        user: User,
        organization_domain_id: uuid.UUID,
    ) -> OrganizationDomainPublic:
        """Prove control of a claimed domain by finding its TXT record.

        Idempotent: an already-verified claim is returned untouched without a
        second lookup, so a double click costs nothing and re-verifying cannot
        move ``verified_at`` forward.
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)

        row = await self._require_domain(organization_domain_id, organization.id)
        if row.verified_at is not None:
            return self._to_public(row)

        expected = f"{DOMAIN_VERIFICATION_TXT_PREFIX}{row.verification_token}"
        if expected not in await resolve_txt_records(row.domain):
            raise OrganizationDomainNotVerifiedError(row.domain)

        verified = await self.domains.mark_verified(row, verified_at=datetime.now(UTC))
        await self.db.commit()
        await self.db.refresh(verified)
        return self._to_public(verified)

    async def delete_domain_for_user(self, *, user: User, organization_domain_id: uuid.UUID) -> None:
        """Drop a claim. Owners and admins only.

        Memberships auto-join already created are deliberately left alone:
        somebody who joined last month is a colleague at that point, not an
        artifact of the claim, and removing the roster on a DNS change would be
        a far bigger action than the one the admin asked for.
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)

        row = await self._require_domain(organization_domain_id, organization.id)
        await self.domains.delete_domain(row)
        await self.db.commit()

    # ------------------------------------------------------------------
    # Sign-in
    # ------------------------------------------------------------------

    async def auto_join_for_user(self, user: User) -> OrganizationMember | None:
        """Add ``user`` to the organization that has proven their email domain.

        Called on every successful sign-in. Returns the membership it created or
        found, or ``None`` when nothing matched, which is the ordinary answer.

        Conservative at every step, because this is the one path that grants
        access without a person deciding to:

        - An unverified address is skipped. Otherwise signing up as
          ``anyone@theircompany.com`` without ever reading the mail would be
          enough to get in.
        - A claim that is disabled, or whose DNS proof has not landed, is
          skipped.
        - An existing membership is returned untouched. A ``suspended`` row
          means somebody was removed on purpose and must not be re-added by
          their next sign-in, and an established role is never overwritten by
          the claim's ``default_role``.
        - ``active_organization_id`` is never moved. Auto-join adds a
          membership; hijacking where the caller is pointed on every sign-in is
          a different and much ruder thing.

        Rows are staged, not committed: the caller owns the transaction, and the
        sign-in that triggered this commits the membership and the session
        together. That is a change from the platform, where auto-join committed
        for itself and had to unwind on failure.
        """
        # An identity with no address at all is the deployment's master-key
        # operator, which signs in through the same route. It has no domain to
        # match and no verification to have done, so it falls out here rather
        # than at the address check below.
        if user.email is None or user.email_verified_at is None:
            return None

        domain = email_domain(user.email)
        if domain is None:
            return None

        claim = await self.domains.get_by_domain(domain)
        if claim is None or not claim.enabled or claim.verified_at is None:
            return None

        existing = await self.members.get_by_organization_and_user(claim.organization_id, user.id)
        if existing is not None:
            return existing

        try:
            # The unique (organization_id, user_id) constraint settles two
            # concurrent sign-ins. The savepoint is what keeps the loser's
            # IntegrityError from poisoning the transaction the session row is
            # also being written in, which would turn a lost race into a failed
            # sign-in.
            async with self.db.begin_nested():
                membership = await self.members.create_membership(
                    organization_id=claim.organization_id,
                    user_id=user.id,
                    role=claim.default_role,
                    status="active",
                )
        except IntegrityError:
            return await self.members.get_by_organization_and_user(claim.organization_id, user.id)

        logger.info(
            "Identity %s auto-joined organization %s as %s by email domain %s",
            user.id,
            claim.organization_id,
            claim.default_role,
            domain,
        )
        return membership

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _require_domain(
        self,
        organization_domain_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> OrganizationDomain:
        row = await self.domains.get_by_id_and_organization(organization_domain_id, organization_id)
        if row is None:
            raise OrganizationDomainNotFoundError(organization_domain_id)
        return row

    @staticmethod
    def _validated_claim(raw_domain: str) -> str:
        domain = normalized_domain(raw_domain)
        if not is_registrable_domain(domain):
            raise UnregistrableDomainError(raw_domain)
        if is_public_email_domain(domain):
            raise PublicEmailDomainError(domain)
        return domain

    @staticmethod
    def _to_public(row: OrganizationDomain) -> OrganizationDomainPublic:
        """Serialize a claim, folding the token into the record to publish.

        Built field by field rather than by validating the ORM row, because
        ``verification_record`` is a property and ``verification_token`` is not
        on the public schema at all: a ``model_validate`` that later gained
        ``from_attributes`` would start copying the raw token across.
        """
        return OrganizationDomainPublic(
            id=row.id,
            organization_id=row.organization_id,
            domain=row.domain,
            default_role=row.default_role,
            enabled=row.enabled,
            verification_record=row.verification_record,
            verified_at=row.verified_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


__all__ = ["OrganizationDomainService"]
