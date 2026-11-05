"""What the caps that gate an API key leave it, read the way the budget gate counts them."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from gateway.models.api_keys import APIKey
from gateway.models.budgets import SCOPE_ORGANIZATION, SCOPE_WORKSPACE
from gateway.models.money import as_float
from gateway.repositories.budgets import BudgetRepositories, CeilingUsage, LimitUsage, ScopeIdSets
from gateway.schemas.budgets import KeyBudgetPublic, KeyLimitPublic
from gateway.services.budgets._periods import period_window
from gateway.services.budgets._scoped_enforcement import SCOPE_PRECEDENCE, identity_uuid
from gateway.services.organizations import UserAccess
from gateway.services.tenancy.organization_service import OrganizationService

# Ceilings a key shares with everyone else in its workspace or organization: a key
# holder reads how much room they leave, not their limit or the others' spend.
_SHARED_SCOPES = frozenset({SCOPE_WORKSPACE, SCOPE_ORGANIZATION})

_USER_SCOPE = "user"


class _KeyLimits:
    """The caps a request made with a key has to pass, as the gate sees them now."""

    def __init__(self, repositories: BudgetRepositories, organizations: OrganizationService, users: UserAccess) -> None:
        self._budgets = repositories.budgets
        self._ceilings = repositories.ceilings
        self._organizations = organizations
        self._users = users

    async def read(self, api_key: APIKey) -> KeyBudgetPublic:
        """Every cap on the key: its user's budget, then each ceiling on a scope it bills to, most specific first.

        The scopes are the ones the gate resolves for a request with this key: its
        workspace and that workspace's organization, the key itself, and its
        owner's memberships of both. The user's budget is the key owner's, which a
        service key's end users do not share.
        """
        limits: list[LimitUsage] = []
        if api_key.user_id is not None:
            user_limit = await self._user_limit(api_key.user_id)
            if user_limit is not None:
                limits.append(user_limit)
        ceilings = await self._ceilings.usage_in_scopes(await self._scopes(api_key))
        limits.extend(
            sorted(
                ceilings,
                key=lambda ceiling: (
                    SCOPE_PRECEDENCE.get(ceiling.scope, len(SCOPE_PRECEDENCE)),
                    0 if ceiling.provider_key_id is not None else 1,
                ),
            )
        )
        now = datetime.now(UTC)
        public = [_public(limit, now) for limit in limits]
        return KeyBudgetPublic(budgets_enforced=True, remaining_usd=_least_remaining(public), limits=public)

    async def _user_limit(self, user_id: str) -> LimitUsage | None:
        """The budget a user is on with the user's use of it, or None when it is on none."""
        use = await self._users.budget_use(user_id)
        if use is None:
            return None
        budget = (await self._budgets.get_many([use.budget_id])).get(use.budget_id)
        if budget is None:
            return None
        return LimitUsage(
            scope=_USER_SCOPE,
            provider_key_id=None,
            budget_duration_sec=budget.budget_duration_sec,
            reset_alignment=budget.reset_alignment,
            usage=CeilingUsage(
                max_budget=budget.max_budget,
                current_spend=use.spend,
                reserved_spend=use.reserved,
                token_limit=budget.token_limit,
                current_tokens=use.current_tokens,
                reserved_tokens=use.reserved_tokens,
                request_limit=budget.request_limit,
                current_requests=use.current_requests,
                reserved_requests=use.reserved_requests,
                period_end=use.next_budget_reset_at,
            ),
        )

    async def _scopes(self, api_key: APIKey) -> ScopeIdSets:
        """The scopes a request with this key bills to, resolved as the budget gate resolves them."""
        workspace_id = api_key.workspace_id
        organization_id = await self._organizations.get_organization_id_for_workspace(workspace_id)
        identity = None if api_key.user_id is None else identity_uuid(api_key.user_id)
        workspace_member_id: uuid.UUID | None = None
        organization_member_id: uuid.UUID | None = None
        if identity is not None and organization_id is not None:
            workspace_member_id, organization_member_id = await self._organizations.get_membership_ids(
                workspace_id=workspace_id, organization_id=organization_id, user_id=identity
            )
        return ScopeIdSets(
            organization_ids=() if organization_id is None else (str(organization_id),),
            workspace_ids=(str(workspace_id),),
            organization_member_ids=() if organization_member_id is None else (str(organization_member_id),),
            workspace_member_ids=() if workspace_member_id is None else (str(workspace_member_id),),
            api_key_ids=(api_key.id,),
        )


def _public(limit: LimitUsage, now: datetime) -> KeyLimitPublic:
    """One cap as the key holder reads it.

    Used means recorded use plus the holds of requests still in flight, which is
    what the gate refuses on. A window that has ended reads as the fresh window
    the next request will roll it to: recorded use back to zero, holds kept.
    """
    usage = limit.usage
    resets_at = usage.period_end
    ended = resets_at is not None and now >= resets_at
    if ended:
        try:
            window = period_window(now, duration=limit.budget_duration_sec, alignment=limit.reset_alignment)
        except ValueError:
            # An alignment written around the API; the gate leaves such a window in place.
            window = None
            ended = False
        else:
            resets_at = None if window is None else window[1]
    spent = Decimal(0) if ended else usage.current_spend
    tokens = 0 if ended else usage.current_tokens
    requests = 0 if ended else usage.current_requests
    # Kept exact until the response: the gate compares decimals, and a float
    # difference would report a few units in the last place less than it admits.
    used_usd = spent + usage.reserved_spend
    used_tokens = tokens + usage.reserved_tokens
    used_requests = requests + usage.reserved_requests
    max_budget = usage.max_budget
    shared = limit.scope in _SHARED_SCOPES
    return KeyLimitPublic(
        scope=limit.scope,
        provider=limit.provider_key_id,
        resets_at=resets_at,
        limit_usd=None if shared else as_float(max_budget),
        used_usd=None if shared or max_budget is None else as_float(used_usd),
        remaining_usd=None if max_budget is None else as_float(max(Decimal(0), max_budget - used_usd)),
        limit_tokens=None if shared else usage.token_limit,
        used_tokens=None if shared or usage.token_limit is None else used_tokens,
        remaining_tokens=None if usage.token_limit is None else max(0, usage.token_limit - used_tokens),
        limit_requests=None if shared else usage.request_limit,
        used_requests=None if shared or usage.request_limit is None else used_requests,
        remaining_requests=None if usage.request_limit is None else max(0, usage.request_limit - used_requests),
    )


def _least_remaining(limits: Sequence[KeyLimitPublic]) -> float | None:
    """The fewest dollars any cap on every provider leaves, or None when none caps dollars."""
    remaining = [limit.remaining_usd for limit in limits if limit.provider is None and limit.remaining_usd is not None]
    return min(remaining) if remaining else None
