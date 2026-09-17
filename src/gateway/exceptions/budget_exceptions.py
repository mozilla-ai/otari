"""Errors the budget surfaces raise, and the HTTP status each carries."""

from gateway.exceptions import TenancyConflictError, TenancyNotFoundError


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


class OrganizationBudgetNotFoundError(TenancyNotFoundError):
    """No budget under this id belongs to the caller's organization.

    One status and one message for three different facts: the id names nothing,
    it names a deployment budget, or it names another tenant's. Deliberately
    indistinguishable, for the reason :class:`TenancyNotFoundError` gives; telling
    them apart would make the response an existence oracle over other tenants'
    spend configuration.
    """

    def __init__(self, budget_id: object):
        super().__init__(f"Budget {budget_id} not found")


class OrganizationBudgetInUseError(TenancyConflictError):
    """The budget still holds ceilings or workspace defaults.

    Both foreign keys are ``RESTRICT``, so the database would refuse the delete
    anyway, as an ``IntegrityError`` with nothing naming what to go and change.
    Raised here so the refusal can say which and how many.
    """

    def __init__(self, budget_id: object, *, ceilings: int, defaults: int):
        held = []
        if ceilings:
            held.append(f"{ceilings} spend {'ceiling' if ceilings == 1 else 'ceilings'}")
        if defaults:
            held.append(f"{defaults} workspace member {'default' if defaults == 1 else 'defaults'}")
        super().__init__(
            f"Budget {budget_id} is still used by {' and '.join(held)}. Remove or repoint them before deleting it."
        )


class OrganizationBudgetHeldElsewhereError(TenancyConflictError):
    """Something outside this organization's own surface still names the budget.

    ``users.budget_id`` and ``budget_reset_logs.budget_id``, neither of which is
    a tenant's to see. Since otari#881 neither assignment site will point a
    gateway user at a tenant's budget, so a live hold is one made before that,
    and a reset record outlives the assignment that produced it either way. Left
    unchecked the first is nulled out by the ORM and the second fails at the
    commit, so the refusal says the budget is held without naming the rows
    holding it.
    """

    def __init__(self, budget_id: object):
        super().__init__(
            f"Budget {budget_id} is still in use outside this organization and cannot be deleted. "
            "Ask a deployment operator to release it."
        )


class OrganizationScopeNotFoundError(TenancyNotFoundError):
    """The identity a ceiling would cap is not one in the caller's organization.

    Covers a scope id that names nothing and one that names a row in another
    organization, as one answer and for the same reason as
    :class:`OrganizationBudgetNotFoundError`. This is the cross-tenant check on
    the ceilings surface: a scope id travels as a bare uuid with nothing in it
    saying whose it is.
    """

    def __init__(self, scope_type: object, scope_id: object):
        super().__init__(f"No {scope_type} '{scope_id}' in this organization")


class OrganizationScopedBudgetNotFoundError(TenancyNotFoundError):
    def __init__(self, ceiling_id: object):
        super().__init__(f"Spend ceiling {ceiling_id} not found")


class OrganizationScopedBudgetAlreadyExistsError(TenancyConflictError):
    """One ceiling per scope, and per scope and provider.

    The two partial unique indexes on ``scoped_budgets`` are the real
    enforcement; this reports the same rule in words a caller can act on, since
    neither index name says anything useful to one.
    """

    def __init__(self, scope_type: object, scope_id: object):
        super().__init__(
            f"A spend ceiling already exists for this {scope_type} and provider. Edit that ceiling instead."
        )


__all__ = [
    "OrganizationBudgetHeldElsewhereError",
    "OrganizationBudgetInUseError",
    "OrganizationBudgetNotFoundError",
    "OrganizationScopeNotFoundError",
    "OrganizationScopedBudgetAlreadyExistsError",
    "OrganizationScopedBudgetNotFoundError",
    "WorkspaceBudgetDefaultAlreadyExistsError",
    "WorkspaceBudgetDefaultBudgetNotFoundError",
    "WorkspaceBudgetDefaultNotFoundError",
]
