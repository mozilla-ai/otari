"""Errors that the budgets domain may raise.

The surface errors carry the HTTP status each renders as.
The repository errors are internal: a service translates each into a surface error and never renders it.
"""

from gateway.core.error_codes import END_USER_BUDGET_NOT_ALLOWED
from gateway.exceptions import (
    TenancyConflictError,
    TenancyForbiddenError,
    TenancyNotFoundError,
    TenancyValidationError,
)


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
    it names a deployment budget, or it names another tenant's. Telling them apart
    would make the response an existence oracle over other tenants' spend
    configuration.
    """

    def __init__(self, budget_id: object):
        super().__init__(f"Budget {budget_id} not found")


class OrganizationBudgetInUseError(TenancyConflictError):
    """The budget still holds ceilings or workspace defaults.

    Both foreign keys are ``RESTRICT``, so the database refuses the delete anyway,
    without naming what to change. Raised here so the refusal can say which rows
    hold the budget and how many.
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

    ``users.budget_id``, which is not a tenant's to see, so the refusal does not
    name the rows holding it.
    """

    def __init__(self, budget_id: object):
        super().__init__(
            f"Budget {budget_id} is still in use outside this organization and cannot be deleted. "
            "Ask a deployment operator to release it."
        )


class DeploymentBudgetNotFoundError(TenancyNotFoundError):
    def __init__(self, budget_id: object):
        super().__init__(f"Budget with id '{budget_id}' not found")


class DeploymentBudgetOwnedByOrganizationError(TenancyConflictError):
    """The operator may edit an organization's budget but not delete it out from under the tenant."""

    def __init__(self) -> None:
        super().__init__(
            "This budget is owned by an organization. It can be edited here but not "
            "deleted; the organization that owns it manages it."
        )


class DeploymentBudgetNotReplaceableError(TenancyConflictError):
    """A PUT named the id of an organization's budget, which only that organization replaces."""

    def __init__(self, budget_id: str) -> None:
        super().__init__(f"Budget '{budget_id}' belongs to an organization; change it on that organization's budgets")


class DeploymentBudgetIsMemberDefaultError(TenancyConflictError):
    """A workspace hands the budget to its members, so the delete is refused by workspace name."""

    def __init__(self, workspaces: list[str]):
        super().__init__(
            f"This budget is the member default for {', '.join(workspaces)}. Change or remove that "
            "default on the workspace (Organization > Workspaces > Edit) before deleting it."
        )


class DeploymentBudgetIsEndUserDefaultError(TenancyConflictError):
    """Service keys start their end users on the budget, so the delete is refused by key name.

    Deleting it would leave every end user those keys create from then on uncapped.
    """

    def __init__(self, keys: list[str]):
        shown = ", ".join(keys[:5]) + (f" and {len(keys) - 5} more" if len(keys) > 5 else "")
        super().__init__(
            f"This budget is the default end-user budget of {shown}. Change that default on the key "
            "(Keys > Edit > Service key) before deleting it."
        )


class DeploymentBudgetEnforcedError(TenancyConflictError):
    """Ceilings still enforce the budget.

    Counted rather than named: a scope id is a bare uuid, so listing them would say less than the number does.
    """

    def __init__(self, ceilings: int):
        super().__init__(
            f"This budget is enforced by {ceilings} spend {'ceiling' if ceilings == 1 else 'ceilings'}. "
            "A member's ceiling is changed on Members & roles (Edit > Workspace access); others are managed "
            "through /api/v1/scoped-budgets."
        )


class DeploymentBudgetStillReferencedError(TenancyConflictError):
    def __init__(self, budget_id: object):
        super().__init__(f"Budget {budget_id} is still in use and cannot be deleted.")


class OrganizationScopeNotFoundError(TenancyNotFoundError):
    """The identity a ceiling would cap is not one in the caller's organization.

    Covers a scope id that names nothing and one that names a row in another
    organization, as one answer and for the reason
    :class:`OrganizationBudgetNotFoundError` gives. A scope id is a bare uuid, so
    this is the cross-tenant check on the ceilings surface.
    """

    def __init__(self, scope_type: object, scope_id: object):
        super().__init__(f"No {scope_type} '{scope_id}' in this organization")


class OrganizationScopedBudgetNotFoundError(TenancyNotFoundError):
    def __init__(self, ceiling_id: object):
        super().__init__(f"Spend ceiling {ceiling_id} not found")


class OrganizationScopedBudgetAlreadyExistsError(TenancyConflictError):
    """One ceiling per scope, and per scope and provider.

    The two partial unique indexes on ``scoped_budgets`` are the enforcement. This
    reports the same rule in words a caller can act on, because an index name does
    not.
    """

    def __init__(self, scope_type: object, scope_id: object):
        super().__init__(
            f"A spend ceiling already exists for this {scope_type} and provider. Edit that ceiling instead."
        )


class BudgetStillReferencedError(Exception):
    """A row still names the budget, so the delete was refused."""

    def __init__(self, budget_id: object):
        super().__init__(f"Budget {budget_id} is still referenced")


class MemberBudgetPolicyAlreadyExistsError(Exception):
    """A policy already caps this workspace's members for this provider."""

    def __init__(self, workspace_id: object, provider_key_id: object):
        provider = "every provider" if provider_key_id is None else f"provider '{provider_key_id}'"
        super().__init__(f"Workspace {workspace_id} already has a member budget policy for {provider}")


class SpendCeilingAlreadyExistsError(Exception):
    """A ceiling already caps this scope for this provider."""

    def __init__(self, scope_type: object, scope_id: object):
        super().__init__(f"A spend ceiling already exists for {scope_type} {scope_id}")


class EndUserBudgetNotFoundError(TenancyNotFoundError):
    """A service key names an end-user budget that does not exist, or that a tenant owns.

    One answer for both, as ``POST /users`` gives: an end user is a deployment
    user, so a tenant's budget is not one it may be capped at.
    """

    def __init__(self, budget_id: str):
        super().__init__(f"Budget with id '{budget_id}' not found")


class EndUserBudgetNotAllowedError(TenancyForbiddenError):
    """A service key named a budget that is not on its list of end-user budgets."""

    error_code = END_USER_BUDGET_NOT_ALLOWED

    def __init__(self, budget_id: str):
        super().__init__(f"This key may not assign budget '{budget_id}' to its end users")


class EndUserDefaultNotListedError(TenancyValidationError):
    """A key's default end-user budget is not on its list of end-user budgets."""

    def __init__(self) -> None:
        super().__init__("end_user_budget_id must be one of end_user_budget_ids")


class EndUserNotFoundError(TenancyNotFoundError):
    """No end user of the key's owner goes by this id."""

    def __init__(self, external_id: str):
        super().__init__(f"End user '{external_id}' not found")


class NotAServiceKeyError(TenancyValidationError):
    """End users were addressed through a key that cannot have any."""

    def __init__(self, key_id: str):
        super().__init__(f"API key '{key_id}' is not a service key")


class EndUserIdInvalidError(TenancyValidationError):
    """A service key named an end user by an id it cannot be stored under."""

    def __init__(self, requirement: str):
        super().__init__(f"'user' {requirement} to name an end user")


class EndUserOwnerUnavailableError(TenancyForbiddenError):
    """The service key's own user is blocked or deleted, so none of its end users may spend."""

    def __init__(self) -> None:
        super().__init__("The user this key belongs to is blocked")


__all__ = [
    "BudgetStillReferencedError",
    "EndUserBudgetNotAllowedError",
    "EndUserBudgetNotFoundError",
    "EndUserDefaultNotListedError",
    "EndUserIdInvalidError",
    "EndUserNotFoundError",
    "EndUserOwnerUnavailableError",
    "MemberBudgetPolicyAlreadyExistsError",
    "NotAServiceKeyError",
    "OrganizationBudgetHeldElsewhereError",
    "OrganizationBudgetInUseError",
    "OrganizationBudgetNotFoundError",
    "OrganizationScopeNotFoundError",
    "OrganizationScopedBudgetAlreadyExistsError",
    "OrganizationScopedBudgetNotFoundError",
    "SpendCeilingAlreadyExistsError",
    "WorkspaceBudgetDefaultAlreadyExistsError",
    "WorkspaceBudgetDefaultBudgetNotFoundError",
    "WorkspaceBudgetDefaultNotFoundError",
]
