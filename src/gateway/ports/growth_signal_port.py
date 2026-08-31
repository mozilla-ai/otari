"""Lifecycle notifications for whichever CRM or support messenger a build runs.

The seam between the core and whichever build tells an outside vendor about a
user's lifecycle: signup, activation milestones, onboarding, profile edits, and
account deletion. The core has no growth-marketing or support-messaging vendor of its
own, and that is the point: an operator running Otari serves their own users on
their own channels, and nothing about those users' lifecycle should reach a
vendor the deployment never chose. So the core adapter is a Null Object, and an
overlay binds one that fans these out to the vendors it runs.

Every method is fire-and-forget: it schedules work on the caller's
``BackgroundTasks`` and returns immediately, never raising, so a vendor outage
can never affect the request that triggered it. That is why these methods
return nothing and take a ``BackgroundTasks`` rather than doing the work inline
and handing back a result: there is nothing for a caller to act on, only
something to fire and forget.

Stability: this interface is not frozen while Otari is pre-1.0. Splitting the
CRM and the support messenger behind two ports, should they need to vary
independently, is an anticipated change; overlay authors should expect the
shape to move.
"""

import uuid
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from fastapi import BackgroundTasks


class GrowthActivationEvent(StrEnum):
    """A lifecycle milestone worth notifying a vendor about.

    Owned by the port rather than by whichever vendor answers it, so a caller
    never imports a vendor-flavored constant: it names the milestone in domain
    terms and leaves the vendor's own vocabulary to the adapter.

    ``SIGNED_UP`` overlaps :meth:`GrowthSignalPort.record_signup`, and a signup
    fires that method alone: it carries the address, the name and the creation
    time a vendor needs to make the record in the first place, where this enum
    carries only an identity. The value stays because an adapter whose vendor
    models signup as one more activation event needs something to map onto, but
    nothing in this tree emits it, so an adapter can treat it as an overlay's
    own vocabulary rather than a second notification to de-duplicate against.
    """

    SIGNED_UP = "signed_up"
    API_KEY_CREATED = "api_key_created"  # pragma: allowlist secret
    FIRST_ROUTE_CONFIGURED = "first_route_configured"
    FIRST_REQUEST_ROUTED = "first_request_routed"
    BUDGET_RULE_SET = "budget_rule_set"


class GrowthSignalPort(Protocol):
    """What a build does with a user lifecycle event, if anything."""

    async def record_signup(
        self,
        *,
        background_tasks: BackgroundTasks,
        user_id: uuid.UUID,
        email: str,
        full_name: str | None,
        created_at: datetime,
    ) -> None:
        """Notify of a brand-new account, immediately after it is created."""
        ...

    async def record_activation(
        self,
        *,
        background_tasks: BackgroundTasks,
        event: GrowthActivationEvent,
        user_id: uuid.UUID,
        email: str,
    ) -> None:
        """Notify that a user crossed an activation milestone.

        Callers detect first occurrence themselves, as they already must to
        decide whether to call this at all; an adapter may additionally
        de-duplicate on its own end, so a redundant call is safe.

        Only ``API_KEY_CREATED`` is emitted, from
        ``POST /v1/organizations/me/keys``, where first occurrence is the owner
        holding no key yet. The other three stay uncalled, and an overlay should
        not expect them:

        - ``FIRST_ROUTE_CONFIGURED`` and ``BUDGET_RULE_SET`` would fire from
          ``POST /v1/routing/policies`` and ``POST /v1/budgets``, which are
          deployment-wide and gated on ``require_deployment_operator``. The
          milestone would then name whoever holds deployment-wide authority: on
          a hosted deployment the operator running it rather than any of the
          tenants whose activation a vendor is watching, and under a header
          master key nobody at all, since that credential resolves to the
          bootstrap operator, which deliberately holds no address
          (``services/tenancy/provisioning_service``). The same reason picks the
          member key route over the operator's ``POST /v1/keys``.
        - ``FIRST_REQUEST_ROUTED`` has no moment to fire from.
          ``services/tenancy/workspace_activation_service`` derives activation
          from the first successful usage row every time the guide is read,
          rather than recording it, so there is no instant at which a workspace
          becomes activated, only a polled read that would notify on every load.
        """
        ...

    async def record_onboarding_completed(
        self,
        *,
        background_tasks: BackgroundTasks,
        user_id: uuid.UUID,
        email: str,
        answers: Mapping[str, Any],
        full_name: str | None,
    ) -> None:
        """Notify that a user completed onboarding, with their raw answers.

        ``answers`` are the validated onboarding answers keyed by question key.
        The port carries no vendor property mapping: turning an answer into a
        vendor's object model (a CRM contact property, say) is the adapter's
        job, so a caller never builds a vendor-shaped property dict itself.
        That split is the whole reason the answers cross the seam in domain
        terms.

        Nothing calls this yet, because the questions and the answers they
        produce are not in this tree: the reconciled tenancy schema carries no
        onboarding columns (see ``gateway.models.tenancy``). The method is
        here because the seam is where the split belongs whenever they arrive,
        and because the answers are an argument rather than a column read, so
        the port does not wait on the schema.
        """
        ...

    async def record_profile_updated(
        self,
        *,
        background_tasks: BackgroundTasks,
        user_id: uuid.UUID,
        email: str,
        full_name: str | None,
    ) -> None:
        """Notify of an edited display name worth reflecting on the vendor's record.

        Nothing calls this, for the reason ``record_onboarding_completed``
        gives about its own answers: the surface is not in this tree. An
        identity's ``full_name`` is written at signup and by OAuth and never
        edited again, and ``PATCH /v1/users`` edits the ``alias`` on a
        request-plane attribution row, which is a billing label rather than a
        person's display name.
        """
        ...

    async def record_account_deleted(
        self,
        *,
        background_tasks: BackgroundTasks,
        user_id: uuid.UUID,
        email: str,
    ) -> None:
        """Notify that an account was deleted, so a vendor can mark it churned.

        Called with primitives captured before the delete, so an adapter must
        not assume the user row still exists when its background task runs.

        Nothing calls this either: no route here deletes an account.
        ``DELETE /v1/users`` soft-deletes an attribution row and revokes the
        keys on it, which ends a spend identity rather than a person, and the
        one thing that touches an identity itself is
        ``PATCH /v1/admin/users/{user_id}`` deactivating it, which an operator
        can undo and which this port has no counterpart for.
        """
        ...
