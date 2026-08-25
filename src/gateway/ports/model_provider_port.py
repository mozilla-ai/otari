"""Deployment-owned inference credentials for a request that brings none.

The seam between the core and whichever build serves a request that carries no
BYO provider key. A caller's own stored key never reaches this port: resolving
one is identical in every build, so it stays a plain, unseamed lookup upstream
of here. Only what happens when there is no BYO key varies, and hosted
inference (a metered fleet the deployment owns and proxies to) is the "core
port + hosted adapter" row of ``ARCHITECTURE.md``'s capability lines.

Self-hosting is a first-class path *upstream* of this port, not behind it: a
deployment pointing at its own backends resolves a credential the ordinary way
and never asks here. So the core adapter answers "unavailable" for every
candidate by design rather than by omission, and an overlay binds an adapter
that resolves against the upstreams it owns.

Stability: this interface is not frozen while Otari is pre-1.0. Overlay authors
should pin a released tag and expect the shape to move.
"""

import uuid
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class HostedCredential:
    """A resolved, deployment-owned connection that can serve one candidate.

    ``response_provider`` is the upstream name usage and telemetry key on, which
    may differ from the public managed name the caller asked for, matching
    ``provider`` on a resolve attempt (``docs/hybrid-mode-protocol.md``).

    Returned only when the credential is actually usable. A build that cannot
    serve a candidate (its own credential for it is absent, disabled, or
    undecryptable) answers ``None`` from
    :meth:`ModelProviderPort.resolve_hosted_credential` rather than a value
    carrying an empty key.
    """

    api_key: str
    api_base: str | None
    response_provider: str


class HostedAccessDeniedError(Exception):
    """Raised when an organization may not use the upstream that would serve a candidate.

    The port owns this error so a caller can handle a refusal without naming
    the adapter that produced it. An adapter may raise a subclass carrying its
    own wording. ``workspace_id`` is attribution only, never part of the access
    decision (which keys on the organization alone), and is ``None`` for a
    caller with no workspace context.
    """

    def __init__(self, message: str, workspace_id: uuid.UUID | None = None) -> None:
        super().__init__(message)
        self.workspace_id = workspace_id


class ModelProviderPort(Protocol):
    """What a build must answer to serve a candidate that brings no BYO key."""

    async def resolve_hosted_credential(
        self,
        *,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID | None,
        provider: str,
        model: str | None,
    ) -> HostedCredential | None:
        """Resolve a deployment-owned credential to serve ``provider``.

        ``model`` narrows which upstream serves the candidate (by its
        per-model catalog entry, say); ``None`` means no specific model, which
        resolves to the provider's default upstream. ``workspace_id`` is
        ``None`` for a caller with no workspace context; it is attribution
        only, never part of the access decision, which keys on
        ``organization_id`` alone.

        Returns:
            ``None`` when this build has no hosted-inference path for this
            candidate at all: the core answer for every candidate, and an
            overlay's honest answer when its own credential for the candidate
            is absent, disabled, or undecryptable.

        Raises:
            HostedAccessDeniedError: If the organization may not use the
                upstream that would serve ``model``.

        """
        ...
