"""A caller a route authenticated and authorized, in place of an API key."""

import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SessionPrincipal:
    """Who a completion runs as when it was not authorized by a key.

    The completion pipeline's own credential step accepts an API key or the
    deployment's master key and nothing else, deliberately: a dashboard cookie
    honored there would let any signed-in member of any organization spend the
    *default* workspace's credential and bill it, because that is where a
    keyless request resolves (otari-ai#1880, and the comment at the call site).
    That rule is unchanged, and this does not relax it. What it adds is a second
    way in, for a route that has already done the work the rule exists to force:

    - ``user_id`` is derived from the authenticated identity, never accepted
      from the request, so spend binds to the caller and to nobody else.
    - ``workspace_id`` is one the caller was proved a member of
      (``resolve_workspace_in_organization``), so credential resolution stays
      inside their own tenancy rather than falling through to the default
      workspace.
    - ``allowed_models`` is that user's own model default, so the pipeline's
      allow-list gate binds exactly as it would for one of their keys. A session
      holds no key to narrow the default with, so the default is the effective
      list.

    Standalone only, and only ever built by a route that resolved all three;
    ``services/playground_service.resolve_playground_principal`` is the one that
    does. It lives here rather than beside either of them because the pipeline
    consumes it and a service produces it, and a service may not import the API
    layer (``scripts/check_architecture.py``).

    A request carrying one leaves ``api_key_id`` null on its usage row, which is
    accurate: it was made from a session and not with a key, and the row's
    ``user_id`` is what attributes it.
    """

    user_id: str
    workspace_id: uuid.UUID
    allowed_models: list[str] | None
