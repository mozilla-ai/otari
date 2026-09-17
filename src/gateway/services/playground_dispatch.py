"""Running a Playground completion from a deployment that serves no inference.

A hosted control plane holds every tenant's wallet, credentials and budgets and
runs none of their traffic: a completion served here would skip the usage report
that debits the wallet, which is otari#822 and the reason the inference routers
are not mounted. The Playground still belongs on that deployment, because the
page is management traffic everywhere except for the one request that dispatches
a model, so this module is what that one request does instead.

It forwards. The route authenticates the session and resolves the same
:class:`~gateway.types.session_principal.SessionPrincipal` it resolves
standalone, and then, rather than calling the local pipeline, presents that
principal to ``config.data_plane_url`` as an ordinary API key and streams the
answer back. Nothing about routing, budgets, guardrails, tools, pricing or
settlement happens here: the data-plane gateway runs the pipeline and reports the
usage upstream, and that report is what debits the wallet. The billing path is
the one every other request on this deployment already takes.

**Why it has to be a real API key.** The wire between a control plane and its
gateway carries one (``docs/hybrid-mode-protocol.md``): the gateway forwards it
as ``X-User-Token`` and the control plane authenticates it on the way back. There
is no channel on which this deployment can simply assert who to bill, so the
credential has to be something the resolve path already accepts. The key carries
the caller's own user, workspace and model allow-list, so spend, budgets and the
allow-list gate bind exactly as they would for a key of theirs.

**Transport.** This hop carries a credential, so it belongs on a network the
operator controls or behind TLS. ``data_plane_url`` accepts ``http`` for the same
reason ``platform.base_url`` does, and deliberately: the two are one protocol's
two directions between the same pair of processes, and the reverse one carries
the caller's own API key. Holding this direction to https while that one accepts
plaintext would describe a boundary that is not there. otari#1295 settles the
pair together.

**Why it is stored, and why that is not otari-ai#1598 again.** This deployment
has to *present* the credential rather than merely verify one, which is the one
thing a hash cannot do, so the plaintext is encrypted at rest with the same
``secret_box`` that holds provider credentials. What #1598 retired was a bearer
handed to the *browser*; this one never leaves the process, is never returned by
any endpoint, and is excluded from the key listings because it is machinery
rather than a credential anybody manages. One row per caller per workspace,
reused, so a chat page does not mint a key per message.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx
from sqlalchemy import select
from sqlmodel import col

from gateway.auth.models import generate_api_key, hash_key, key_prefix, key_suffix
from gateway.core.config import API_ROOT
from gateway.models.api_keys import APIKey
from gateway.services.secret_box import decrypt_secret, encrypt_secret

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from gateway.types.session_principal import SessionPrincipal

# What the row is called wherever a human does see it, which is the usage list:
# a Playground completion is billed like any other request and its usage row
# names the key that made it.
_KEY_NAME = "Playground"

# The data plane is this deployment's own gateway rather than a provider, so the
# connect budget is short: a slow answer is the model, and an unreachable gateway
# is a misconfiguration worth surfacing quickly. No read timeout, because a
# streamed completion is idle between tokens by design and the client's own
# disconnect is what ends an abandoned one.
_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)

# Hop-by-hop and framing headers belong to this response rather than the upstream
# one: the body is re-framed on the way through, so a Content-Length copied from
# the data plane would describe a different message.
_DROPPED_RESPONSE_HEADERS = frozenset(
    {"content-length", "content-encoding", "transfer-encoding", "connection", "keep-alive"}
)


def completions_url(data_plane_url: str) -> str:
    """Where the data-plane gateway serves chat completions.

    ``data_plane_url`` is validated to carry no API root and no trailing slash
    (``GatewayConfig._validate_data_plane_url``), so the path is appended here the
    way the dashboard appends it for somebody holding a new key.
    """
    return f"{data_plane_url}{API_ROOT}/chat/completions"


async def resolve_dispatch_key(db: AsyncSession, *, principal: SessionPrincipal) -> str:
    """The credential standing for this caller on the data plane, minted once.

    Looked up by the caller and the workspace they are about to spend in, so two
    members of one workspace hold different keys and one member's two workspaces
    do too: the key is what tells the data plane whose budget to bind and whose
    usage to record, and a shared one would merge both.

    The allow-list is refreshed on every use rather than frozen at mint time.
    ``SessionPrincipal`` derives it from the caller's current user default, and a
    key that kept the value it was minted with would let somebody keep reaching a
    model an operator has since taken away from them.
    """
    # The oldest match rather than the only one. Two first messages sent at once,
    # from two tabs, race here and can both insert: nothing in the schema forbids
    # a second row, and a lookup that insisted on exactly one would turn that into
    # a 500 on every later request. Whichever row wins is a complete credential,
    # so the loser is simply never read.
    row = (
        await db.execute(
            select(APIKey)
            .where(
                col(APIKey.user_id) == principal.user_id,
                col(APIKey.workspace_id) == principal.workspace_id,
                col(APIKey.internal_secret).is_not(None),
            )
            .order_by(col(APIKey.created_at), col(APIKey.id))
            .limit(1)
        )
    ).scalars().first()

    if row is None:
        plaintext = generate_api_key()
        row = APIKey(
            id=str(uuid.uuid4()),
            workspace_id=principal.workspace_id,
            key_hash=hash_key(plaintext),
            key_prefix=key_prefix(plaintext),
            key_suffix=key_suffix(plaintext),
            key_name=_KEY_NAME,
            user_id=principal.user_id,
            allowed_models=principal.allowed_models,
            # Playground spend is spend. A budget-exempt key here would make the
            # one page that runs completions the one page that ignores budgets.
            exclude_from_budget=False,
            internal_secret=encrypt_secret(plaintext),
        )
        db.add(row)
    else:
        plaintext = decrypt_secret(row.internal_secret or "")
        row.allowed_models = principal.allowed_models
        # An operator who deactivated this row was deactivating the Playground for
        # nobody: it is not on any screen they could have meant. Reactivated
        # rather than replaced, so the usage already attributed to it stays
        # attributed to it.
        row.is_active = True

    # Committed before the request is forwarded, because the gateway
    # authenticates this key by asking this same deployment: an uncommitted row is
    # one the resolve cannot see.
    await db.commit()
    return plaintext


async def forward_completion(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
) -> tuple[int, dict[str, str], AsyncIterator[bytes]]:
    """Send one completion to the data plane and hand back its answer.

    Returns the upstream status, the headers worth carrying over, and the body as
    a byte stream that closes the connection once it is exhausted. Streaming and
    non-streaming take the same path: the difference is the upstream's own content
    type, which rides along in the headers, so this does not have to know which
    one it is forwarding and a response shape the gateway gains is forwarded the
    day it lands.

    Nothing from the browser's request is carried except the body. In particular
    no dashboard cookie: the data plane authenticates the key above, and must not
    also be handed a credential that means something else here.
    """
    client = httpx.AsyncClient(timeout=_TIMEOUT)
    request = client.build_request(
        "POST",
        url,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "text/event-stream, application/json"},
    )
    try:
        upstream = await client.send(request, stream=True)
    except httpx.HTTPError:
        await client.aclose()
        raise

    headers = {name: value for name, value in upstream.headers.items() if name.lower() not in _DROPPED_RESPONSE_HEADERS}

    async def body() -> AsyncIterator[bytes]:
        try:
            # Decoded rather than raw. The upstream may answer a content encoding
            # this client offered, and ``Content-Encoding`` is dropped above
            # because the body is re-framed here; forwarding the compressed bytes
            # without the header that explains them is a body no browser can read.
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return upstream.status_code, headers, body()
