"""Live-provider validation for the apron-auth code exchange.

The rest of the suite stubs apron-auth's outbound half, so it proves wiring and
never the exchange itself. A green unit suite is not evidence that the request
shape apron-auth sends is one the provider accepts: a provider's own guide and a
spec-correct OAuth library can disagree on body encoding and client
authentication, and the disagreement only shows up against the real endpoint.
This module is that check, and it is the gate for turning a provider on.

Skipped unless ``OTARI_OAUTH_LIVE_TESTS=1`` and the provider's variables are
set, because an authorization code is single-use and only a person completing a
consent screen can produce one.

**It runs against the gateway's own database**, which the rest of this module's
history did not. The exchange now sends a PKCE ``code_verifier``, and the only
copy of it is the ``oauth_pending_state`` row the ``/authorize`` call wrote, so
the test has to read the row the running gateway left rather than mint anything
of its own. That is also what makes it a real check: a verifier this test
invented would prove the provider accepts *a* verifier, not the one this
deployment would have sent.

To run it for Google::

    # 1. Register an OAuth client with the provider whose redirect URI is
    #    exactly http://localhost:8000/auth/google/callback, then start the
    #    gateway with:
    #      OTARI_PUBLIC_BASE_URL=http://localhost:8000
    #      OTARI_OAUTH_GOOGLE_CLIENT_ID=...
    #      OTARI_OAUTH_GOOGLE_CLIENT_SECRET=...
    #      OTARI_DATABASE_URL=...   (the same one this test will read)
    # 2. Call GET /api/v1/auth/oauth/google/authorize with curl -c so the flow
    #    cookie (otari_oauth_flow) it sets is kept, open the URL it returns and
    #    complete the consent screen. The browser lands on
    #    /#/auth/google/callback?code=...&state=...; copy both out of the
    #    address bar, and the cookie value out of the jar. The state finds the
    #    verifier; the cookie is what the row is bound to.
    # 3. Run immediately, since the code expires in minutes and is single-use,
    #    and the pending state expires in ten.
    OTARI_OAUTH_LIVE_TESTS=1 \\
    OTARI_PUBLIC_BASE_URL=http://localhost:8000 \\
    OTARI_DATABASE_URL=... \\
    OTARI_OAUTH_GOOGLE_CLIENT_ID=... \\
    OTARI_OAUTH_GOOGLE_CLIENT_SECRET=... \\
    OTARI_OAUTH_LIVE_GOOGLE_CODE='4/0Ax...' \\
    OTARI_OAUTH_LIVE_GOOGLE_STATE='...' \\
    OTARI_OAUTH_LIVE_GOOGLE_FLOW_SECRET='...' \\
    uv run pytest tests/integration/test_oauth_live_provider.py -k google -v

GitHub is the same with ``GITHUB`` in place of ``GOOGLE``. The redirect URI is
not a variable of its own: it is derived from ``OTARI_PUBLIC_BASE_URL`` the same
way the running gateway derives it, which is the point. If the value registered
with the provider and the value here disagreed, this test would pass against a
configuration the gateway cannot reproduce.
"""

import os

import pytest

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session, init_db
from gateway.services import oauth_service

pytestmark = pytest.mark.skipif(
    os.environ.get("OTARI_OAUTH_LIVE_TESTS") != "1",
    reason="set OTARI_OAUTH_LIVE_TESTS=1 and a fresh authorization code to run",
)


def _live_code(provider: str) -> str:
    """The operator-supplied authorization code for ``provider``, or skip."""
    code = os.environ.get(f"OTARI_OAUTH_LIVE_{provider.upper()}_CODE", "")
    if not code:
        pytest.skip(f"OTARI_OAUTH_LIVE_{provider.upper()}_CODE is required")
    return code


def _live_state(provider: str) -> str:
    """The ``state`` the provider redirected back with, or skip.

    It is the key to the pending row holding the PKCE verifier, so without it
    there is nothing to exchange with even though the code itself is valid.
    """
    state = os.environ.get(f"OTARI_OAUTH_LIVE_{provider.upper()}_STATE", "")
    if not state:
        pytest.skip(f"OTARI_OAUTH_LIVE_{provider.upper()}_STATE is required")
    return state


def _live_flow_secret(provider: str) -> str:
    """The flow cookie ``/authorize`` set, or skip: the row answers to nothing else."""
    secret = os.environ.get(f"OTARI_OAUTH_LIVE_{provider.upper()}_FLOW_SECRET", "")
    if not secret:
        pytest.skip(f"OTARI_OAUTH_LIVE_{provider.upper()}_FLOW_SECRET is required")
    return secret


def _live_config(provider: str) -> GatewayConfig:
    """The deployment's own configuration, or skip if it does not offer ``provider``."""
    config = GatewayConfig()
    if provider not in config.oauth_providers:
        pytest.skip(
            f"OTARI_PUBLIC_BASE_URL, OTARI_OAUTH_{provider.upper()}_CLIENT_ID and "
            f"OTARI_OAUTH_{provider.upper()}_CLIENT_SECRET are required"
        )
    return config


@pytest.mark.parametrize("provider", ["google", "github"])
@pytest.mark.asyncio
async def test_a_real_authorization_code_exchanges_for_an_identity(provider: str) -> None:
    """The exchange this gateway would perform, against the provider's own endpoint.

    Asserts on the identity rather than the tokens, because the identity is what
    crosses ``IdentityProviderPort`` and therefore what a sign-in depends on: an
    address, and a provider that affirmatively vouches for it. A provider
    returning an address it will not vouch for is refused by the base build's
    adapter, so an exchange that only proved tokens came back would not tell us a
    sign-in could work.
    """
    config = _live_config(provider)
    code = _live_code(provider)
    state = _live_state(provider)
    flow_secret = _live_flow_secret(provider)

    init_db(config)
    async with create_session() as db:
        identity = await oauth_service.exchange_code(
            config, provider, code=code, state=state, flow_secret=flow_secret, db=db
        )
        # Committed, because the state was consumed for real: leaving it
        # spendable would contradict what this module is checking.
        await db.commit()

    assert identity.provider == provider
    assert identity.email, "the provider returned no address, so nothing here could sign in"
    # Not merely truthy: this is the tri-state collapsed onto a bool, and an
    # unasserted value has to arrive as False rather than as something a gate
    # would read as verified.
    assert identity.email_verified is True


@pytest.mark.parametrize("provider", ["google", "github"])
def test_the_authorization_url_this_gateway_builds_is_the_one_that_was_consented_to(
    provider: str,
) -> None:
    """Pin the redirect URI, which is the half a live exchange cannot check for us.

    The exchange above fails with an opaque ``invalid_grant`` when the URI sent
    with it differs from the one the authorization request carried, which is the
    single most common way this flow breaks and the least informative failure it
    produces. Both come from ``public_base_url``, so this asserts the derived
    value out loud: a mismatch then reads as "register this URI with the
    provider" instead of "the code did not work".
    """
    config = _live_config(provider)

    uri = oauth_service.redirect_uri(config, provider)

    assert uri == f"{(config.public_base_url or '').rstrip('/')}/auth/{provider}/callback"
    assert "#" not in uri, "a provider rejects a redirect URI carrying a fragment"
