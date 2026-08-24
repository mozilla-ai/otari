"""Live-provider validation for the apron-auth code exchange.

The rest of the suite stubs apron-auth's outbound half, so it proves wiring and
never the exchange itself. A green unit suite is not evidence that the request
shape apron-auth sends is one the provider accepts: a provider's own guide and a
spec-correct OAuth library can disagree on body encoding and client
authentication, and the disagreement only shows up against the real endpoint.
This module is that check, and it is the gate for turning a provider on.

Skipped unless ``OTARI_OAUTH_LIVE_TESTS=1`` and the provider's variables are
set, because an authorization code is single-use and only a person completing a
consent screen can produce one. Nothing here touches a database, so it needs no
PostgreSQL either.

To run it for Google::

    # 1. Register an OAuth client with the provider whose redirect URI is
    #    exactly http://localhost:8000/auth/google/callback, then start the
    #    gateway with:
    #      OTARI_PUBLIC_BASE_URL=http://localhost:8000
    #      OTARI_OAUTH_GOOGLE_CLIENT_ID=...
    #      OTARI_OAUTH_GOOGLE_CLIENT_SECRET=...
    # 2. Open the URL that GET /v1/auth/oauth/google/authorize returns and
    #    complete the consent screen. The browser lands on
    #    /#/auth/google/callback?code=...; copy that code out of the address bar.
    # 3. Run immediately, since the code expires in minutes and is single-use.
    OTARI_OAUTH_LIVE_TESTS=1 \\
    OTARI_PUBLIC_BASE_URL=http://localhost:8000 \\
    OTARI_OAUTH_GOOGLE_CLIENT_ID=... \\
    OTARI_OAUTH_GOOGLE_CLIENT_SECRET=... \\
    OTARI_OAUTH_LIVE_GOOGLE_CODE='4/0Ax...' \\
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

    identity = await oauth_service.exchange_code(config, provider, code=code)

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
