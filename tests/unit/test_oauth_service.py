"""The OAuth half of dashboard sign-in: configuration, URLs, and the exchange.

Covers what this deployment owns, which is what ``services/oauth_service.py``
kept when the protocol mechanics moved onto apron-auth: which providers are
configured, which scopes are asked for, where the provider is told to send the
browser back to, and the two carry-overs that must survive the port (PKCE stays
off, and a tri-state ``email_verified`` collapses on the unverified side).

The live exchange itself is not here and cannot be: a green suite that stubs
apron-auth proves wiring and never that the request shape it sends is one a
provider accepts. ``tests/integration/test_oauth_live_provider.py`` is that
check, behind an opt-in flag.
"""

from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from apron_auth.providers import github as apron_github
from apron_auth.providers import google as apron_google

from gateway.core.config import OAUTH_PROVIDERS, GatewayConfig
from gateway.services import oauth_service
from gateway.services.tenancy.errors import OAuthExchangeError, OAuthNotConfiguredError


def configured(**overrides: Any) -> GatewayConfig:
    """A deployment with both providers registered and an address of its own."""
    settings: dict[str, Any] = {
        "public_base_url": "https://otari.example.com",
        "oauth_google_client_id": "google-id",
        "oauth_google_client_secret": "google-secret",
        "oauth_github_client_id": "github-id",
        "oauth_github_client_secret": "github-secret",
    }
    return GatewayConfig(**(settings | overrides))


class TestWhichProvidersAreOnOffer:
    def test_a_deployment_that_configured_none_offers_none(self) -> None:
        # The default, and what makes the sign-in screen carry no OAuth
        # affordance out of the box rather than a pair of dead buttons.
        assert GatewayConfig().oauth_providers == ()

    def test_both_halves_of_a_pair_are_needed(self) -> None:
        config = GatewayConfig(
            public_base_url="https://otari.example.com",
            oauth_google_client_id="google-id",
        )
        # An ID with no secret would fail at the provider, so the button is not
        # offered and then refused.
        assert config.oauth_providers == ()

    def test_a_gateway_that_does_not_know_its_own_address_offers_none(self) -> None:
        config = GatewayConfig(
            oauth_google_client_id="google-id",
            oauth_google_client_secret="google-secret",  # noqa: S106
        )
        # The redirect URI is derived from public_base_url, so without one there
        # is no authorization URL to build.
        assert config.oauth_providers == ()

    def test_providers_are_sorted_so_the_sign_in_screen_is_stable(self) -> None:
        assert configured().oauth_providers == ("github", "google")

    def test_one_configured_provider_does_not_offer_the_other(self) -> None:
        config = GatewayConfig(
            public_base_url="https://otari.example.com",
            oauth_github_client_id="github-id",
            oauth_github_client_secret="github-secret",  # noqa: S106
        )
        assert config.oauth_providers == ("github",)

    def test_the_service_and_the_config_name_the_same_providers(self) -> None:
        # Asserted at import as well; restated here so the failure names the
        # rule rather than arriving as a collection error.
        assert set(oauth_service._PROVIDERS) == set(OAUTH_PROVIDERS)


class TestRedirectUri:
    @pytest.mark.parametrize("provider", OAUTH_PROVIDERS)
    def test_carries_no_fragment_so_a_provider_will_accept_it(self, provider: str) -> None:
        # RFC 6749 forbids a fragment in a redirection URI and Google rejects
        # one outright, which is why this is not a dashboard hash path.
        uri = oauth_service.redirect_uri(configured(), provider)

        assert urlsplit(uri).fragment == ""
        assert "#" not in uri

    def test_names_the_provider_so_two_clients_do_not_share_one_uri(self) -> None:
        assert (
            oauth_service.redirect_uri(configured(), "google")
            == "https://otari.example.com/auth/google/callback"
        )
        assert (
            oauth_service.redirect_uri(configured(), "github")
            == "https://otari.example.com/auth/github/callback"
        )

    def test_a_trailing_slash_on_the_base_url_does_not_double_up(self) -> None:
        config = configured(public_base_url="https://otari.example.com/")

        assert (
            oauth_service.redirect_uri(config, "google")
            == "https://otari.example.com/auth/google/callback"
        )


class TestAuthorizationUrl:
    def test_google_asks_for_the_scopes_its_identity_handler_reads_back(self) -> None:
        url = oauth_service.authorization_url(configured(), "google", state="s")
        query = parse_qs(urlsplit(url).query)

        assert urlsplit(url).netloc == "accounts.google.com"
        assert query["scope"] == ["openid email profile"]
        assert query["response_type"] == ["code"]
        assert query["client_id"] == ["google-id"]
        assert query["redirect_uri"] == ["https://otari.example.com/auth/google/callback"]
        assert query["state"] == ["s"]

    def test_github_asks_for_the_scopes_its_identity_handler_reads_back(self) -> None:
        # /user plus /user/emails, which is what makes a verified address
        # available at callback time.
        url = oauth_service.authorization_url(configured(), "github", state="s")
        query = parse_qs(urlsplit(url).query)

        assert urlsplit(url).netloc == "github.com"
        assert query["scope"] == ["read:user user:email"]
        assert query["client_id"] == ["github-id"]

    @pytest.mark.parametrize("provider", OAUTH_PROVIDERS)
    def test_no_code_challenge_is_sent(self, provider: str) -> None:
        # PKCE is deliberately off: authorize and callback are independent
        # requests with no store between them, so a verifier minted here would
        # have nowhere to live until the exchange. A challenge this flow cannot
        # answer would break every sign-in.
        query = parse_qs(urlsplit(oauth_service.authorization_url(configured(), provider, state="s")).query)

        assert "code_challenge" not in query
        assert "code_challenge_method" not in query

    @pytest.mark.parametrize("provider", OAUTH_PROVIDERS)
    def test_an_unconfigured_provider_refuses_and_names_the_settings(self, provider: str) -> None:
        with pytest.raises(OAuthNotConfiguredError) as caught:
            oauth_service.authorization_url(GatewayConfig(), provider, state="s")

        assert caught.value.status_code == 503
        assert f"oauth_{provider}_client_id" in caught.value.message
        assert "public_base_url" in caught.value.message

    def test_a_provider_this_build_never_named_is_refused(self) -> None:
        with pytest.raises(OAuthNotConfiguredError):
            oauth_service.authorization_url(configured(), "not-a-provider", state="s")


class TestState:
    def test_is_unguessable_and_fresh_each_time(self) -> None:
        values = {oauth_service.new_state() for _ in range(50)}

        assert len(values) == 50
        assert all(len(value) >= 32 for value in values)


class TestPkceStaysOff:
    @pytest.mark.parametrize("provider", OAUTH_PROVIDERS)
    def test_the_preset_would_have_asked_for_pkce(self, provider: str) -> None:
        # The half of the guard that makes the other half meaningful. If a
        # preset ever shipped with use_pkce already false, the assertion below
        # would pass while proving nothing, and a later apron-auth release
        # flipping the default back would go unnoticed.
        preset = apron_google.preset if provider == "google" else apron_github.preset
        provider_config, _ = preset(
            client_id="id",
            client_secret="secret",  # noqa: S106
            scopes=["openid"],
        )

        assert provider_config.use_pkce is True

    @pytest.mark.parametrize("provider", OAUTH_PROVIDERS)
    def test_this_flow_clears_it(self, provider: str) -> None:
        # apron-auth reads use_pkce only inside get_authorization_url, which
        # this module does not call, so clearing it changes nothing today. It is
        # the guard that keeps adopting that method later from silently starting
        # to send a challenge this flow cannot answer; see the module docstring.
        preset = apron_google.preset if provider == "google" else apron_github.preset
        provider_config, _ = preset(
            client_id="id",
            client_secret="secret",  # noqa: S106
            scopes=["openid"],
        )

        assert oauth_service._without_pkce(provider_config).use_pkce is False


class TestExchange:
    """The exchange with apron-auth's network calls stubbed out."""

    @staticmethod
    def _stub_client(monkeypatch: pytest.MonkeyPatch, profile: Any) -> None:
        class _Client:
            async def exchange_code(self, **_: Any) -> object:
                return object()

            async def fetch_identity(self, _tokens: object) -> Any:
                return profile

        monkeypatch.setattr(oauth_service, "_client", lambda *_args, **_kwargs: _Client())

    @staticmethod
    def _profile(**overrides: Any) -> SimpleNamespace:
        fields: dict[str, Any] = {
            "email": "member@example.com",
            "name": "A Member",
            "email_verified": True,
        }
        return SimpleNamespace(**(fields | overrides))

    @pytest.mark.asyncio
    async def test_returns_the_identity_the_provider_vouches_for(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_client(monkeypatch, self._profile())

        identity = await oauth_service.exchange_code(configured(), "google", code="c")

        assert identity.provider == "google"
        assert identity.email == "member@example.com"
        assert identity.full_name == "A Member"
        assert identity.email_verified is True

    @pytest.mark.asyncio
    async def test_an_unasserted_email_verified_collapses_to_unverified(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # apron-auth reports email_verified as tri-state. This edition resolves
        # on a bool, and silence is not an assertion: it must not be laundered
        # into a verified identity. otari-ai#1551 moves resolution onto the
        # tri-state model, once, on the platform.
        self._stub_client(monkeypatch, self._profile(email_verified=None))

        identity = await oauth_service.exchange_code(configured(), "google", code="c")

        assert identity.email_verified is False

    @pytest.mark.asyncio
    async def test_an_explicit_false_is_unverified_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_client(monkeypatch, self._profile(email_verified=False))

        identity = await oauth_service.exchange_code(configured(), "google", code="c")

        assert identity.email_verified is False

    @pytest.mark.asyncio
    async def test_a_failed_exchange_does_not_carry_the_providers_words_to_the_caller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # apron-auth's exchange errors carry the provider's RFC 6749 error and
        # error_description verbatim, and this message reaches both the client
        # and the log aggregator (CWE-532). The cause stays on the traceback.
        secret = "invalid_grant: code was already redeemed by client 1234"  # noqa: S105

        class _Client:
            async def exchange_code(self, **_: Any) -> object:
                raise RuntimeError(secret)

            async def fetch_identity(self, _tokens: object) -> Any:  # pragma: no cover - never reached
                raise AssertionError

        monkeypatch.setattr(oauth_service, "_client", lambda *_a, **_k: _Client())

        with pytest.raises(OAuthExchangeError) as caught:
            await oauth_service.exchange_code(configured(), "google", code="c")

        assert secret not in caught.value.message
        assert caught.value.message == "Google did not complete the sign-in. Try again."
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert secret in str(caught.value.__cause__)

    @pytest.mark.asyncio
    async def test_a_failed_identity_fetch_is_the_same_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Client:
            async def exchange_code(self, **_: Any) -> object:
                return object()

            async def fetch_identity(self, _tokens: object) -> Any:
                raise RuntimeError("userinfo 500")

        monkeypatch.setattr(oauth_service, "_client", lambda *_a, **_k: _Client())

        with pytest.raises(OAuthExchangeError):
            await oauth_service.exchange_code(configured(), "github", code="c")

    @pytest.mark.asyncio
    async def test_an_unconfigured_provider_refuses_before_any_outbound_call(self) -> None:
        with pytest.raises(OAuthNotConfiguredError):
            await oauth_service.exchange_code(GatewayConfig(), "google", code="c")


class TestProviderLabel:
    def test_writes_each_provider_the_way_it_writes_itself(self) -> None:
        assert oauth_service.provider_label("google") == "Google"
        assert oauth_service.provider_label("github") == "GitHub"

    def test_falls_back_rather_than_raising_inside_an_error_message(self) -> None:
        # The only caller is a refusal, and one that fails to render is worse
        # than one naming a provider nobody configured.
        assert oauth_service.provider_label("acme-oidc") == "acme-oidc"
