"""Workload identity: verifying an OpenID Connect identity token and matching it to a key's federation rule.

Keys are generated here and the issuer's documents are served by a stub, so
nothing reaches the network.
"""

import base64
import json
import time
from collections.abc import Callable
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jwt.algorithms import ECAlgorithm, RSAAlgorithm
from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.core.settings.api_keys import FederationIssuer
from gateway.models.api_keys import APIKeyFederationRule
from gateway.schemas.api_keys import FederationRuleCreate
from gateway.services.api_keys import (
    FederatedIdentity,
    FederatedTokenVerifier,
    FederationRefusedError,
    FederationUnavailableError,
    match_federation_rules,
    unverified_issuer,
)
from gateway.services.api_keys._federation import (
    JWKS_REFETCH_INTERVAL_SECONDS,
    JWKS_RETRY_INTERVAL_SECONDS,
    JWKS_TTL_SECONDS,
)
from gateway.services.url_safety import UnsafeURLError, validate_federation_url

ISSUER = "https://issuer.example.com"
JWKS_URI = "https://issuer.example.com/keys"
DISCOVERY_URI = f"{ISSUER}/.well-known/openid-configuration"
AUDIENCE = "https://otari.example.com"
SUBJECT = "113051416744238958061"
EMAIL = "hackbot-api@project.iam.gserviceaccount.com"


def _rsa(bits: int = 2048) -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=bits)


SIGNING_KEY = _rsa()
OTHER_KEY = _rsa()


def _jwk(private_key: Any, kid: str, alg: str = "RS256") -> dict[str, Any]:
    algorithm = ECAlgorithm if isinstance(private_key, ec.EllipticCurvePrivateKey) else RSAAlgorithm
    jwk: dict[str, Any] = algorithm.to_jwk(private_key.public_key(), as_dict=True)
    return {**jwk, "kid": kid, "alg": alg, "use": "sig"}


def _claims(**overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": SUBJECT,
        "email": EMAIL,
        "email_verified": True,
        "iat": now,
        "exp": now + 3600,
    }
    claims.update(overrides)
    return {name: value for name, value in claims.items() if value is not None}


def _token(
    private_key: Any = SIGNING_KEY,
    *,
    kid: str = "k1",
    algorithm: str = "RS256",
    drop: tuple[str, ...] = (),
    **claims: Any,
) -> str:
    payload = {name: value for name, value in _claims(**claims).items() if name not in drop}
    return jwt.encode(payload, private_key, algorithm=algorithm, headers={"kid": kid})


def _b64(data: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


def _unsigned(header: dict[str, Any], **claims: Any) -> str:
    return f"{_b64(header)}.{_b64(_claims(**claims))}."


class _Issuer:
    """Serves an issuer's discovery document and JWKS, counting fetches and failing on demand."""

    def __init__(self, keys: list[dict[str, Any]], *, discovery_issuer: str = ISSUER) -> None:
        self.keys = keys
        self.discovery_issuer = discovery_issuer
        self.fetches: list[str] = []
        self.failing = False

    async def __call__(self, url: str) -> Any:
        from gateway.services.api_keys._federation import _FetchError

        self.fetches.append(url)
        if self.failing:
            raise _FetchError
        if url == DISCOVERY_URI:
            return {"issuer": self.discovery_issuer, "jwks_uri": JWKS_URI}
        if url == JWKS_URI:
            return {"keys": self.keys}
        raise _FetchError


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _verifier(
    issuer: _Issuer, *, jwks_uri: str | None = JWKS_URI, clock: Callable[[], float] = time.monotonic
) -> FederatedTokenVerifier:
    return FederatedTokenVerifier([FederationIssuer(issuer=ISSUER, jwks_uri=jwks_uri)], fetch=issuer, clock=clock)


async def _refusal(verifier: FederatedTokenVerifier, token: str) -> str:
    with pytest.raises(FederationRefusedError) as refused:
        await verifier.verify(token)
    return refused.value.reason


# Verification


@pytest.mark.asyncio
async def test_a_valid_token_yields_its_identity() -> None:
    verifier = _verifier(_Issuer([_jwk(SIGNING_KEY, "k1")]))

    identity = await verifier.verify(_token(email=EMAIL.upper(), aud=[AUDIENCE, "other"]))

    assert identity == FederatedIdentity(
        issuer=ISSUER, audiences=(AUDIENCE, "other"), subject=SUBJECT, email=EMAIL, expires_at=identity.expires_at
    )
    assert identity.expires_at > time.time()


@pytest.mark.asyncio
async def test_the_jwks_location_is_discovered_from_the_issuer() -> None:
    issuer = _Issuer([_jwk(SIGNING_KEY, "k1")])
    verifier = _verifier(issuer, jwks_uri=None)

    await verifier.verify(_token())

    assert issuer.fetches == [DISCOVERY_URI, JWKS_URI]


@pytest.mark.asyncio
async def test_a_discovery_document_for_another_issuer_is_refused() -> None:
    verifier = _verifier(_Issuer([_jwk(SIGNING_KEY, "k1")], discovery_issuer="https://evil.example.com"), jwks_uri=None)

    with pytest.raises(FederationUnavailableError):
        await verifier.verify(_token())


@pytest.mark.asyncio
async def test_an_issuer_not_on_the_allow_list_is_refused_without_a_fetch() -> None:
    issuer = _Issuer([_jwk(SIGNING_KEY, "k1")])
    verifier = _verifier(issuer)

    assert await _refusal(verifier, _token(iss="https://accounts.evil.example.com")) == "federation_unknown_issuer"
    assert issuer.fetches == []


@pytest.mark.asyncio
async def test_a_token_signed_by_another_key_is_refused() -> None:
    verifier = _verifier(_Issuer([_jwk(SIGNING_KEY, "k1")]))

    assert await _refusal(verifier, _token(OTHER_KEY)) == "federation_invalid_token"


@pytest.mark.asyncio
async def test_a_tampered_payload_is_refused() -> None:
    verifier = _verifier(_Issuer([_jwk(SIGNING_KEY, "k1")]))
    header, _, signature = _token().split(".")
    forged = f"{header}.{_b64(_claims(sub='someone-else'))}.{signature}"

    assert await _refusal(verifier, forged) == "federation_invalid_token"


@pytest.mark.asyncio
async def test_alg_none_is_refused() -> None:
    issuer = _Issuer([_jwk(SIGNING_KEY, "k1")])
    verifier = _verifier(issuer)

    assert await _refusal(verifier, _unsigned({"alg": "none", "kid": "k1", "typ": "JWT"})) == "federation_bad_algorithm"
    assert issuer.fetches == []


@pytest.mark.asyncio
async def test_hs256_signed_with_the_public_key_is_refused() -> None:
    """The classic key-confusion attack: HMAC keyed with the issuer's public key, which anyone can read."""
    verifier = _verifier(_Issuer([_jwk(SIGNING_KEY, "k1")]))
    public_pem = SIGNING_KEY.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    header = _b64({"alg": "HS256", "kid": "k1", "typ": "JWT"})
    payload = _b64(_claims())
    signature = jwt.get_algorithm_by_name("HS256").sign(f"{header}.{payload}".encode(), public_pem)
    token = f"{header}.{payload}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"

    assert await _refusal(verifier, token) == "federation_bad_algorithm"


@pytest.mark.asyncio
async def test_a_token_cannot_choose_another_algorithm_than_its_key() -> None:
    verifier = _verifier(_Issuer([_jwk(SIGNING_KEY, "k1", alg="RS256")]))

    assert await _refusal(verifier, _token(algorithm="PS256")) == "federation_invalid_token"


@pytest.mark.asyncio
async def test_symmetric_keys_in_the_jwks_are_never_used() -> None:
    oct_key = {"kty": "oct", "k": _b64({"x": 1}), "kid": "k1", "alg": "HS256"}
    verifier = _verifier(_Issuer([oct_key]))

    with pytest.raises(FederationUnavailableError):
        await verifier.verify(_token())


@pytest.mark.asyncio
async def test_an_rsa_key_shorter_than_2048_bits_is_refused() -> None:
    weak = _rsa(1024)
    verifier = _verifier(_Issuer([_jwk(weak, "k1")]))

    assert await _refusal(verifier, _token(weak)) == "federation_invalid_token"


@pytest.mark.asyncio
async def test_an_ec_key_verifies() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    verifier = _verifier(_Issuer([_jwk(key, "e1", alg="ES256")]))

    identity = await verifier.verify(_token(key, kid="e1", algorithm="ES256"))

    assert identity.subject == SUBJECT


@pytest.mark.asyncio
async def test_an_expired_token_is_refused() -> None:
    verifier = _verifier(_Issuer([_jwk(SIGNING_KEY, "k1")]))
    now = int(time.time())

    assert await _refusal(verifier, _token(iat=now - 7200, exp=now - 120)) == "federation_expired"


@pytest.mark.asyncio
async def test_clock_skew_is_tolerated_up_to_a_minute() -> None:
    verifier = _verifier(_Issuer([_jwk(SIGNING_KEY, "k1")]))
    now = int(time.time())

    await verifier.verify(_token(iat=now - 3600, exp=now - 30))
    await verifier.verify(_token(iat=now + 30, nbf=now + 30))


@pytest.mark.asyncio
@pytest.mark.parametrize("claim", ["nbf", "iat"])
async def test_a_token_from_the_future_is_refused(claim: str) -> None:
    verifier = _verifier(_Issuer([_jwk(SIGNING_KEY, "k1")]))
    now = int(time.time())

    claims: dict[str, Any] = {claim: now + 600}

    assert await _refusal(verifier, _token(**claims)) == "federation_invalid_token"


@pytest.mark.asyncio
@pytest.mark.parametrize("claim", ["sub", "aud", "exp", "iat"])
async def test_a_token_missing_a_required_claim_is_refused(claim: str) -> None:
    verifier = _verifier(_Issuer([_jwk(SIGNING_KEY, "k1")]))

    assert await _refusal(verifier, _token(drop=(claim,))) == "federation_invalid_token"


@pytest.mark.asyncio
async def test_a_token_without_a_key_id_is_refused() -> None:
    verifier = _verifier(_Issuer([_jwk(SIGNING_KEY, "k1")]))
    token = jwt.encode(_claims(), SIGNING_KEY, algorithm="RS256")

    assert await _refusal(verifier, token) == "federation_invalid_token"


@pytest.mark.asyncio
@pytest.mark.parametrize("verified", [False, "true", None])
async def test_an_email_not_marked_verified_is_dropped(verified: object) -> None:
    verifier = _verifier(_Issuer([_jwk(SIGNING_KEY, "k1")]))

    identity = await verifier.verify(_token(email_verified=verified))

    assert identity.email is None


# The key cache


@pytest.mark.asyncio
async def test_keys_are_cached_within_their_lifetime() -> None:
    issuer = _Issuer([_jwk(SIGNING_KEY, "k1")])
    clock = _Clock()
    verifier = _verifier(issuer, clock=clock)

    await verifier.verify(_token())
    clock.now += JWKS_TTL_SECONDS - 1
    await verifier.verify(_token())

    assert issuer.fetches == [JWKS_URI]


@pytest.mark.asyncio
async def test_expired_keys_are_refetched_and_never_used_when_the_fetch_fails() -> None:
    issuer = _Issuer([_jwk(SIGNING_KEY, "k1")])
    clock = _Clock()
    verifier = _verifier(issuer, clock=clock)
    await verifier.verify(_token())

    clock.now += JWKS_TTL_SECONDS
    issuer.failing = True

    with pytest.raises(FederationUnavailableError):
        await verifier.verify(_token())
    assert issuer.fetches == [JWKS_URI, JWKS_URI]


@pytest.mark.asyncio
async def test_a_failed_fetch_is_not_retried_on_every_request() -> None:
    issuer = _Issuer([_jwk(SIGNING_KEY, "k1")])
    issuer.failing = True
    clock = _Clock()
    verifier = _verifier(issuer, clock=clock)

    for _ in range(3):
        with pytest.raises(FederationUnavailableError):
            await verifier.verify(_token())
    assert len(issuer.fetches) == 1

    issuer.failing = False
    clock.now += JWKS_RETRY_INTERVAL_SECONDS
    await verifier.verify(_token())
    assert len(issuer.fetches) == 2


@pytest.mark.asyncio
async def test_an_unknown_key_id_refetches_at_most_once_a_minute() -> None:
    issuer = _Issuer([_jwk(SIGNING_KEY, "k1")])
    clock = _Clock()
    verifier = _verifier(issuer, clock=clock)
    await verifier.verify(_token())

    clock.now += 1
    for kid in ("made-up-1", "made-up-2", "made-up-3"):
        assert await _refusal(verifier, _token(kid=kid)) == "federation_unknown_key"
    assert issuer.fetches == [JWKS_URI]


@pytest.mark.asyncio
async def test_a_rotated_key_is_picked_up() -> None:
    issuer = _Issuer([_jwk(SIGNING_KEY, "k1")])
    clock = _Clock()
    verifier = _verifier(issuer, clock=clock)
    await verifier.verify(_token())

    issuer.keys = [_jwk(SIGNING_KEY, "k1"), _jwk(OTHER_KEY, "k2")]
    clock.now += JWKS_REFETCH_INTERVAL_SECONDS
    identity = await verifier.verify(_token(OTHER_KEY, kid="k2"))

    assert identity.subject == SUBJECT
    assert issuer.fetches == [JWKS_URI, JWKS_URI]


@pytest.mark.asyncio
async def test_cached_keys_survive_a_failed_refetch_for_an_unknown_key_id() -> None:
    issuer = _Issuer([_jwk(SIGNING_KEY, "k1")])
    clock = _Clock()
    verifier = _verifier(issuer, clock=clock)
    await verifier.verify(_token())

    issuer.failing = True
    clock.now += JWKS_REFETCH_INTERVAL_SECONDS
    assert await _refusal(verifier, _token(kid="k2")) == "federation_unknown_key"

    await verifier.verify(_token())


# Recognizing a token


def test_unverified_issuer_reads_a_jwt() -> None:
    assert unverified_issuer(_token()) == ISSUER


@pytest.mark.parametrize(
    "credential",
    [
        "tk-" + "a" * 64,
        "tk_live.abcdef",
        "a.b.c",
        f"{_b64({'alg': 'RS256'})}.{_b64({'iss': 1})}.sig",
        f"{_b64({'typ': 'JWT'})}.{_b64({'iss': ISSUER})}.sig",
        "x" * 20000,
    ],
)
def test_unverified_issuer_ignores_anything_else(credential: str) -> None:
    assert unverified_issuer(credential) is None


# Matching rules


def _rule(
    rule_id: str = "r1",
    *,
    key: str = "key-1",
    audience: str = AUDIENCE,
    subject: str | None = None,
    email: str | None = EMAIL,
    issuer: str = ISSUER,
) -> APIKeyFederationRule:
    return APIKeyFederationRule(
        id=rule_id, api_key_id=key, issuer=issuer, audience=audience, subject=subject, email=email
    )


EXP = 2_000_000_000
IDENTITY = FederatedIdentity(issuer=ISSUER, audiences=(AUDIENCE,), subject=SUBJECT, email=EMAIL, expires_at=EXP)


def test_a_rule_naming_the_email_matches() -> None:
    assert match_federation_rules([_rule()], IDENTITY).api_key_id == "key-1"


def test_a_rule_naming_subject_and_email_needs_both() -> None:
    rule = _rule(subject=SUBJECT)

    assert match_federation_rules([rule], IDENTITY).api_key_id == "key-1"
    with pytest.raises(FederationRefusedError, match="federation_no_rule"):
        match_federation_rules([rule], FederatedIdentity(ISSUER, (AUDIENCE,), "another-subject", EMAIL, EXP))


@pytest.mark.parametrize(
    "identity",
    [
        FederatedIdentity(ISSUER, (AUDIENCE,), SUBJECT, None, EXP),
        FederatedIdentity(ISSUER, (AUDIENCE,), SUBJECT, "someone@project.iam.gserviceaccount.com", EXP),
        FederatedIdentity(ISSUER, ("https://elsewhere.example.com",), SUBJECT, EMAIL, EXP),
        FederatedIdentity("https://other-issuer.example.com", (AUDIENCE,), SUBJECT, EMAIL, EXP),
    ],
    ids=["unverified-email", "other-email", "other-audience", "other-issuer"],
)
def test_a_mismatch_matches_nothing(identity: FederatedIdentity) -> None:
    with pytest.raises(FederationRefusedError, match="federation_no_rule"):
        match_federation_rules([_rule()], identity)


def test_a_subject_rule_does_not_match_another_subject() -> None:
    with pytest.raises(FederationRefusedError, match="federation_no_rule"):
        match_federation_rules(
            [_rule(subject="another", email=None)], FederatedIdentity(ISSUER, (AUDIENCE,), SUBJECT, EMAIL, EXP)
        )


def test_a_rule_naming_neither_subject_nor_email_never_matches() -> None:
    with pytest.raises(FederationRefusedError, match="federation_no_rule"):
        match_federation_rules([_rule(subject=None, email=None)], IDENTITY)


def test_rules_of_two_keys_matching_one_token_authenticate_neither() -> None:
    with pytest.raises(FederationRefusedError, match="federation_ambiguous_rule"):
        match_federation_rules([_rule("r1", key="key-1"), _rule("r2", key="key-2", subject=SUBJECT)], IDENTITY)


def test_several_rules_of_one_key_name_that_key() -> None:
    match = match_federation_rules([_rule("r2", subject=SUBJECT, email=None), _rule("r1")], IDENTITY)

    assert (match.api_key_id, match.rule_ids) == ("key-1", ("r1", "r2"))


# Settings and request validation


def test_federation_is_off_by_default() -> None:
    assert GatewayConfig().federation_issuers == []


def test_an_issuer_may_be_written_as_its_url() -> None:
    config = GatewayConfig.model_validate({"federation_issuers": ["https://accounts.google.com"]})

    assert config.federation_issuers == [FederationIssuer(issuer="https://accounts.google.com")]


@pytest.mark.parametrize(
    "entry",
    [
        "http://accounts.google.com",
        "accounts.google.com",
        "https://user:pass@issuer.example.com",
        "https://issuer.example.com?x=1",
        {"issuer": ISSUER, "jwks_uri": "http://issuer.example.com/keys"},
        {"issuer": ISSUER, "unknown": True},
    ],
)
def test_an_issuer_that_is_not_a_plain_https_url_is_refused(entry: object) -> None:
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate({"federation_issuers": [entry]})


def test_an_issuer_listed_twice_is_refused() -> None:
    with pytest.raises(ValidationError, match="more than once"):
        GatewayConfig.model_validate({"federation_issuers": [ISSUER, {"issuer": ISSUER, "jwks_uri": JWKS_URI}]})


def test_a_hybrid_gateway_refuses_federation() -> None:
    config = GatewayConfig.model_validate({"mode": "hybrid", "federation_issuers": [ISSUER]})

    with pytest.raises(ValueError, match="federation_issuers"):
        config.validate_declared_access()


def test_a_rule_must_name_a_subject_or_an_email() -> None:
    with pytest.raises(ValidationError, match="names a subject, an email, or both"):
        FederationRuleCreate(issuer=ISSUER, audience=AUDIENCE)


def test_a_rule_email_is_stored_lower_case() -> None:
    assert FederationRuleCreate(issuer=ISSUER, audience=AUDIENCE, email=" Bot@Example.COM ").email == "bot@example.com"


# Outbound URL checks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    ["http://issuer.example.com/keys", "https://10.0.0.1/keys", "https://169.254.169.254/keys", "https://127.0.0.1/k"],
)
async def test_a_jwks_url_must_be_public_https(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        await validate_federation_url(url)


@pytest.mark.asyncio
async def test_a_private_issuer_may_be_allowed_but_never_over_http() -> None:
    await validate_federation_url("https://10.0.0.1/keys", allow_private_hosts=True)
    with pytest.raises(UnsafeURLError):
        await validate_federation_url("http://10.0.0.1/keys", allow_private_hosts=True)


@pytest.mark.asyncio
async def test_the_verifier_fetches_with_the_configured_private_host_setting() -> None:
    """The default fetcher refuses a private JWKS address unless the verifier was built allowing it."""
    issuer = [FederationIssuer(issuer=ISSUER, jwks_uri="https://10.0.0.1/keys")]

    with pytest.raises(FederationUnavailableError):
        await FederatedTokenVerifier(issuer).verify(_token())
    assert FederatedTokenVerifier(issuer, allow_private_hosts=True).allow_private_hosts is True


def test_the_private_host_setting_is_validated_at_config_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_FEDERATION_ALLOW_PRIVATE_HOSTS", "sometimes")
    with pytest.raises(ValidationError, match="federation_allow_private_hosts"):
        GatewayConfig()

    monkeypatch.setenv("OTARI_FEDERATION_ALLOW_PRIVATE_HOSTS", "true")
    assert GatewayConfig().federation_allow_private_hosts is True


# Starting without a secret


def test_federation_without_a_secret_key_refuses_to_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from gateway.main import create_app

    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)
    config = GatewayConfig.model_validate(
        {"federation_issuers": [ISSUER], "database_url": f"sqlite:///{tmp_path / 'otari.db'}"}
    )

    with pytest.raises(ValueError, match="OTARI_SECRET_KEY"):
        config.validate_declared_access()
    with pytest.raises(ValueError, match="OTARI_SECRET_KEY"):
        create_app(config)


def test_federation_with_a_secret_key_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.services.secret_box import generate_secret_key

    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    GatewayConfig.model_validate({"federation_issuers": [ISSUER]}).validate_declared_access()


def test_no_secret_key_is_needed_with_federation_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)

    GatewayConfig().validate_declared_access()
