"""The wire half of signing a user in to an MCP server, per the MCP authorization specification.

Discovery of the protected resource (RFC 9728) and of its authorization server (RFC 8414, with
the OpenID Connect fallbacks), dynamic client registration (RFC 7591), the authorization request
with S256 PKCE (RFC 7636) and the resource indicator (RFC 8707), the code exchange, the refresh
(RFC 6749 section 6) and revocation (RFC 7009).

Nothing here reads or writes a row. Every URL is the remote server's or its authorization server's
word, so each one passes :func:`gateway.services.url_safety.validate_mcp_url` before anything is
sent to it, as a credential-carrying URL: https only. Redirects are never followed and a response
is read up to a fixed size. Nothing a remote party sent is logged or carried in an error message
beyond a status and a short, charset-checked OAuth error code.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import math
import re
import secrets
from dataclasses import dataclass
from email.utils import unquote
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import parse_http_list

import httpx
from mcp.types import LATEST_PROTOCOL_VERSION

from gateway.log_config import logger
from gateway.services.url_safety import UnsafeURLError, validate_mcp_url

_DISCOVERY_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_TOKEN_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_MAX_RESPONSE_BYTES = 1024 * 1024
_CLIENT_NAME = "Otari"
# Above any real lifetime and any real epoch second this century, below what a datetime overflows on.
_MAX_SECONDS = 2**32

AUTH_METHOD_NONE = "none"
AUTH_METHOD_POST = "client_secret_post"
AUTH_METHOD_BASIC = "client_secret_basic"


class McpOAuthError(Exception):
    """A sign-in step failed. The message is safe to show: it never repeats what a remote party sent."""


class McpOAuthDiscoveryError(McpOAuthError):
    """The server's authorization setup could not be discovered, or is not one the gateway can use."""


class McpOAuthPkceUnsupportedError(McpOAuthDiscoveryError):
    """The authorization server does not say it supports S256 PKCE: a setting a manager has to see."""


class McpOAuthRegistrationError(McpOAuthDiscoveryError):
    """The authorization server registers no client for the gateway, so a manager has to supply one."""


class McpOAuthTokenError(McpOAuthError):
    """The token endpoint refused a code exchange or a refresh."""

    def __init__(self, message: str, *, status_code: int | None = None, error_code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code

    @property
    def is_permanent(self) -> bool:
        """Whether the grant itself is gone, so sending the same refresh token again cannot succeed.

        Only ``invalid_grant`` says so (RFC 6749 section 5.2). A refused client, a 5xx, a 429, an
        unclassified 400 or a malformed answer may yet succeed with the same token.
        """
        return self.error_code == "invalid_grant"

    @property
    def refuses_client(self) -> bool:
        """Whether the authorization server refused the client and not the grant.

        ``invalid_client`` or ``unauthorized_client``. A bare 401 is not read as one: a proxy in
        front of the token endpoint answers the same way.
        """
        return self.error_code in ("invalid_client", "unauthorized_client")


_DIGITS = re.compile(r"[0-9]{1,64}")
_OAUTH_ERROR_CODE = re.compile(r"[\x20-\x21\x23-\x5b\x5d-\x7e]{1,64}")


@dataclass(frozen=True, slots=True)
class McpOAuthMetadata:
    """What discovery found: where a user signs in, and where the gateway trades a code for tokens."""

    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None = None
    revocation_endpoint: str | None = None
    # The scope to request, chosen by the specification's order; None sends no scope parameter.
    scope: str | None = None
    # The RFC 8707 resource indicator, set when the server published protected-resource metadata.
    resource: str | None = None
    token_auth_methods: tuple[str, ...] = ()
    # The issuer the metadata was fetched for and names itself as, which a callback's ``iss`` must match.
    issuer: str | None = None
    # RFC 9207: the authorization server sends ``iss`` with every authorization response.
    sends_iss: bool = False


@dataclass(frozen=True, slots=True)
class McpOAuthClient:
    """The client the gateway presents itself as to one authorization server."""

    client_id: str
    client_secret: str | None = None
    auth_method: str = AUTH_METHOD_NONE
    # When a registered client's secret stops working, as an epoch second; None when it does not expire.
    secret_expires_at: int | None = None


@dataclass(frozen=True, slots=True)
class McpOAuthTokens:
    """A token endpoint's answer."""

    access_token: str
    refresh_token: str | None = None
    expires_in: int | None = None
    scope: str | None = None


def _http_client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    """The client every request here goes out on; a test swaps this for one on a mock transport."""
    return httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False)


async def _checked(url: str, label: str) -> str:
    """``url``, once it has passed the MCP URL safety check as a URL a credential travels to."""
    try:
        await validate_mcp_url(url, has_authorization_token=True, label=label)
    except UnsafeURLError as exc:
        logger.warning("MCP OAuth URL refused label=%s reason=%s", label, exc)
        raise McpOAuthDiscoveryError(f"The {label} URL failed its safety check") from None
    return url


async def _send(
    method: str,
    url: str,
    *,
    label: str,
    timeout: httpx.Timeout,
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
    data: dict[str, str] | None = None,
    auth: httpx.Auth | None = None,
    deadline: float | None = None,
) -> tuple[httpx.Response, dict[str, Any] | None]:
    """Send one request to a checked URL; return the response and its JSON object, if it sent one.

    ``deadline`` bounds sending and reading, in seconds. It covers nothing after the body is read,
    so an answer that arrived is always returned: a refresh token the server rotated is not lost
    to a deadline that fires while the connection closes.
    """
    await _checked(url, label)
    body = bytearray()
    client = _http_client(timeout)
    response: httpx.Response | None = None
    try:
        async with asyncio.timeout(deadline):
            # Identity, so the raw stream is the body: httpx's decoded iterator closes the response
            # when it ends, which would put closing inside the deadline.
            request = client.build_request(
                method, url, headers={**(headers or {}), "Accept-Encoding": "identity"}, json=json, data=data
            )
            response = await client.send(request, auth=auth, stream=True)
            stream = response.stream
            assert isinstance(stream, httpx.AsyncByteStream)
            async for chunk in stream:
                body.extend(chunk)
                if len(body) > _MAX_RESPONSE_BYTES:
                    raise McpOAuthDiscoveryError(f"The {label} answered with more than the gateway reads")
    except TimeoutError:
        logger.info("MCP OAuth request timed out label=%s", label)
        raise McpOAuthDiscoveryError(f"The {label} did not answer in time") from None
    except httpx.HTTPError as exc:
        logger.info("MCP OAuth request failed label=%s error=%s", label, type(exc).__name__)
        raise McpOAuthDiscoveryError(f"The {label} could not be reached") from None
    finally:
        # Shielded: closing is not allowed to undo an answer already read.
        await asyncio.shield(_close(client, response))
    try:
        payload = httpx.Response(200, content=bytes(body)).json()
    except ValueError:
        payload = None
    return response, payload if isinstance(payload, dict) else None


async def _close(client: httpx.AsyncClient, response: httpx.Response | None) -> None:
    if response is not None:
        await response.aclose()
    await client.aclose()


# --------------------------------------------------------------------------- #
# The 401 challenge
# --------------------------------------------------------------------------- #

_TOKEN = r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+"
_AUTH_PARAM = re.compile(rf"^({_TOKEN})\s*=\s*(.+)$")
_CHALLENGE = re.compile(rf"^({_TOKEN})(?:\s+(.+))?$")


def _auth_param(value: str) -> tuple[str, str] | None:
    match = _AUTH_PARAM.fullmatch(value.strip())
    return (match.group(1).lower(), unquote(match.group(2).strip())) if match else None


def parse_bearer_challenge(www_authenticate: str) -> tuple[str | None, str | None]:
    """The ``resource_metadata`` URL and the ``scope`` of a header's first Bearer challenge that has either.

    A header can hold several challenges, so each parameter is grouped with its own before one is chosen.
    """
    challenges: list[tuple[str, dict[str, str]]] = []
    current: dict[str, str] | None = None
    for item in parse_http_list(www_authenticate):
        param = _auth_param(item)
        if param is not None and current is not None:
            current.setdefault(*param)
            continue
        match = _CHALLENGE.fullmatch(item.strip())
        if not match:
            current = None
            continue
        scheme, first = match.groups()
        current = {}
        challenges.append((scheme.lower(), current))
        if first:
            param = _auth_param(first)
            if param is not None:
                current[param[0]] = param[1]
    for scheme, params in challenges:
        if scheme == "bearer" and ("resource_metadata" in params or "scope" in params):
            return params.get("resource_metadata"), params.get("scope")
    return None, None


async def _challenge(server_url: str) -> tuple[str | None, str | None]:
    """Ask the server, unauthenticated, what it challenges with: its metadata URL and scope, if it says."""
    response, _ = await _send(
        "POST",
        server_url,
        label="MCP server",
        timeout=_DISCOVERY_TIMEOUT,
        headers={"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": LATEST_PROTOCOL_VERSION},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": _CLIENT_NAME, "version": "1"},
            },
        },
    )
    if response.status_code != 401:
        return None, None
    return parse_bearer_challenge(response.headers.get("WWW-Authenticate", ""))


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def protected_resource_urls(hinted: str | None, server_url: str) -> list[str]:
    """Where to look for protected-resource metadata: the challenge's URL, then the two well-known ones."""
    origin, path = _origin(server_url), urlsplit(server_url).path
    urls = [hinted] if hinted else []
    if path and path != "/":
        urls.append(urljoin(origin, f"/.well-known/oauth-protected-resource{path}"))
    urls.append(urljoin(origin, "/.well-known/oauth-protected-resource"))
    return list(dict.fromkeys(urls))


def authorization_server_urls(authorization_server: str) -> list[str]:
    """Where to look for an authorization server's metadata: RFC 8414, then the OpenID Connect spellings."""
    origin, path = _origin(authorization_server), urlsplit(authorization_server).path.rstrip("/")
    if not path:
        return [
            urljoin(origin, "/.well-known/oauth-authorization-server"),
            urljoin(origin, "/.well-known/openid-configuration"),
        ]
    return [
        urljoin(origin, f"/.well-known/oauth-authorization-server{path}"),
        urljoin(origin, f"/.well-known/openid-configuration{path}"),
        urljoin(origin, f"{path}/.well-known/openid-configuration"),
    ]


def _covers(resource: str, server_url: str) -> bool:
    """Whether a published ``resource`` names the server: the same origin, and its path at or above the server's."""
    if _origin(resource) != _origin(server_url):
        return False
    resource_path = urlsplit(resource).path.rstrip("/") + "/"
    return (urlsplit(server_url).path.rstrip("/") + "/").startswith(resource_path)


async def _document(url: str, label: str) -> tuple[int, dict[str, Any] | None]:
    response, payload = await _send(
        "GET", url, label=label, timeout=_DISCOVERY_TIMEOUT, headers={"MCP-Protocol-Version": LATEST_PROTOCOL_VERSION}
    )
    return response.status_code, payload if response.status_code == 200 else None


def _string(document: dict[str, Any], field: str) -> str | None:
    value = document.get(field)
    return value if isinstance(value, str) and value else None


def _strings(document: dict[str, Any], field: str) -> list[str]:
    value = document.get(field)
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


async def discover(server_url: str) -> McpOAuthMetadata:
    """Find where users of an MCP server sign in. See :func:`_discover`.

    Raises:
        McpOAuthDiscoveryError: nothing usable was found, or a URL in it failed its safety check.
    """
    try:
        return await _discover(server_url)
    except ValueError:
        # A URL in the server's metadata that no parser accepts, such as a broken IPv6 host.
        raise McpOAuthDiscoveryError("The server's OAuth metadata holds a URL that is not valid") from None


async def _discover(server_url: str) -> McpOAuthMetadata:
    """Find where users of an MCP server sign in.

    The server's own 401 challenge is asked for first, then its protected-resource metadata at the
    challenge's URL and the two well-known ones. A server that publishes none (the 2025-03-26
    revision) is its own authorization server, and then the metadata's ``issuer`` must be the
    server's origin, so such a document cannot move the sign-in to another origin.

    Raises:
        McpOAuthDiscoveryError: nothing usable was found, or a URL in it failed its safety check.
    """
    hinted, challenge_scope = await _challenge(server_url)
    resource_document: dict[str, Any] | None = None
    unavailable = False
    for url in protected_resource_urls(hinted, server_url):
        status, resource_document = await _document(url, "protected resource metadata")
        if resource_document is not None:
            break
        unavailable = unavailable or status >= 500 or status == 429

    if resource_document is None:
        # A 5xx or a 429 says nothing about whether the document exists, so nothing is guessed past it.
        if hinted is not None or unavailable:
            raise McpOAuthDiscoveryError("The server's protected resource metadata could not be read")
        origin = _origin(server_url)
        _, document = await _document(
            urljoin(origin, "/.well-known/oauth-authorization-server"), "authorization server metadata"
        )
        if document is None:
            raise McpOAuthDiscoveryError("The server publishes no OAuth metadata")
        if document.get("issuer") not in (origin, f"{origin}/"):
            raise McpOAuthDiscoveryError("The server's OAuth metadata names another origin as its issuer")
        return await _metadata(document, scope=challenge_scope, resource=None, issuer=_string(document, "issuer"))

    resource = _string(resource_document, "resource")
    if resource is not None and not _covers(resource, server_url):
        raise McpOAuthDiscoveryError("The server's protected resource metadata describes another resource")
    servers = _strings(resource_document, "authorization_servers")
    if not servers:
        raise McpOAuthDiscoveryError("The server's protected resource metadata names no authorization server")
    document = None
    for url in authorization_server_urls(servers[0]):
        _, document = await _document(url, "authorization server metadata")
        if document is not None:
            break
    if document is None:
        raise McpOAuthDiscoveryError("The authorization server's metadata could not be read")
    issuer = _string(document, "issuer")
    if issuer is None or issuer.rstrip("/") != servers[0].rstrip("/"):
        # RFC 8414 section 3.3: the metadata must name the issuer it was fetched for, as the fallback path requires.
        raise McpOAuthDiscoveryError("The authorization server's metadata does not name its own issuer")
    scope = challenge_scope or " ".join(_strings(resource_document, "scopes_supported")) or None
    return await _metadata(document, scope=scope, resource=resource or server_url, issuer=issuer)


async def _metadata(
    document: dict[str, Any], *, scope: str | None, resource: str | None, issuer: str | None
) -> McpOAuthMetadata:
    """Read an authorization server's metadata, checking every endpoint the gateway or a browser will be sent to.

    Every endpoint comes from this one document, whose ``issuer`` the caller has checked is the one
    it was fetched for, so the token endpoint is always one the issuer itself named.
    """
    authorization_endpoint = _string(document, "authorization_endpoint")
    token_endpoint = _string(document, "token_endpoint")
    if authorization_endpoint is None or token_endpoint is None:
        raise McpOAuthDiscoveryError("The authorization server's metadata lacks an endpoint")
    # The MCP authorization specification: a client refuses an authorization server that does not
    # say it supports S256, including one that says nothing.
    if "S256" not in _strings(document, "code_challenge_methods_supported"):
        raise McpOAuthPkceUnsupportedError(
            "The authorization server does not say it supports S256 PKCE, which signing in requires"
        )
    registration_endpoint = _string(document, "registration_endpoint")
    revocation_endpoint = _string(document, "revocation_endpoint")
    await _checked(authorization_endpoint, "authorization endpoint")
    await _checked(token_endpoint, "token endpoint")
    if registration_endpoint is not None:
        await _checked(registration_endpoint, "registration endpoint")
    if revocation_endpoint is not None:
        await _checked(revocation_endpoint, "revocation endpoint")
    return McpOAuthMetadata(
        authorization_endpoint=authorization_endpoint,
        token_endpoint=token_endpoint,
        registration_endpoint=registration_endpoint,
        revocation_endpoint=revocation_endpoint,
        scope=scope,
        resource=resource,
        token_auth_methods=tuple(_strings(document, "token_endpoint_auth_methods_supported")),
        issuer=issuer,
        sends_iss=document.get("authorization_response_iss_parameter_supported") is True,
    )


def auth_method_for(client_secret: str | None, metadata: McpOAuthMetadata) -> str:
    """How a client a manager supplied authenticates at the token endpoint: by what it holds and the server takes."""
    if not client_secret:
        return AUTH_METHOD_NONE
    methods = metadata.token_auth_methods
    if methods and AUTH_METHOD_POST not in methods and AUTH_METHOD_BASIC in methods:
        return AUTH_METHOD_BASIC
    return AUTH_METHOD_POST


# --------------------------------------------------------------------------- #
# Dynamic client registration
# --------------------------------------------------------------------------- #


async def register_client(registration_endpoint: str, redirect_uri: str) -> McpOAuthClient:
    """Register the gateway as a client of an authorization server.

    A confidential client is asked for first. Some servers register only public PKCE clients and
    answer 400 to a request for a secret, so that is tried once more as a public client.

    Raises:
        McpOAuthRegistrationError: the server refused, which no retry and no user can change.
    """
    response, payload = None, None
    for auth_method in (AUTH_METHOD_POST, AUTH_METHOD_NONE):
        response, payload = await _send(
            "POST",
            registration_endpoint,
            label="registration endpoint",
            timeout=_DISCOVERY_TIMEOUT,
            json={
                "client_name": _CLIENT_NAME,
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": auth_method,
            },
        )
        if response.status_code in (200, 201):
            client_id = _string(payload or {}, "client_id")
            if client_id is None:
                break
            secret = _string(payload or {}, "client_secret")
            method = _string(payload or {}, "token_endpoint_auth_method") or auth_method
            if secret is None:
                method = AUTH_METHOD_NONE
            elif method not in (AUTH_METHOD_POST, AUTH_METHOD_BASIC):
                method = AUTH_METHOD_POST
            expires = (payload or {}).get("client_secret_expires_at")
            return McpOAuthClient(
                client_id=client_id,
                client_secret=secret,
                auth_method=method,
                # RFC 7591 section 3.2.1: an epoch second, where zero means the secret does not expire.
                secret_expires_at=_lifetime(expires) if secret is not None else None,
            )
        logger.info("MCP OAuth client registration refused status=%s method=%s", response.status_code, auth_method)
        if response.status_code != 400 or _error_code(payload) == "invalid_redirect_uri":
            break
    raise McpOAuthRegistrationError("The authorization server did not register the gateway as a client")


# --------------------------------------------------------------------------- #
# PKCE and the authorization request
# --------------------------------------------------------------------------- #


def new_code_verifier() -> str:
    """A PKCE code verifier: 43 characters, 256 bits."""
    return secrets.token_urlsafe(32)


def code_challenge(verifier: str) -> str:
    """The S256 challenge of a verifier."""
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")


def authorization_url(
    metadata: McpOAuthMetadata, *, client_id: str, redirect_uri: str, state: str, verifier: str
) -> str:
    """The URL to send the user's browser to."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": code_challenge(verifier),
        "code_challenge_method": "S256",
    }
    if metadata.scope:
        params["scope"] = metadata.scope
    if metadata.resource:
        params["resource"] = metadata.resource
    separator = "&" if urlsplit(metadata.authorization_endpoint).query else "?"
    return f"{metadata.authorization_endpoint}{separator}{urlencode(params)}"


# --------------------------------------------------------------------------- #
# The token endpoint
# --------------------------------------------------------------------------- #


def _error_code(payload: dict[str, Any] | None) -> str | None:
    """The RFC 6749 section 5.2 ``error`` of an answer: the short code alone, never the rest of the body."""
    code = (payload or {}).get("error")
    return code if isinstance(code, str) and _OAUTH_ERROR_CODE.fullmatch(code) else None


def _lifetime(value: object) -> int | None:
    """A usable count of seconds, as ``expires_in`` is; None when the server stated none or nothing usable."""
    if isinstance(value, str):
        # ASCII digits only: str.isdigit() also admits characters int() refuses, and int() refuses very long ones.
        digits = value.strip()
        value = int(digits) if _DIGITS.fullmatch(digits) else None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        return None
    # A number no clock can hold is read as the largest one that is kept.
    return int(min(value, _MAX_SECONDS))


async def _token_request(
    token_endpoint: str,
    client: McpOAuthClient,
    form: dict[str, str],
    *,
    resource: str | None,
    action: str,
    deadline: float | None = None,
) -> McpOAuthTokens:
    if resource:
        form["resource"] = resource
    auth: httpx.Auth | None = None
    if client.auth_method == AUTH_METHOD_BASIC and client.client_secret:
        auth = httpx.BasicAuth(client.client_id, client.client_secret)
    else:
        form["client_id"] = client.client_id
        if client.auth_method != AUTH_METHOD_NONE and client.client_secret:
            form["client_secret"] = client.client_secret
    try:
        response, payload = await _send(
            "POST",
            token_endpoint,
            label="token endpoint",
            timeout=_TOKEN_TIMEOUT,
            data=form,
            auth=auth,
            deadline=deadline,
        )
    except McpOAuthDiscoveryError as exc:
        raise McpOAuthTokenError(str(exc)) from None
    if response.status_code not in (200, 201):
        error_code = _error_code(payload)
        logger.info("MCP OAuth %s refused status=%s error=%s", action, response.status_code, error_code)
        raise McpOAuthTokenError(
            f"The authorization server refused the {action}", status_code=response.status_code, error_code=error_code
        )
    access_token = _string(payload or {}, "access_token")
    token_type = (payload or {}).get("token_type", "Bearer")
    if access_token is None or not isinstance(token_type, str) or token_type.lower() != "bearer":
        raise McpOAuthTokenError(f"The authorization server's answer to the {action} holds no bearer token")
    return McpOAuthTokens(
        access_token=access_token,
        refresh_token=_string(payload or {}, "refresh_token"),
        expires_in=_lifetime((payload or {}).get("expires_in")),
        scope=_string(payload or {}, "scope"),
    )


async def exchange_code(
    token_endpoint: str,
    client: McpOAuthClient,
    *,
    code: str,
    redirect_uri: str,
    verifier: str,
    resource: str | None,
) -> McpOAuthTokens:
    """Trade an authorization code for tokens.

    Raises:
        McpOAuthTokenError: the token endpoint refused, or could not be reached.
    """
    form = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri, "code_verifier": verifier}
    return await _token_request(token_endpoint, client, form, resource=resource, action="code exchange")


async def refresh(
    token_endpoint: str,
    client: McpOAuthClient,
    *,
    refresh_token: str,
    resource: str | None,
    deadline: float | None = None,
) -> McpOAuthTokens:
    """Trade a refresh token for a new access token. The token sent is kept when the server issues no new one.

    ``deadline`` bounds the exchange up to reading its answer, never after it (see ``_send``).

    Raises:
        McpOAuthTokenError: the token endpoint refused, or could not be reached; ``is_permanent`` says which.
    """
    form = {"grant_type": "refresh_token", "refresh_token": refresh_token}
    tokens = await _token_request(token_endpoint, client, form, resource=resource, action="refresh", deadline=deadline)
    if tokens.refresh_token is None:
        return McpOAuthTokens(
            access_token=tokens.access_token,
            refresh_token=refresh_token,
            expires_in=tokens.expires_in,
            scope=tokens.scope,
        )
    return tokens


async def revoke(revocation_endpoint: str, client: McpOAuthClient, *, token: str) -> bool:
    """Ask the authorization server to end a token. Best effort: a refusal is reported, never raised."""
    form = {"token": token}
    auth: httpx.Auth | None = None
    if client.auth_method == AUTH_METHOD_BASIC and client.client_secret:
        auth = httpx.BasicAuth(client.client_id, client.client_secret)
    else:
        form["client_id"] = client.client_id
        if client.auth_method != AUTH_METHOD_NONE and client.client_secret:
            form["client_secret"] = client.client_secret
    try:
        response, _ = await _send(
            "POST", revocation_endpoint, label="revocation endpoint", timeout=_DISCOVERY_TIMEOUT, data=form, auth=auth
        )
    except McpOAuthError:
        return False
    return response.status_code == 200
