"""Reading the official MCP Registry: one bounded page at a time, and which of its entries the catalog keeps."""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, NamedTuple, Self

import httpx

from gateway.models.mcp_catalog import SOURCE_REGISTRY
from gateway.schemas.mcp_catalog import McpCatalogEntry
from gateway.services.url_safety import UnsafeURLError, validate_mcp_url, validate_public_https_url_shape

REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0.1/servers"
PAGE_SIZE = 100
MAX_PAGE_BYTES = 4 * 1024 * 1024
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_ATTEMPTS = 3
_BACKOFF_SECONDS = 2.0
# Well under the ten minutes after which another replica takes a silent sync for dead.
PAGE_DEADLINE_SECONDS = 180.0

_OFFICIAL_META = "io.modelcontextprotocol.registry/official"
# The one transport a stored server is dialed over.
_TRANSPORT = "streamable-http"
# The one credential a stored server can send.
_TOKEN_HEADER = "authorization"

MAX_NAME_LENGTH = 255
# A stored server's name holds at most this many characters, and a catalog title becomes one.
MAX_TITLE_LENGTH = 128
MAX_DESCRIPTION_LENGTH = 2000
MAX_URL_LENGTH = 2048
MAX_VERSION_LENGTH = 64

_CONTROL_OR_SPACE = re.compile(r"[\x00-\x1f\x7f\s]+")
# The image types a browser draws in an <img>, where none can run a script. An icon that names no type is kept.
_ICON_TYPES = frozenset({"image/svg+xml", "image/png", "image/webp", "image/jpeg", "image/gif"})
# Name segments that say nothing about whose server it is.
_GENERIC_SEGMENTS = frozenset({"mcp", "mcp-server", "server", "remote", "api"})


class RegistryError(Exception):
    """A registry page could not be read. The message is written here and carries nothing the registry sent."""


class RegistryPage(NamedTuple):
    """One page of registry entries, and the cursor of the page after it, or None on the last."""

    items: list[dict[str, Any]]
    next_cursor: str | None


FetchPage = Callable[[str | None], Awaitable[RegistryPage]]


def _storable(value: str) -> bool:
    """Whether every database the gateway runs on can store ``value``: a lone surrogate or a NUL is refused by one."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return "\x00" not in value


def _text(value: object, limit: int) -> str | None:
    """``value`` as one line of at most ``limit`` characters, or None when it is not storable text or is empty."""
    if not isinstance(value, str) or not _storable(value):
        return None
    line = _CONTROL_OR_SPACE.sub(" ", value).strip()
    return line[:limit].rstrip() or None


def fallback_title(name: str) -> str:
    """A readable title for a server that publishes none, from its reverse-DNS name.

    ``com.stripe/mcp`` reads as ``Stripe`` and ``io.github.getsentry/sentry-mcp`` as ``Sentry``.
    """
    namespace, _, segment = name.partition("/")
    segment = segment.removeprefix("mcp-").removesuffix("-mcp").removesuffix("-server")
    if segment and segment not in _GENERIC_SEGMENTS:
        label = segment
    else:
        # The namespace is a reversed domain: drop the top-level label, and a code host's own.
        own = namespace.split(".")[1:]
        # Unless the code host is the owner itself, as com.gitlab/mcp is.
        labels = [part for part in own if part not in {"github", "gitlab", "mcp"}] or own
        label = labels[-1] if labels else namespace
    label = re.sub(r"[-_]+", " ", label).strip() or name
    return label[:1].upper() + label[1:]


def _connectable_remote(server: dict[str, Any]) -> tuple[str, bool] | None:
    """The URL of the first remote a stored server could dial, and whether it declares a required token."""
    remotes = server.get("remotes")
    if not isinstance(remotes, list):
        return None
    for remote in remotes:
        if not isinstance(remote, dict) or remote.get("type") != _TRANSPORT:
            continue
        url = remote.get("url")
        # A templated URL has to be filled in per customer, which a stored server cannot do.
        if not isinstance(url, str) or len(url) > MAX_URL_LENGTH or "{" in url:
            continue
        url = url.strip()
        if not _storable(url) or _CONTROL_OR_SPACE.search(url):
            continue
        try:
            validate_public_https_url_shape(url, label="MCP catalog server")
        except UnsafeURLError:
            continue
        headers = remote.get("headers")
        required = {
            str(header.get("name", "")).strip().lower()
            for header in (headers if isinstance(headers, list) else [])
            if isinstance(header, dict) and header.get("isRequired")
        }
        if required - {_TOKEN_HEADER}:
            continue
        return url, _TOKEN_HEADER in required
    return None


def _icon_rank(icon: dict[str, Any]) -> tuple[int, int, int]:
    """How well an icon suits the dashboard, lower first: not a dark-theme variant, a vector, then the larger."""
    largest = 0
    sizes = icon.get("sizes")
    for size in sizes if isinstance(sizes, list) else []:
        if size == "any":
            largest = max(largest, 10_000)
        elif isinstance(size, str) and re.fullmatch(r"\d{1,5}x\d{1,5}", size):
            largest = max(largest, int(size.partition("x")[0]))
    return (icon.get("theme") == "dark", icon.get("mimeType") != "image/svg+xml", -largest)


def _icon_url(server: dict[str, Any]) -> str | None:
    """The best of the icons the entry names, as a public https URL, or None when it names none usable.

    Only the URL is kept. The gateway never fetches it: the dashboard draws it with an <img>.
    """
    icons = server.get("icons")
    usable: list[tuple[tuple[int, int, int], str]] = []
    for icon in icons if isinstance(icons, list) else []:
        if not isinstance(icon, dict):
            continue
        src = icon.get("src")
        mime = icon.get("mimeType")
        if not isinstance(src, str) or len(src) > MAX_URL_LENGTH:
            continue
        if mime is not None and (not isinstance(mime, str) or mime not in _ICON_TYPES):
            continue
        src = src.strip()
        if not _storable(src) or _CONTROL_OR_SPACE.search(src):
            continue
        try:
            validate_public_https_url_shape(src, label="MCP catalog icon")
        except UnsafeURLError:
            continue
        usable.append((_icon_rank(icon), src))
    return min(usable)[1] if usable else None


def entry_from_registry_item(item: object) -> McpCatalogEntry | None:
    """The catalog entry for one registry list item, or None for one Otari cannot connect to.

    Kept: an active server with a streamable HTTP remote at a public https URL
    that needs no header but ``Authorization``. Everything else is left out: a
    package to run locally, an SSE-only remote, a templated URL, a deprecated entry.
    """
    if not isinstance(item, dict):
        return None
    server = item.get("server")
    if not isinstance(server, dict):
        return None
    name = server.get("name")
    if not isinstance(name, str) or not name or len(name) > MAX_NAME_LENGTH:
        return None
    if not _storable(name) or _CONTROL_OR_SPACE.search(name):
        return None
    meta = item.get("_meta")
    official = meta.get(_OFFICIAL_META) if isinstance(meta, dict) else None
    status = official.get("status") if isinstance(official, dict) else None
    if status is not None and status != "active":
        return None
    remote = _connectable_remote(server)
    if remote is None:
        return None
    url, requires_token = remote
    website = _text(server.get("websiteUrl"), MAX_URL_LENGTH)
    if website is not None:
        try:
            validate_public_https_url_shape(website, label="MCP catalog website")
        except UnsafeURLError:
            website = None
    return McpCatalogEntry(
        name=name,
        title=_text(server.get("title"), MAX_TITLE_LENGTH) or fallback_title(name)[:MAX_TITLE_LENGTH],
        description=_text(server.get("description"), MAX_DESCRIPTION_LENGTH),
        url=url,
        requires_token=requires_token,
        website_url=website,
        icon_url=_icon_url(server),
        version=_text(server.get("version"), MAX_VERSION_LENGTH),
        source=SOURCE_REGISTRY,
    )


class RegistryClient:
    """Fetches registry pages. Follows no redirect, and reads at most ``MAX_PAGE_BYTES`` of an answer."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._transport = transport
        self._sleep = sleep
        # Opened by the first page fetched, so a poll that finds no sync due builds nothing.
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if self._client is not None:
            await self._client.aclose()

    def _open(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                transport=self._transport,
                timeout=_TIMEOUT,
                follow_redirects=False,
                headers={"User-Agent": "otari-gateway", "Accept": "application/json"},
            )
        return self._client

    async def fetch_page(self, cursor: str | None) -> RegistryPage:
        """The page after ``cursor``, or the first. A 5xx or a network error is retried; anything else is not.

        The whole page, retries included, gets ``PAGE_DEADLINE_SECONDS``: the timeouts above bound each
        read, and a registry sending a byte at a time would never trip them.

        Raises:
            RegistryError: the page could not be read.
        """
        try:
            async with asyncio.timeout(PAGE_DEADLINE_SECONDS):
                return await self._fetch_page(cursor)
        except TimeoutError:
            raise RegistryError(f"a registry page took more than {int(PAGE_DEADLINE_SECONDS)} seconds") from None

    async def _fetch_page(self, cursor: str | None) -> RegistryPage:
        try:
            # The address is checked on every page, so a name re-pointed mid-crawl is refused too.
            await validate_mcp_url(REGISTRY_URL, has_authorization_token=False, label="MCP registry")
        except UnsafeURLError:
            raise RegistryError("the registry's address failed the gateway's URL safety check") from None
        params: dict[str, str | int] = {"version": "latest", "limit": PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor
        failure = "no answer"
        for attempt in range(_ATTEMPTS):
            if attempt:
                await self._sleep(_BACKOFF_SECONDS * attempt)
            try:
                status_code, body = await self._get(params)
            except httpx.HTTPError as exc:
                failure = f"network error ({type(exc).__name__})"
                continue
            if status_code >= 500:
                failure = f"HTTP {status_code}"
                continue
            if status_code != 200:
                raise RegistryError(f"the registry answered HTTP {status_code}")
            return _parse_page(body)
        raise RegistryError(f"the registry did not answer after {_ATTEMPTS} attempts: {failure}")

    async def _get(self, params: dict[str, str | int]) -> tuple[int, bytes]:
        async with self._open().stream("GET", REGISTRY_URL, params=params) as response:
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_PAGE_BYTES:
                    raise RegistryError(f"a registry page exceeded {MAX_PAGE_BYTES} bytes")
                chunks.append(chunk)
            return response.status_code, b"".join(chunks)


def _parse_page(body: bytes) -> RegistryPage:
    try:
        payload = json.loads(body)
    except ValueError:
        raise RegistryError("the registry answered with something that is not JSON") from None
    servers = payload.get("servers") if isinstance(payload, dict) else None
    if not isinstance(servers, list):
        raise RegistryError("the registry answered with an unexpected shape")
    # The registry's schema requires ``metadata`` on every page, so a page without it is not read as the last one.
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RegistryError("the registry answered with an unexpected shape")
    cursor = metadata.get("nextCursor")
    # Read as the last page, a cursor of the wrong shape would let the sync delete every server it had not reached.
    if cursor is not None and not (isinstance(cursor, str) and cursor):
        raise RegistryError("the registry answered with a cursor of an unexpected shape")
    return RegistryPage(items=[item for item in servers if isinstance(item, dict)], next_cursor=cursor)
