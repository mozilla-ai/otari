"""MCP server logos: what counts as an image, how bright it is, where it is looked for, who published a server,
how one candidate is fetched, and how a logo is served."""

import io
import ipaddress
from collections.abc import Sequence

import httpx
import pytest
from PIL import Image
from starlette.datastructures import MutableHeaders
from starlette.requests import Request

from gateway.api.routes._logo_response import logo_response
from gateway.main import _add_security_headers
from gateway.services.mcp_catalog import _logos as logos
from gateway.services.mcp_catalog._extras import EXTRA_SERVERS
from gateway.services.mcp_catalog._logo_fetch import LogoFetcher, declared_icons
from gateway.services.mcp_catalog._popular import POPULAR_SERVERS
from gateway.services.web_retrieval_policy import IPAddress


def _png(color: tuple[int, int, int, int], size: int = 16, *, filled: bool = False) -> bytes:
    image = Image.new("RGBA", (size, size), color if filled else (0, 0, 0, 0))
    if not filled:
        for x in range(4, size - 4):
            for y in range(4, size - 4):
                image.putpixel((x, y), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_an_image_type_is_read_from_its_bytes() -> None:
    assert logos.sniff_image(_png((0, 0, 0, 255))) == "image/png"
    assert logos.sniff_image(b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0h1v1z"/></svg>') == "image/svg+xml"
    assert logos.sniff_image(b"<html><body>not found</body></html>") is None
    assert logos.sniff_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40) is None


def test_an_svg_with_a_document_type_is_refused_so_no_entity_expands() -> None:
    bomb = (
        b'<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;">]>'
        b'<svg xmlns="http://www.w3.org/2000/svg"><text>&b;</text></svg>'
    )
    assert logos.sniff_image(bomb) is None


def test_a_raster_declaring_too_many_pixels_is_refused() -> None:
    buffer = io.BytesIO()
    Image.new("1", (5000, 5000)).save(buffer, format="PNG")
    assert logos.sniff_image(buffer.getvalue()) is None


def test_luminance_reads_the_ink_and_a_filled_square_needs_no_tile() -> None:
    dark = logos.logo_luminance(_png((10, 10, 10, 255)), "image/png")
    light = logos.logo_luminance(_png((250, 250, 250, 255)), "image/png")
    filled = logos.logo_luminance(_png((0, 0, 0, 255), filled=True), "image/png")
    svg = logos.logo_luminance(
        b'<svg xmlns="http://www.w3.org/2000/svg"><g fill="#fff"><path/></g></svg>', "image/svg+xml"
    )

    assert dark is not None and dark < logos.DARK_LOGO_BELOW
    assert light is not None and light > logos.LIGHT_LOGO_ABOVE
    assert filled is None
    assert svg == 1.0
    assert logos.logo_tone(dark) == "dark"
    assert logos.logo_tone(light) == "light"
    assert logos.logo_tone(0.5) is None
    assert logos.logo_tone(dark, "never") is None
    assert logos.logo_tone(None, "always") == "tile"


def test_a_catalog_server_is_looked_for_at_its_icon_then_its_publisher_then_its_github_org() -> None:
    assert logos.catalog_logo_sources("com.stripe/mcp", "https://docs.stripe.com/mcp", "https://stripe.com/i.svg") == [
        "https://stripe.com/i.svg",
        "https://stripe.com/apple-touch-icon.png",
        "https://stripe.com/favicon.ico",
        "https://docs.stripe.com/apple-touch-icon.png",
        "https://docs.stripe.com/favicon.ico",
        "https://stripe.com/",
    ]
    assert logos.catalog_logo_sources("io.github.upstash/context7", None, None) == [
        "https://github.com/upstash.png?size=128"
    ]
    assert logos.host_logo_sources("https://mcp.example.com/mcp") == [
        "https://example.com/apple-touch-icon.png",
        "https://example.com/favicon.ico",
        "https://mcp.example.com/apple-touch-icon.png",
        "https://mcp.example.com/favicon.ico",
        "https://example.com/",
    ]
    assert logos.host_logo_sources("https://thing.vercel.app/mcp") == [
        "https://thing.vercel.app/apple-touch-icon.png",
        "https://thing.vercel.app/favicon.ico",
        "https://thing.vercel.app/",
    ]


@pytest.mark.parametrize(
    ("name", "title", "url", "website", "kind"),
    [
        ("com.stripe/mcp", "Stripe", "https://mcp.stripe.com", None, "vendor"),
        ("io.github.getsentry/sentry-mcp", "Sentry", "https://mcp.sentry.dev/mcp", None, "vendor"),
        ("com.figma.mcp/mcp", "Figma MCP Server", "https://mcp.figma.com/mcp", None, "vendor"),
        ("com.monday/monday.com", "Monday.com", "https://mcp.monday.com/mcp", None, "vendor"),
        ("uk.co.rentercheck/rentercheck", "RenterCheck", "https://rentercheck.co.uk/api/mcp", None, "vendor"),
        ("io.github.pipeworx-io/intercom", "Intercom", "https://gateway.pipeworx.io/intercom", None, "community"),
        ("com.example/notion", "Notion", "https://example.vercel.app/mcp", None, "community"),
        # Its maker's own, but a title that says more than the maker's name could be anyone's server for it.
        ("ai.fudge/fudge", "Fudge for Shopify", "https://mcp.fudge.ai/mcp", None, "community"),
        (
            "io.usefulapi/fieldwire",
            "Fieldwire MCP by usefulapi",
            "https://fieldwire.usefulapi.io/mcp",
            None,
            "community",
        ),
        ("io.github.mcp-dir/crefisa-mcp", "Crefisa MCP", "https://api.mcp.ai/p_crefisa", None, "community"),
        # A name on a hosting platform is anyone's to take.
        ("app.vercel.figma/mcp", "Figma", "https://figma.vercel.app/mcp", "https://figma.vercel.app", "community"),
        # Listed by hand, and only at the host it was checked on.
        ("io.github.github/github-mcp-server", "GitHub", "https://api.githubcopilot.com/mcp/", None, "vendor"),
        ("io.github.github/github-mcp-server", "GitHub", "https://mcp.example.com/mcp", None, "community"),
        ("io.github.someone/github", "GitHub", "https://api.githubcopilot.com/mcp/", None, "community"),
    ],
)
def test_who_published_a_server(name: str, title: str, url: str, website: str | None, kind: str) -> None:
    assert logos.publisher_kind(name, title, url, website) == kind


# Each Popular entry's title and website as the registry listed them on 2026-11-12.
_POPULAR_LISTED = {
    "io.github.github/github-mcp-server": ("GitHub", None),
    "com.notion/mcp": ("Notion", None),
    "app.linear/linear": ("Linear", "https://linear.app"),
    "io.github.getsentry/sentry-mcp": ("Sentry", "https://mcp.sentry.dev"),
    "com.atlassian/atlassian-mcp-server": ("Atlassian Rovo MCP Server", "https://support.atlassian.com/"),
    "com.figma.mcp/mcp": ("Figma MCP Server", None),
    "com.stripe/mcp": ("Stripe", None),
    "com.vercel/vercel-mcp": ("Vercel", None),
    "com.supabase/mcp": ("Supabase", "https://supabase.com/mcp"),
    "com.neon/mcp": ("Neon", "https://neon.com/docs/ai/neon-mcp-server"),
    "com.gitlab/mcp": ("Gitlab", None),
    "io.github.upstash/context7": ("Context7", "https://context7.com"),
    "co.huggingface/hf-mcp-server": ("Hugging Face", None),
    "com.cloudflare.mcp/mcp": ("Cloudflare", None),
    "io.github.grafana/mcp-grafana": ("Grafana", "https://grafana.com/docs/grafana/latest/developer-resources/mcp/"),
    "io.github.PostHog/mcp": ("PostHog MCP Server", None),
    "com.canva.mcp/mcp": ("Canva MCP", None),
    "com.monday/monday.com": ("Monday.com", None),
    "com.paypal.mcp/mcp": ("Paypal", None),
    "com.zapier/mcp": ("Zapier", "https://docs.zapier.com/mcp/home"),
    "ai.granola/mcp": ("Granola", "https://www.granola.ai"),
}


def test_every_popular_server_is_its_makers_own() -> None:
    # GitHub's and Atlassian's pass by the hand-kept list, the rest by the rule.
    community = {
        entry.name
        for entry in POPULAR_SERVERS
        if logos.publisher_kind(entry.name, *_POPULAR_LISTED[entry.name][:1], entry.url, _POPULAR_LISTED[entry.name][1])
        == "community"
    }
    assert community == set()
    assert {name for name, _ in logos.OFFICIAL_ENDPOINTS} <= {entry.name for entry in POPULAR_SERVERS}


def test_every_bundled_extra_is_a_makers_own_https_server_listed_once() -> None:
    assert len({extra.name for extra in EXTRA_SERVERS}) == len(EXTRA_SERVERS)
    for extra in EXTRA_SERVERS:
        assert extra.url.startswith("https://")
        assert logos.publisher_kind(extra.name, extra.title, extra.url, extra.website_url) == "vendor"
        # The first place its logo is looked for is the icon named here, else its maker's site.
        assert logos.catalog_logo_sources(extra.name, extra.website_url, extra.icon_url)


def test_every_bundled_logo_loads_and_names_a_popular_server() -> None:
    popular = {entry.name for entry in POPULAR_SERVERS}
    for name in logos._BUNDLED_FILES:
        logo = logos.bundled_logo(name)
        assert name in popular
        assert logo is not None and logo.content_type == "image/svg+xml"
        assert logos.sniff_image(logo.content) == "image/svg+xml"


class _Resolver:
    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    async def resolve(self, host: str, port: int) -> Sequence[IPAddress]:
        answer = self.answers.get(host)
        return (ipaddress.ip_address(answer),) if answer else ()


def _fetcher(handler: httpx.MockTransport, answers: dict[str, str]) -> LogoFetcher:
    return LogoFetcher(resolver=_Resolver(answers), transport_factory=lambda: handler)


@pytest.mark.asyncio
async def test_a_logo_is_fetched_through_a_checked_redirect() -> None:
    png = _png((10, 10, 10, 255))
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.headers['host']}{request.url.path}")
        if request.headers["host"] == "example.com":
            return httpx.Response(302, headers={"location": "https://cdn.example.net/logo.png"})
        return httpx.Response(200, content=png, headers={"content-type": "text/html"})

    fetcher = _fetcher(
        httpx.MockTransport(handle), {"example.com": "93.184.216.34", "cdn.example.net": "93.184.216.35"}
    )
    logo = await fetcher.fetch("https://example.com/favicon.ico")
    await fetcher.aclose()

    # The type comes from the bytes, not the header that called them HTML.
    assert logo is not None and logo.content_type == "image/png"
    assert seen == ["example.com/favicon.ico", "cdn.example.net/logo.png"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("location", "answers"),
    [
        ("https://internal.example/logo.png", {"internal.example": "10.0.0.1"}),
        ("https://metadata.example/logo.png", {"metadata.example": "169.254.169.254"}),
        ("http://plain.example/logo.png", {"plain.example": "93.184.216.36"}),
    ],
)
async def test_a_redirect_to_an_internal_address_or_plain_http_is_refused(
    location: str, answers: dict[str, str]
) -> None:
    dialed: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        dialed.append(request.headers["host"])
        return httpx.Response(302, headers={"location": location})

    fetcher = _fetcher(httpx.MockTransport(handle), {"example.com": "93.184.216.34", **answers})
    assert await fetcher.fetch("https://example.com/favicon.ico") is None
    await fetcher.aclose()
    assert dialed == ["example.com"]


@pytest.mark.asyncio
async def test_too_many_redirects_an_oversized_body_and_an_http_start_are_refused() -> None:
    hops = iter(range(10))

    def loop(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": f"https://example.com/{next(hops)}"})

    def huge(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x89PNG\r\n\x1a\n" + b"\x00" * (300 * 1024))

    answers = {"example.com": "93.184.216.34"}
    for handler in (loop, huge):
        fetcher = _fetcher(httpx.MockTransport(handler), answers)
        assert await fetcher.fetch("https://example.com/favicon.ico") is None
        await fetcher.aclose()
    fetcher = _fetcher(httpx.MockTransport(huge), answers)
    assert await fetcher.fetch("http://example.com/favicon.ico") is None
    await fetcher.aclose()


def _request(query: str = "", headers: dict[str, str] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/logo",
            "query_string": query.encode(),
            "headers": [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()],
        }
    )


def test_a_logo_is_served_so_it_cannot_run_and_is_cached_by_its_version() -> None:
    logo = logos.make_logo(b'<svg xmlns="http://www.w3.org/2000/svg"><path/></svg>', "image/svg+xml")

    versioned = logo_response(_request(f"v={logo.etag}"), logo)
    bare = logo_response(_request(), logo)
    again = logo_response(_request(headers={"If-None-Match": f'"{logo.etag}"'}), logo)
    missing = logo_response(_request(), None)

    assert versioned.media_type == "image/svg+xml"
    assert versioned.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in versioned.headers["content-security-policy"]
    assert versioned.headers["cache-control"] == "private, max-age=604800, immutable"
    assert bare.headers["cache-control"] == "private, max-age=300"
    assert again.status_code == 304
    assert (missing.status_code, missing.headers["cache-control"]) == (404, "no-store")


def test_the_security_headers_keep_a_private_cache_a_route_asked_for_and_nothing_else() -> None:
    own = MutableHeaders(raw=[(b"cache-control", b"private, max-age=300")])
    _add_security_headers(own, "/api/v1/mcp-catalog/logos/x", 200)
    public = MutableHeaders(raw=[(b"cache-control", b"public, max-age=300")])
    _add_security_headers(public, "/api/v1/keys", 200)
    failed = MutableHeaders(raw=[(b"cache-control", b"private, max-age=300")])
    _add_security_headers(failed, "/api/v1/mcp-catalog/logos/x", 404)

    assert own["cache-control"] == "private, max-age=300"
    assert public["cache-control"] == "private, no-store, no-cache"
    assert failed["cache-control"] == "private, no-store, no-cache"


def test_a_home_page_is_read_only_for_the_icons_it_declares() -> None:
    page = (
        '<link rel="stylesheet" href="/a.css"><link rel="icon" href="/favicon.ico">'
        '<link rel="apple-touch-icon" sizes="180x180" href="https://cdn.example.com/touch.png">'
        '<link rel="icon" href="http://plain.example.com/i.png"><link rel="icon" href="data:image/png;base64,AA">'
    )
    assert declared_icons(page, "https://www.example.com/") == [
        "https://cdn.example.com/touch.png",
        "https://www.example.com/favicon.ico",
    ]


@pytest.mark.asyncio
async def test_a_site_root_is_searched_for_its_declared_icon() -> None:
    png = _png((10, 10, 10, 255))

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text='<head><link rel="icon" href="/static/logo.png"></head>')
        if request.url.path == "/static/logo.png":
            return httpx.Response(200, content=png)
        return httpx.Response(404)

    fetcher = _fetcher(httpx.MockTransport(handle), {"example.com": "93.184.216.34"})
    logo = await fetcher.fetch("https://example.com/")
    await fetcher.aclose()

    assert logo is not None and logo.content == png
