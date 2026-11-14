"""What a server's logo is and where to look for it, with no network and no database.

Most registry entries name no icon, so each server gets a short list of places to
try, best first: the icon its registry entry names, the publisher's and the
website's touch icon and favicon, then the GitHub organization's avatar for an
``io.github.*`` name. A server added by URL alone gets its host's. The first
image that passes :func:`sniff_image` is the logo.

Also here: whether a catalog entry is its maker's own server or someone else's
(:func:`publisher_kind`), and how bright a logo is, so a dashboard can put a tile
behind one that would vanish on its background.
"""

import hashlib
import io
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from importlib import resources
from typing import ClassVar, Literal, Self, cast
from urllib.parse import urlsplit

# A fetched logo larger than this is refused.
MAX_LOGO_BYTES = 256 * 1024
# An admin's upload: the dashboard sends a 256 px PNG, so this is generous.
MAX_UPLOAD_BYTES = 512 * 1024
# A raster declaring more pixels than this is refused before it is decoded.
_MAX_PIXELS = 4096 * 4096

PublisherKind = Literal["vendor", "community"]
LogoTone = Literal["dark", "light", "tile"]
LogoTile = Literal["auto", "always", "never"]

_RASTER_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"\x00\x00\x01\x00", "image/x-icon"),
)

# A logo darker than this disappears on a dark background, and one lighter than LIGHT_LOGO_ABOVE on a light one.
DARK_LOGO_BELOW = 0.35
LIGHT_LOGO_ABOVE = 0.85
_SAMPLE_SIZE = 64
# A raster whose edge is this opaque this much of the way round carries its own background.
_OPAQUE = 250
_FILLED_EDGE = 0.75
_FILL = re.compile(r"fill\s*:\s*(#[0-9a-f]{3,6}|none|white|black)\b", re.IGNORECASE)
_SHAPES = frozenset({"path", "rect", "circle", "ellipse", "polygon", "polyline", "line", "text"})


@dataclass(frozen=True)
class Logo:
    """An image the gateway serves as a logo."""

    content: bytes
    content_type: str
    etag: str
    luminance: float | None

    MAX_UPLOAD_BYTES: ClassVar[int] = MAX_UPLOAD_BYTES

    @classmethod
    def of(cls, content: bytes, content_type: str) -> Self:
        """The logo these bytes make, with its digest and how bright its ink is."""
        return cls(content, content_type, etag_of(content), logo_luminance(content, content_type))

    @staticmethod
    def sniff(content: bytes) -> str | None:
        """The image type these bytes are, or None when they are no image a logo may be."""
        return sniff_image(content)

    @staticmethod
    def tone(luminance: float | None, tile: str = "auto") -> LogoTone | None:
        """Which background a logo this bright needs a tile on (``logo_tone``)."""
        return logo_tone(luminance, tile)


def etag_of(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def _parse_svg(content: bytes) -> ElementTree.Element | None:
    """The root of a well-formed SVG, or None. A document type is refused outright, so no entity is ever expanded."""
    head = content[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in content.lower():
        return None
    try:
        root = ElementTree.fromstring(content)
    # An encoding the parser does not know, or a multi-byte one it does not take, is a LookupError or a ValueError.
    except (ElementTree.ParseError, LookupError, ValueError):
        return None
    return root if root.tag.rsplit("}", 1)[-1] == "svg" else None


def _raster_size_ok(content: bytes) -> bool:
    from PIL import Image, UnidentifiedImageError  # noqa: PLC0415 - only logo handling needs it

    try:
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return False
    return 0 < width * height <= _MAX_PIXELS


def sniff_image(content: bytes) -> str | None:
    """The image type ``content`` really is, or None. Never trusts a Content-Type header.

    An SVG counts only when it parses with no document type; a raster only when Pillow
    reads its header and it is not absurdly large.
    """
    for signature, content_type in _RASTER_SIGNATURES:
        if content.startswith(signature):
            return content_type if _raster_size_ok(content) else None
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp" if _raster_size_ok(content) else None
    head = content[:512].lstrip().lower()
    if head.startswith((b"<?xml", b"<svg", b"<!--")) and _parse_svg(content) is not None:
        return "image/svg+xml"
    return None


def _channel_luminance(red: float, green: float, blue: float) -> float:
    """Rec. 709 luma of 0 to 1 sRGB values."""
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _hex_luminance(color: str) -> float | None:
    value = color.lstrip("#").lower()
    if value == "white":
        return 1.0
    if value == "black":
        return 0.0
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6 or not re.fullmatch(r"[0-9a-f]{6}", value):
        return None
    red, green, blue = (int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return _channel_luminance(red, green, blue)


def _raster_luminance(content: bytes) -> float | None:
    from PIL import Image  # noqa: PLC0415 - only logo handling needs it

    with Image.open(io.BytesIO(content)) as image:
        image.thumbnail((_SAMPLE_SIZE, _SAMPLE_SIZE))
        rgba = image.convert("RGBA")
        width, height = rgba.size
        # RGBA mode: every pixel is four 0 to 255 channels.
        pixels = cast(list[tuple[int, int, int, int]], list(rgba.get_flattened_data()))
    # A logo drawn on its own filled square never fades into a background, whatever its colors.
    edge = [
        pixels[y * width + x]
        for y in range(height)
        for x in range(width)
        if x in (0, width - 1) or y in (0, height - 1)
    ]
    if edge and sum(alpha >= _OPAQUE for *_, alpha in edge) / len(edge) >= _FILLED_EDGE:
        return None
    weight = sum(alpha for *_, alpha in pixels)
    if weight == 0:
        return None
    # Transparent pixels show the background, not the logo, so each pixel counts by its opacity.
    return float(sum(_channel_luminance(r / 255, g / 255, b / 255) * alpha for r, g, b, alpha in pixels) / weight)


def _declared_fill(element: ElementTree.Element) -> str | None:
    style = element.attrib.get("style", "")
    for source in (style, f"fill:{element.attrib.get('fill', '')}"):
        match = _FILL.search(source)
        if match:
            return match.group(1)
    return None


def _svg_luminance(content: bytes) -> float | None:
    """Average brightness of the flat colors an SVG's shapes paint with, fills inherited as SVG inherits them.

    A shape with no fill anywhere above it paints black. Gradients and colors this cannot read are skipped.
    """
    root = _parse_svg(content)
    if root is None:
        return None
    values: list[float] = []

    def visit(element: ElementTree.Element, inherited: str) -> None:
        fill = _declared_fill(element) or inherited
        if element.tag.rsplit("}", 1)[-1] in _SHAPES and fill.lower() != "none":
            luminance = _hex_luminance(fill)
            if luminance is not None:
                values.append(luminance)
        for child in element:
            visit(child, fill)

    visit(root, "black")
    return sum(values) / len(values) if values else None


def logo_luminance(content: bytes, content_type: str) -> float | None:
    """How bright a logo's ink is, 0 to 1, or None when it cannot be told. Never raises."""
    try:
        if content_type == "image/svg+xml":
            return _svg_luminance(content)
        return _raster_luminance(content)
    except Exception:  # noqa: BLE001 - measuring is best effort, and a logo that cannot be measured gets no tile
        return None


def make_logo(content: bytes, content_type: str) -> Logo:
    return Logo.of(content, content_type)


def logo_tone(luminance: float | None, tile: str = "auto") -> LogoTone | None:
    """``dark`` when the logo needs a light tile on a dark background, ``light`` a gray one on a light
    background, ``tile`` one on every theme, None when it reads on both or an admin said never."""
    if tile == "never":
        return None
    if tile == "always":
        return "tile"
    if luminance is None:
        return None
    if luminance < DARK_LOGO_BELOW:
        return "dark"
    if luminance > LIGHT_LOGO_ABOVE:
        return "light"
    return None


# Country suffixes under which a company registers its own name: the label left of them is the owner.
_COUNTRY_SUFFIXES = frozenset({"co.uk", "com.au", "co.jp", "com.br", "co.nz", "co.in", "com.mx"})
# Hosting suffixes under which anyone can take a name in minutes, so a name there proves nothing about its owner.
_HOSTING_SUFFIXES = frozenset(
    {
        "vercel.app",
        "workers.dev",
        "pages.dev",
        "netlify.app",
        "onrender.com",
        "herokuapp.com",
        "fly.dev",
        "railway.app",
        "up.railway.app",
        "github.io",
        "run.app",
        "deno.dev",
        "azurewebsites.net",
        "smithery.ai",
    }
)
_SHARED_SUFFIXES = _COUNTRY_SUFFIXES | _HOSTING_SUFFIXES

# Makers' own servers the rule in :func:`publisher_kind` cannot recognize, as (registry name, endpoint host).
# HAND-MAINTAINED, and each entry needs evidence from the maker itself, checked on the date given.
OFFICIAL_ENDPOINTS: frozenset[tuple[str, str]] = frozenset(
    {
        # GitHub serves it from a domain not named GitHub. The README of github.com/github/github-mcp-server
        # gives this URL as the remote server, and the registry lists it under GitHub's own organization.
        # Checked 2026-11-12.
        ("io.github.github/github-mcp-server", "api.githubcopilot.com"),
        # On atlassian.com under Atlassian's own namespace; only its title ("Atlassian Rovo MCP Server") says
        # more than the domain's name. Checked 2026-11-12.
        ("com.atlassian/atlassian-mcp-server", "mcp.atlassian.com"),
    }
)
# Words a title may add to the maker's name and still be the maker's name.
_GENERIC_TITLE_WORDS = frozenset({"mcp", "server", "remote", "official", "the"})


def _host(url_or_host: str | None) -> str:
    if not url_or_host:
        return ""
    host = urlsplit(url_or_host).hostname if "//" in url_or_host else url_or_host
    return (host or "").rstrip(".").lower()


def registrable_domain(url_or_host: str | None) -> str | None:
    """The domain its owner registered: ``mcp.stripe.com`` is ``stripe.com``, ``foo.vercel.app`` itself."""
    labels = [label for label in _host(url_or_host).split(".") if label]
    for size in (3, 2):
        if len(labels) > size and ".".join(labels[-size:]) in _SHARED_SUFFIXES:
            return ".".join(labels[-size - 1 :])
    return ".".join(labels[-2:]) if len(labels) >= 2 else None


def _owner_label(url_or_host: str | None) -> str | None:
    domain = registrable_domain(url_or_host)
    return domain.split(".")[0] if domain else None


def _github_owner(name: str) -> str | None:
    """The GitHub account a server is published under, when its namespace is ``io.github.<account>``."""
    labels = name.partition("/")[0].split(".")
    if len(labels) >= 3 and labels[:2] == ["io", "github"]:
        return labels[2]
    return None


def publisher_from_name(name: str) -> str:
    """Who publishes a server, from its registry namespace.

    ``com.stripe/mcp`` is ``stripe.com``, ``com.paypal.mcp/mcp`` is ``paypal.com``, and
    ``io.github.getsentry/x`` is ``github.com/getsentry``.
    """
    owner = _github_owner(name)
    if owner is not None:
        return f"github.com/{owner}"
    domain = list(reversed(name.partition("/")[0].split(".")))
    while len(domain) > 2 and domain[0] == "mcp":
        domain = domain[1:]
    return ".".join(domain)


def _letters(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _is_named(title: str, domain: str) -> bool:
    """Whether ``title`` is the name of ``domain`` and no more: "Figma MCP Server" and "Monday.com" are, on
    figma.com and monday.com; "Fudge for Shopify" on fudge.ai is not."""
    label, *suffix = domain.split(".")
    words = [word for word in re.split(r"[^a-z0-9]+", title.lower()) if word and word not in _GENERIC_TITLE_WORDS]
    while len(words) > 1 and words[-1] in suffix:
        words.pop()
    return "".join(words) == _letters(label)


def publisher_kind(name: str, title: str, url: str, website_url: str | None) -> PublisherKind:
    """``vendor`` when the server is its maker's own, on the maker's own domain, else ``community``.

    Vendor means three things hold. The domain the endpoint lives on belongs to the publisher: its
    registry namespace, its GitHub organization or its website. That domain is not a hosting
    platform's, where anyone can take a name. And the server's title is that domain's name and no
    more, so "Stripe" on stripe.com is Stripe's, while "Fudge for Shopify" on fudge.ai is nobody's
    official Shopify server and ``io.github.pipeworx-io/intercom`` on pipeworx.io is not Intercom's.
    A server on ``OFFICIAL_ENDPOINTS`` is vendor without the rule.

    This reads names. It cannot tell a maker from someone who registered the maker's name under
    another ending, so a dashboard that shows it should show the endpoint's host beside it.
    """
    if (name, _host(url)) in OFFICIAL_ENDPOINTS:
        return "vendor"
    domain = registrable_domain(url)
    if not domain:
        return "community"
    label, _, suffix = domain.partition(".")
    if len(label) < 3 or suffix in _HOSTING_SUFFIXES:
        return "community"
    owned = {_owner_label(website_url)}
    github_owner = _github_owner(name)
    if github_owner is not None:
        if label in _letters(github_owner):
            owned.add(label)
    else:
        owned.add(_owner_label(publisher_from_name(name)))
    return "vendor" if label in owned and _is_named(title, domain) else "community"


def _site_icons(host: str) -> list[str]:
    return [f"https://{host}/apple-touch-icon.png", f"https://{host}/favicon.ico"]


def _site_page(host: str) -> str:
    """A site's home page, whose declared icons are tried when the usual paths hold none."""
    return f"https://{host}/"


def catalog_logo_sources(name: str, website_url: str | None, icon_url: str | None) -> list[str]:
    """Where to look for a catalog server's logo, best first.

    A GitHub organization's avatar comes last: it is the publisher's, not always the server's
    (Context7 is published by upstash).
    """
    sources: list[str] = [icon_url] if icon_url else []
    publisher = publisher_from_name(name)
    github_owner = _github_owner(name)
    on_github = github_owner is not None
    if not on_github and "." in publisher:
        sources += _site_icons(publisher)
    website_host = _host(website_url)
    if website_host and website_host not in (publisher, "github.com", "www.github.com"):
        sources += _site_icons(website_host)
        domain = registrable_domain(website_host)
        if domain and domain != website_host and domain != publisher:
            sources += _site_icons(domain)
    if not on_github and "." in publisher:
        sources.append(_site_page(publisher))
    if github_owner is not None and re.fullmatch(r"[A-Za-z0-9-]{1,39}", github_owner):
        sources.append(f"https://github.com/{github_owner}.png?size=128")
    return list(dict.fromkeys(sources))


def _url_host(url: str) -> str:
    """The host of an http or https URL, or "" for anything else."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return ""
    if parts.scheme not in ("http", "https"):
        return ""
    return (parts.hostname or "").rstrip(".").lower()


def host_logo_sources(url: str) -> list[str]:
    """Where to look for the logo of a server known only by its URL: its owner's site, then its own host."""
    host = _url_host(url)
    if not host:
        return []
    sources: list[str] = []
    domain = registrable_domain(host)
    if domain and domain != host:
        sources += _site_icons(domain)
    sources += _site_icons(host)
    sources.append(_site_page(domain or host))
    return list(dict.fromkeys(sources))


def host_key(url: str) -> str | None:
    """The logo key of a server known only by its URL, or None for a URL with no host."""
    host = _url_host(url)
    return f"host:{host}" if host else None


def catalog_key(name: str) -> str:
    return f"catalog:{name}"


# Bundled logos for servers on the Popular list, keyed by registry name. From Simple Icons
# (https://simpleicons.org), whose icon data is CC0-1.0; the marks stay their owners' trademarks.
_BUNDLED_FILES: dict[str, str] = {
    "com.stripe/mcp": "stripe.svg",
    "com.paypal.mcp/mcp": "paypal.svg",
    "com.zapier/mcp": "zapier.svg",
    "com.cloudflare.mcp/mcp": "cloudflare.svg",
    "com.supabase/mcp": "supabase.svg",
    "io.github.getsentry/sentry-mcp": "sentry.svg",
    "io.github.grafana/mcp-grafana": "grafana.svg",
}
_bundled_cache: dict[str, Logo] = {}


def bundled_logo(name: str) -> Logo | None:
    """The logo shipped with Otari for this catalog name, or None."""
    file_name = _BUNDLED_FILES.get(name)
    if file_name is None:
        return None
    cached = _bundled_cache.get(name)
    if cached is None:
        content = resources.files("gateway").joinpath("data", "mcp_logos", file_name).read_bytes()
        cached = make_logo(content, "image/svg+xml")
        _bundled_cache[name] = cached
    return cached
