"""Well-known remote MCP servers the MCP Registry does not list, which the catalog lists anyway.

HAND-MAINTAINED. Nothing generates or refreshes this list. Each entry is a maker's own
remote server, confirmed against the maker's own documentation and by asking the endpoint
on the date beside it. A sync stores the list each time it starts, so an entry removed
here leaves the catalog at the next complete sync. One the registry starts listing, under
the same name or at the same URL, is dropped in favor of the registry's: remove it here.

The name follows the registry's reverse-DNS form for the maker's domain, which only the
maker can publish under, so nobody else's entry can take its place.
"""

from typing import NamedTuple


class ExtraServer(NamedTuple):
    """One server as the catalog lists it."""

    name: str
    title: str
    description: str
    url: str
    website_url: str
    icon_url: str | None = None


EXTRA_SERVERS: tuple[ExtraServer, ...] = (
    # Granola's docs (docs.granola.ai/help-center/sharing/integrations/mcp) give this URL, and it answers
    # with an OAuth challenge whose authorization server is mcp-auth.granola.ai. Checked 2026-11-12, when
    # the registry listed only an unrelated local package under Granola's name.
    ExtraServer(
        name="ai.granola/mcp",
        title="Granola",
        description="Search and read your Granola meeting notes.",
        url="https://mcp.granola.ai/mcp",
        website_url="https://www.granola.ai",
        icon_url="https://www.granola.ai/favicon/apple-touch-icon.png",
    ),
)
