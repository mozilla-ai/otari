"""The Popular shelf: well-known remote MCP servers the catalog opens on before anyone searches.

HAND-MAINTAINED. Nothing generates or refreshes this list. Each entry is a vendor's own
server as the MCP Registry listed it on 2026-11-12: its registry name, and the first
streamable HTTP remote of that entry, which is the URL a sync stores. Granola's is not in
the registry and comes from ``_extras.py``. An entry is shown only
while the catalog holds a server at that URL, so one whose vendor moves it drops off the
shelf rather than pointing anywhere new; check the registry and update the URL here.

Left out because the registry listed no server of the vendor's own on that date, only
third-party proxies: Asana, Intercom, Square.
"""

from typing import NamedTuple


class PopularServer(NamedTuple):
    """One shelf entry: the registry name it is listed under, and the URL it is matched by."""

    name: str
    url: str


POPULAR_SERVERS: tuple[PopularServer, ...] = (
    PopularServer("io.github.github/github-mcp-server", "https://api.githubcopilot.com/mcp/"),
    PopularServer("com.notion/mcp", "https://mcp.notion.com/mcp"),
    PopularServer("app.linear/linear", "https://mcp.linear.app/mcp"),
    PopularServer("io.github.getsentry/sentry-mcp", "https://mcp.sentry.dev/mcp"),
    PopularServer("com.atlassian/atlassian-mcp-server", "https://mcp.atlassian.com/v2/mcp"),
    PopularServer("com.figma.mcp/mcp", "https://mcp.figma.com/mcp"),
    PopularServer("com.stripe/mcp", "https://mcp.stripe.com"),
    PopularServer("com.vercel/vercel-mcp", "https://mcp.vercel.com"),
    PopularServer("com.supabase/mcp", "https://mcp.supabase.com/mcp"),
    PopularServer("com.neon/mcp", "https://mcp.neon.tech/mcp"),
    PopularServer("com.gitlab/mcp", "https://gitlab.com/api/v4/mcp"),
    PopularServer("io.github.upstash/context7", "https://mcp.context7.com/mcp"),
    PopularServer("co.huggingface/hf-mcp-server", "https://huggingface.co/mcp?login"),
    PopularServer("com.cloudflare.mcp/mcp", "https://docs.mcp.cloudflare.com/mcp"),
    PopularServer("io.github.grafana/mcp-grafana", "https://mcp.grafana.com/mcp"),
    PopularServer("io.github.PostHog/mcp", "https://mcp.posthog.com/mcp"),
    PopularServer("com.canva.mcp/mcp", "https://mcp.canva.com/mcp"),
    PopularServer("com.monday/monday.com", "https://mcp.monday.com/mcp"),
    PopularServer("com.paypal.mcp/mcp", "https://mcp.paypal.com/mcp"),
    PopularServer("com.zapier/mcp", "https://mcp.zapier.com/api/v1/connect"),
    PopularServer("ai.granola/mcp", "https://mcp.granola.ai/mcp"),
)
