# Recorded MCP Registry pages

Entries copied unchanged from `GET https://registry.modelcontextprotocol.io/v0.1/servers?version=latest&limit=100`
on 2026-11-10, cut down to nine and split over two pages so a test can follow a cursor.
`tests/unit/test_mcp_catalog_registry.py` and `tests/integration/test_mcp_catalog.py` read them.

They cover what the sync has to tell apart: a server with a title and one without, an
optional and a required `Authorization` header, an optional header of another name, a
server with two remotes, a package with no remote, and a deprecated entry.
