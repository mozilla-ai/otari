import { createFileRoute, Outlet } from "@tanstack/react-router"

// A layout, not a page: the list lives in tools.mcp-servers.index.tsx and a
// server's own page in tools.mcp-servers.$serverId.tsx; this only nests them.
export const Route = createFileRoute("/tools/mcp-servers")({
  component: Outlet,
})
