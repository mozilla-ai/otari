import { createFileRoute, Outlet } from "@tanstack/react-router"

// A layout, not a page: the list lives in mcp-servers.index.tsx and a
// server's own page in mcp-servers.$serverId.tsx; this only nests them.
export const Route = createFileRoute("/mcp-servers")({
  component: Outlet,
})
