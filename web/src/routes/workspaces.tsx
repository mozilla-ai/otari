import { createFileRoute, Outlet } from "@tanstack/react-router"

// A layout, not a page: the list lives in workspaces.index.tsx and a
// workspace's own page in workspaces.$workspaceId.tsx; this only nests them
// under one path.
export const Route = createFileRoute("/workspaces")({
  component: Outlet,
})
