import { createFileRoute, Outlet } from "@tanstack/react-router"

// A layout, not a page: /tools has child routes now (one per service), so the
// full page lives in tools.index.tsx and this only nests them.
export const Route = createFileRoute("/tools")({
  component: Outlet,
})
