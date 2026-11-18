import { createFileRoute, Outlet } from "@tanstack/react-router"

// A layout, not a page: the list lives in admin.hosted-providers.index.tsx and
// a provider's own page in admin.hosted-providers.$name.tsx.
export const Route = createFileRoute("/admin/hosted-providers")({
  component: Outlet,
})
