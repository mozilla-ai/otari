import { createFileRoute, Outlet } from "@tanstack/react-router"

// A layout, not a page: the list lives in providers.index.tsx and a provider's
// own page in providers.$name.tsx; this only nests them under one path.
export const Route = createFileRoute("/providers")({
  component: Outlet,
})
