import { createFileRoute, Outlet } from "@tanstack/react-router"

// A layout, not a page: the list lives in routing.index.tsx and a policy's own
// page in routing.$name.tsx; this only nests them under one path.
export const Route = createFileRoute("/routing")({
  component: Outlet,
})
