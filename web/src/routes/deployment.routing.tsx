import { createFileRoute, Outlet } from "@tanstack/react-router"

// A layout, as /routing is: the list and a policy's own page nest under it.
export const Route = createFileRoute("/deployment/routing")({
  component: Outlet,
})
