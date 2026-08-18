import { createFileRoute, Outlet } from "@tanstack/react-router"

// A layout with nothing of its own: the organization surface is two pages
// (General and Members) under one path, so this exists to hold the segment and
// hand the URL to whichever of them matched.
export const Route = createFileRoute("/organization")({
  component: Outlet,
})
