import { createFileRoute, Outlet } from "@tanstack/react-router"

// A layout, not a page: /models has a child per selected model now, so the page
// lives in models.index.tsx and models.$modelId.tsx and this only nests them.
export const Route = createFileRoute("/models")({
  component: Outlet,
})
