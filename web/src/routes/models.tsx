import { createFileRoute, Outlet } from "@tanstack/react-router"

// A layout, not a page: the list lives in models.index.tsx and a model's own
// page in models.$modelId.tsx; this only nests them under one path.
export const Route = createFileRoute("/models")({
  component: Outlet,
})
