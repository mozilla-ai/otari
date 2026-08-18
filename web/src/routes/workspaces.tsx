import { createFileRoute } from "@tanstack/react-router"

import { WorkspacesPage } from "@/features/workspaces/WorkspacesPage"

export const Route = createFileRoute("/workspaces")({
  component: WorkspacesPage,
})
