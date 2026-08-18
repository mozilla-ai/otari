import { createFileRoute } from "@tanstack/react-router"

import { WorkspaceMembersPage } from "@/features/workspaces/WorkspaceMembersPage"

export const Route = createFileRoute("/members")({
  component: WorkspaceMembersPage,
})
