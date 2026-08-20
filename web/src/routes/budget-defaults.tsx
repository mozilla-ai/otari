import { createFileRoute } from "@tanstack/react-router"

import { WorkspaceBudgetDefaultsPage } from "@/features/workspaces/WorkspaceBudgetDefaultsPage"

export const Route = createFileRoute("/budget-defaults")({
  component: WorkspaceBudgetDefaultsPage,
})
