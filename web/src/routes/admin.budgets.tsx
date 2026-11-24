import { createFileRoute } from "@tanstack/react-router"

import { DeploymentBudgetsPage } from "@/features/budgets/DeploymentBudgetsPage"

export const Route = createFileRoute("/admin/budgets")({
  component: DeploymentBudgetsPage,
})
