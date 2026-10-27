import { createFileRoute } from "@tanstack/react-router"

import { DeploymentBudgetsPage } from "@/features/budgets/BudgetsPage"

export const Route = createFileRoute("/deployment/budgets")({
  component: DeploymentBudgetsPage,
})
