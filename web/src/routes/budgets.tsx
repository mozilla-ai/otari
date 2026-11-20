import { createFileRoute } from "@tanstack/react-router"

import { SpendAndBudgetsPage } from "@/features/budgets/OrganizationBudgetsPage"

export const Route = createFileRoute("/budgets")({
  component: SpendAndBudgetsPage,
})
