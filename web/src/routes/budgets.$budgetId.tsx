import { createFileRoute } from "@tanstack/react-router"

import { BudgetDetailPage } from "@/features/budgets/BudgetDetailPage"

// The page takes the id as a prop rather than reading the router itself, so a
// test can hand it one without a route tree.
function SelectedBudget() {
  const { budgetId } = Route.useParams()
  return <BudgetDetailPage budgetId={budgetId} />
}

export const Route = createFileRoute("/budgets/$budgetId")({
  component: SelectedBudget,
})
