import { createFileRoute, Outlet } from "@tanstack/react-router"

// A layout, not a page: the list lives in budgets.index.tsx and one budget in
// budgets.$budgetId.tsx; this only nests them under one path.
export const Route = createFileRoute("/budgets")({
  component: Outlet,
})
