import { createFileRoute } from "@tanstack/react-router";

import { BudgetsPage } from "@/features/budgets/BudgetsPage";

export const Route = createFileRoute("/budgets")({
  component: BudgetsPage,
});
