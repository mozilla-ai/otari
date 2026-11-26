import { createFileRoute } from "@tanstack/react-router"

import { RulesPage } from "@/features/rules/RulesPage"

export const Route = createFileRoute("/rules/")({
  component: RulesPage,
})
