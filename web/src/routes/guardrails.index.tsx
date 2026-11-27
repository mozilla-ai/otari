import { createFileRoute } from "@tanstack/react-router"

import { GuardrailsPage } from "@/features/rules/GuardrailsPage"

export const Route = createFileRoute("/guardrails/")({
  component: GuardrailsPage,
})
