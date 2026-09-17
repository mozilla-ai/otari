import { createFileRoute } from "@tanstack/react-router"

import { GuardrailsPage } from "@/features/tools/GuardrailsPage"

export const Route = createFileRoute("/tools/guardrails")({
  component: GuardrailsPage,
})
