import { createFileRoute } from "@tanstack/react-router"

import { OrganizationGuardrailsPage } from "@/features/guardrails/OrganizationGuardrailsPage"

export const Route = createFileRoute("/organization/guardrails")({
  component: OrganizationGuardrailsPage,
})
