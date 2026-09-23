import { createFileRoute } from "@tanstack/react-router"

import { WorkspaceGuardrailsPage } from "@/features/guardrails/WorkspaceGuardrailsPage"

export const Route = createFileRoute("/tools/guardrails")({
  component: WorkspaceGuardrailsPage,
})
