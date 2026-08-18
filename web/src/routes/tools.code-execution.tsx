import { createFileRoute } from "@tanstack/react-router"

import { ToolsGuardrailsPage } from "@/features/tools/ToolsGuardrailsPage"

export const Route = createFileRoute("/tools/code-execution")({
  component: () => <ToolsGuardrailsPage only="sandbox" />,
})
