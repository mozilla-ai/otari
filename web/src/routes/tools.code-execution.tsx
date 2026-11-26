import { createFileRoute } from "@tanstack/react-router"

import { ToolsFrame } from "@/features/tools/ToolsFrame"
import { ToolsGuardrailsPage } from "@/features/tools/ToolsGuardrailsPage"

export const Route = createFileRoute("/tools/code-execution")({
  component: () => (
    <ToolsFrame current="/tools/code-execution">
      <ToolsGuardrailsPage only="sandbox" />
    </ToolsFrame>
  ),
})
