import { createFileRoute } from "@tanstack/react-router"

import { ToolsFrame } from "@/features/tools/ToolsFrame"
import { ToolsGuardrailsPage } from "@/features/tools/ToolsGuardrailsPage"

export const Route = createFileRoute("/tools/code-execution")({
  component: () => (
    <ToolsFrame title="Code execution" back>
      <ToolsGuardrailsPage only="sandbox" />
    </ToolsFrame>
  ),
})
