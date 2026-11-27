import { createFileRoute } from "@tanstack/react-router"

import { ToolsFrame } from "@/features/tools/ToolsFrame"
import { ToolsGuardrailsPage } from "@/features/tools/ToolsGuardrailsPage"

export const Route = createFileRoute("/tools/web-search")({
  component: () => (
    <ToolsFrame back>
      <ToolsGuardrailsPage only="web_search" />
    </ToolsFrame>
  ),
})
