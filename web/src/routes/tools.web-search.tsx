import { createFileRoute } from "@tanstack/react-router"

import { ToolsFrame } from "@/features/tools/ToolsFrame"
import { ToolsGuardrailsPage } from "@/features/tools/ToolsGuardrailsPage"

export const Route = createFileRoute("/tools/web-search")({
  component: () => (
    <ToolsFrame current="/tools/web-search">
      <ToolsGuardrailsPage only="web_search" />
    </ToolsFrame>
  ),
})
