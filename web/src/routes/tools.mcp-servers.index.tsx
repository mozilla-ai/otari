import { createFileRoute } from "@tanstack/react-router"

import { McpServersPage } from "@/features/tools/McpServersPage"
import { ToolsFrame } from "@/features/tools/ToolsFrame"

export const Route = createFileRoute("/tools/mcp-servers/")({
  component: () => (
    <ToolsFrame current="/tools/mcp-servers">
      <McpServersPage />
    </ToolsFrame>
  ),
})
