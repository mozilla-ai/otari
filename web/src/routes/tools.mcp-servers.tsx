import { createFileRoute } from "@tanstack/react-router"

import { McpServersPage } from "@/features/tools/McpServersPage"

export const Route = createFileRoute("/tools/mcp-servers")({
  component: McpServersPage,
})
