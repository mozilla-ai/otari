import { createFileRoute } from "@tanstack/react-router"

import { McpServersPage } from "@/features/tools/McpServersPage"

export const Route = createFileRoute("/mcp-servers/")({
  component: McpServersPage,
})
