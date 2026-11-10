import { createFileRoute } from "@tanstack/react-router"

import { McpCatalogPage } from "@/features/organization/mcpCatalog/McpCatalogPage"

export const Route = createFileRoute("/organization/mcp-catalog")({
  component: McpCatalogPage,
})
