import { createFileRoute } from "@tanstack/react-router"

import { OrganizationMcpServersPage } from "@/features/organization/mcpServers/OrganizationMcpServersPage"

export const Route = createFileRoute("/organization/mcp-servers/")({
  component: OrganizationMcpServersPage,
})
