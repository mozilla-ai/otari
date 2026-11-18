import { createFileRoute } from "@tanstack/react-router"

import { OrganizationMcpServerDetailPage } from "@/features/organization/mcpServers/OrganizationMcpServerDetailPage"

// The page takes the id as a prop rather than reading the router itself, so a
// test can hand it one without a route tree.
function SelectedServer() {
  const { serverId } = Route.useParams()
  return <OrganizationMcpServerDetailPage serverId={serverId} />
}

export const Route = createFileRoute("/organization/mcp-servers/$serverId")({
  component: SelectedServer,
})
