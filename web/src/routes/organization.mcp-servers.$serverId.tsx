import { createFileRoute } from "@tanstack/react-router"

import { OrganizationMcpServerDetailPage } from "@/features/organization/mcpServers/OrganizationMcpServerDetailPage"

// The page takes the id as a prop rather than reading the router itself, so a
// test can hand it one without a route tree. Keyed on it, so moving to another
// server starts the page afresh and no dialog's draft carries across.
function SelectedServer() {
  const { serverId } = Route.useParams()
  return <OrganizationMcpServerDetailPage key={serverId} serverId={serverId} />
}

export const Route = createFileRoute("/organization/mcp-servers/$serverId")({
  component: SelectedServer,
})
