import { createFileRoute } from "@tanstack/react-router"

import { McpServerDetailPage } from "@/features/tools/McpServerDetailPage"

// The page takes the id as a prop rather than reading the router itself, so a
// test can hand it one without a route tree.
function SelectedServer() {
  const { serverId } = Route.useParams()
  return <McpServerDetailPage serverId={serverId} />
}

export const Route = createFileRoute("/tools/mcp-servers/$serverId")({
  component: SelectedServer,
})
