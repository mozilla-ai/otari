import { createFileRoute } from "@tanstack/react-router"

import { McpServerDetailPage } from "@/features/tools/McpServerDetailPage"

// The page takes the id as a prop rather than reading the router itself, so a
// test can hand it one without a route tree.
function SelectedServer() {
  const { serverId } = Route.useParams()
  const { workspace } = Route.useSearch()
  return <McpServerDetailPage serverId={serverId} workspaceId={workspace} />
}

export const Route = createFileRoute("/tools/mcp-servers/$serverId")({
  // A trace links here with the workspace its call ran in, which need not be
  // the one the sidebar has selected. Only one id is kept; anything else is
  // dropped, and the page reads the selected workspace instead.
  validateSearch: (
    search: Record<string, unknown>,
  ): { workspace?: string } => ({
    workspace:
      typeof search.workspace === "string" && search.workspace !== ""
        ? search.workspace
        : undefined,
  }),
  component: SelectedServer,
})
