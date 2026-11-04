import { createFileRoute } from "@tanstack/react-router"

import { WorkspaceDetailPage } from "@/features/workspaces/WorkspaceDetailPage"

// The page takes the id as a prop rather than reading the router itself, so a
// test can hand it one without a route tree.
function SelectedWorkspace() {
  const { workspaceId } = Route.useParams()
  return <WorkspaceDetailPage workspaceId={workspaceId} />
}

export const Route = createFileRoute("/workspaces/$workspaceId")({
  component: SelectedWorkspace,
})
