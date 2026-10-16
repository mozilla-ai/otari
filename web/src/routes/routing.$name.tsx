import { createFileRoute } from "@tanstack/react-router"

import { PolicyDetailPage } from "@/features/routing/PolicyDetailPage"

// The page takes the name as a prop rather than reading the router itself, so
// a test can hand it one without a route tree.
function SelectedPolicy() {
  const { name } = Route.useParams()
  return <PolicyDetailPage name={name} />
}

export const Route = createFileRoute("/routing/$name")({
  component: SelectedPolicy,
})
