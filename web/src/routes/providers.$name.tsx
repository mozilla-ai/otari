import { createFileRoute } from "@tanstack/react-router"

import { ProviderDetailPage } from "@/features/providers/ProviderDetailPage"

// The page takes the name as a prop rather than reading the router itself, so
// a test can hand it one without a route tree.
function SelectedProvider() {
  const { name } = Route.useParams()
  return <ProviderDetailPage name={name} />
}

export const Route = createFileRoute("/providers/$name")({
  component: SelectedProvider,
})
