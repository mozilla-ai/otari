import { createFileRoute } from "@tanstack/react-router"

import { ModelDetailPage } from "@/features/models/ModelDetailPage"

// The model is the route, so the page takes it as a prop rather than reading
// the router itself: a test can hand it an id without a route tree.
function SelectedModel() {
  const { modelId } = Route.useParams()
  return <ModelDetailPage modelId={modelId} />
}

export const Route = createFileRoute("/models/$modelId")({
  component: SelectedModel,
})
