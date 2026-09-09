import { createFileRoute } from "@tanstack/react-router"

import { ModelCatalogPage } from "@/features/models/ModelCatalogPage"

// The selected model is the route, so the page takes it as a prop rather than
// reading the router itself: the same component renders `/models` with nothing
// selected, and a test can hand it an id without a route tree.
function SelectedModel() {
  const { modelId } = Route.useParams()
  return <ModelCatalogPage modelId={modelId} />
}

export const Route = createFileRoute("/models/$modelId")({
  component: SelectedModel,
})
