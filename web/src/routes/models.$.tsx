import { createFileRoute } from "@tanstack/react-router"

import { ModelDetailPage } from "@/features/models/ModelDetailPage"

// A splat rather than a single segment: a model's id carries its vendor,
// `z-ai/glm-5.3`, so the path under /models has a slash in it. The page takes
// the id as a prop rather than reading the router itself, so a test can hand
// it one without a route tree.
function SelectedModel() {
  const { _splat } = Route.useParams()
  return <ModelDetailPage modelId={_splat ?? ""} />
}

export const Route = createFileRoute("/models/$")({
  component: SelectedModel,
})
