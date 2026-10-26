import { createFileRoute } from "@tanstack/react-router"

import { PolicyDetailPage } from "@/features/routing/PolicyDetailPage"

function SelectedPolicy() {
  const { name } = Route.useParams()
  return <PolicyDetailPage name={name} />
}

export const Route = createFileRoute("/deployment/routing/$name")({
  component: SelectedPolicy,
})
