import { createFileRoute } from "@tanstack/react-router"

import { HostedProviderPage } from "@/features/hostedProviders/HostedProviderPage"

// The page takes the name as a prop rather than reading the router itself, so
// a test can hand it one without a route tree.
function SelectedHostedProvider() {
  const { name } = Route.useParams()
  return <HostedProviderPage name={name} />
}

export const Route = createFileRoute("/admin/hosted-providers/$name")({
  component: SelectedHostedProvider,
})
