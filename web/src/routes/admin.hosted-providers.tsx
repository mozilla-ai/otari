import { createFileRoute } from "@tanstack/react-router"

import { HostedProvidersPage } from "@/features/hostedProviders/HostedProvidersPage"

export const Route = createFileRoute("/admin/hosted-providers")({
  component: HostedProvidersPage,
})
