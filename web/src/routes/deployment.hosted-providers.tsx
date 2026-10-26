import { createFileRoute } from "@tanstack/react-router"

import { HostedProvidersPage } from "@/features/hostedProviders/HostedProvidersPage"

export const Route = createFileRoute("/deployment/hosted-providers")({
  component: HostedProvidersPage,
})
