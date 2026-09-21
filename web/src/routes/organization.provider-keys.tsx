import { createFileRoute } from "@tanstack/react-router"

import { OrganizationProvidersPage } from "@/features/organization/providers/OrganizationProvidersPage"

export const Route = createFileRoute("/organization/provider-keys")({
  component: OrganizationProvidersPage,
})
