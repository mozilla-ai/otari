import { createFileRoute } from "@tanstack/react-router"

import { OrganizationProviderKeysPage } from "@/features/organization/OrganizationProviderKeysPage"

export const Route = createFileRoute("/organization/provider-keys")({
  component: OrganizationProviderKeysPage,
})
