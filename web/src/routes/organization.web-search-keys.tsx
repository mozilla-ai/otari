import { createFileRoute } from "@tanstack/react-router"

import { OrganizationWebSearchKeysPage } from "@/features/organization/webSearchKeys/OrganizationWebSearchKeysPage"

export const Route = createFileRoute("/organization/web-search-keys")({
  component: OrganizationWebSearchKeysPage,
})
