import { createFileRoute } from "@tanstack/react-router"

import { OrganizationDomainsPage } from "@/features/organization/OrganizationDomainsPage"

export const Route = createFileRoute("/organization/domains")({
  component: OrganizationDomainsPage,
})
