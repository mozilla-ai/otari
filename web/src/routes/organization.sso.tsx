import { createFileRoute } from "@tanstack/react-router"

import { OrganizationSsoPage } from "@/features/organization/OrganizationSsoPage"

export const Route = createFileRoute("/organization/sso")({
  component: OrganizationSsoPage,
})
