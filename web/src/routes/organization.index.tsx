import { createFileRoute } from "@tanstack/react-router"

import { OrganizationGeneralPage } from "@/features/organization/OrganizationGeneralPage"

export const Route = createFileRoute("/organization/")({
  component: OrganizationGeneralPage,
})
