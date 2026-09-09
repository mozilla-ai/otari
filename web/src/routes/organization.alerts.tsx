import { createFileRoute } from "@tanstack/react-router"

import { OrganizationAlertsPage } from "@/features/organization/OrganizationAlertsPage"

export const Route = createFileRoute("/organization/alerts")({
  component: OrganizationAlertsPage,
})
