import { createFileRoute } from "@tanstack/react-router"

import { OrganizationUsagePage } from "@/features/usage/OrganizationUsagePage"

export const Route = createFileRoute("/organization/usage")({
  component: OrganizationUsagePage,
})
