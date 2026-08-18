import { createFileRoute } from "@tanstack/react-router"

import { OrganizationMembersPage } from "@/features/organization/OrganizationMembersPage"

export const Route = createFileRoute("/organization/members")({
  component: OrganizationMembersPage,
})
