import { createFileRoute } from "@tanstack/react-router"

import { PendingInvitationsPage } from "@/features/invitations/PendingInvitationsPage"

export const Route = createFileRoute("/invitations")({
  component: PendingInvitationsPage,
})
