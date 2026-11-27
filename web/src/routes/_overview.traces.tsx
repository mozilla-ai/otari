import { createFileRoute } from "@tanstack/react-router"

import { SessionsPage } from "@/features/traces/SessionsPage"

export const Route = createFileRoute("/_overview/traces")({
  component: () => <SessionsPage embedded />,
})
