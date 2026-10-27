import { createFileRoute } from "@tanstack/react-router"

import { ActivityPage } from "@/features/activity/ActivityPage"

export const Route = createFileRoute("/deployment/activity")({
  component: ActivityPage,
})
