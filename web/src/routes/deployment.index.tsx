import { createFileRoute } from "@tanstack/react-router"

import { DeploymentOverview } from "@/features/overview/OverviewPage"

export const Route = createFileRoute("/deployment/")({
  component: DeploymentOverview,
})
