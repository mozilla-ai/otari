import { createFileRoute } from "@tanstack/react-router"

import { RoutingPage } from "@/features/routing/RoutingPage"

export const Route = createFileRoute("/deployment/routing/")({
  component: RoutingPage,
})
