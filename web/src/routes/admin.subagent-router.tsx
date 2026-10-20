import { createFileRoute } from "@tanstack/react-router"

import { SubagentRouterPage } from "@/features/routing/SubagentRouterPage"

export const Route = createFileRoute("/admin/subagent-router")({
  component: SubagentRouterPage,
})
