import { createFileRoute } from "@tanstack/react-router"

import { AgentRecommendersPage } from "@/features/agentRecommenders/AgentRecommendersPage"

export const Route = createFileRoute("/admin/recommender")({
  component: AgentRecommendersPage,
})
