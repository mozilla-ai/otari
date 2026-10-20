import { createFileRoute } from "@tanstack/react-router"
import { HealthChecksPage } from "@/features/health/HealthChecksPage"

export const Route = createFileRoute("/health-checks")({
  component: HealthChecksPage,
})
