import { createFileRoute } from "@tanstack/react-router"

import { ActivityHub } from "@/features/traces/ActivityHub"

export const Route = createFileRoute("/activity")({
  component: ActivityHub,
})
