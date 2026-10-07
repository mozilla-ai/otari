import { createFileRoute } from "@tanstack/react-router"

import { TraceContentPage } from "@/features/traces/TraceContentPage"

export const Route = createFileRoute("/tools/trace-content")({
  component: TraceContentPage,
})
