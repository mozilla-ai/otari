import { createFileRoute } from "@tanstack/react-router"

import { OverviewIndex } from "@/features/overview/OverviewPage"

// A pathless layout over Home (`/`) and the two views of Sessions
// (`/sessions`, `/requests`), which share a title and a tab row.
export const Route = createFileRoute("/_overview")({
  component: OverviewIndex,
})
