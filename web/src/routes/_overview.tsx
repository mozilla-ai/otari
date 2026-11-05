import { createFileRoute } from "@tanstack/react-router"

import { OverviewIndex } from "@/features/overview/OverviewPage"

// A pathless layout: the Overview's heading, attention strip and view tabs,
// with one child route per view (`/`, `/requests`, `/sessions`).
export const Route = createFileRoute("/_overview")({
  component: OverviewIndex,
})
