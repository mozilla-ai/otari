import { createFileRoute } from "@tanstack/react-router"

import { ToolPricingPage } from "@/features/toolPricing/ToolPricingPage"

export const Route = createFileRoute("/admin/web-search")({
  component: () => <ToolPricingPage service="web_search" />,
})
