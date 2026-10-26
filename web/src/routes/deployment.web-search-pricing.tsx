import { createFileRoute } from "@tanstack/react-router"

import { ToolPricingPage } from "@/features/toolPricing/ToolPricingPage"

export const Route = createFileRoute("/deployment/web-search-pricing")({
  component: () => <ToolPricingPage service="web_search" />,
})
