import { createFileRoute } from "@tanstack/react-router"

import { ToolPricingPage } from "@/features/toolPricing/ToolPricingPage"

export const Route = createFileRoute("/deployment/code-execution-pricing")({
  component: () => <ToolPricingPage service="code_execution" />,
})
