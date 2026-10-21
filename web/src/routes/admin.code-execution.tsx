import { createFileRoute } from "@tanstack/react-router"

import { ToolPricingPage } from "@/features/toolPricing/ToolPricingPage"

export const Route = createFileRoute("/admin/code-execution")({
  component: () => <ToolPricingPage service="code_execution" />,
})
