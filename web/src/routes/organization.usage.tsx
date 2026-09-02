import { createFileRoute } from "@tanstack/react-router"

import { UsagePage } from "@/features/usage/UsagePage"

export const Route = createFileRoute("/organization/usage")({
  component: () => <UsagePage scope="organization" />,
})
