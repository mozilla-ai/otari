import { createFileRoute } from "@tanstack/react-router"

import { RulePage } from "@/features/rules/RulePage"

function SelectedRule() {
  const { ruleId } = Route.useParams()
  return <RulePage ruleId={ruleId} />
}

export const Route = createFileRoute("/rules/$ruleId")({
  component: SelectedRule,
})
