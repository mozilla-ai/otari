import { createFileRoute } from "@tanstack/react-router"

import { RulePage } from "@/features/rules/RulePage"

function SelectedGuardrail() {
  const { id } = Route.useParams()
  return <RulePage ruleId={id} />
}

export const Route = createFileRoute("/guardrails/$id")({
  component: SelectedGuardrail,
})
