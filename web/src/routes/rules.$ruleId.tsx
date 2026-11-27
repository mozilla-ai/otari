import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/rules/$ruleId")({
  // The old path of one guardrail's page. Ids did not change with the address.
  beforeLoad: ({ params, search }) => {
    throw redirect({
      to: "/guardrails/$id",
      params: { id: params.ruleId },
      search,
      replace: true,
    })
  },
})
