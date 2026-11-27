import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/tools/guardrails")({
  // The workspace's guardrails are the Content moderation band of Guardrails
  // now. A link to the old page still lands there, with no history entry.
  beforeLoad: () => {
    throw redirect({ to: "/guardrails", replace: true })
  },
})
