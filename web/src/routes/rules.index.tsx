import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/rules/")({
  // The old path of Guardrails. Its `type` picked one of the tabs Rules had
  // (Spend, Access, Data), which are now bands on one view, so it is dropped
  // and the rest of the query travels. Leaves no history entry.
  beforeLoad: ({ search }) => {
    const { type: _type, ...rest } = search
    throw redirect({ to: "/guardrails", search: rest, replace: true })
  },
})
