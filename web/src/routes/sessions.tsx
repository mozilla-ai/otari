import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/sessions")({
  // The old path of Traces. The redirect keeps the whole query string, so a
  // bookmark or a shared link still opens the same filtered list or the same
  // open session. In `beforeLoad`, so the dead path is never mounted and
  // leaves no history entry.
  beforeLoad: ({ search }) => {
    throw redirect({
      to: "/traces",
      search,
      replace: true,
    })
  },
})
