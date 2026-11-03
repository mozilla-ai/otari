import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/activity")({
  // Retired for the Requests tab of the Overview. The redirect keeps the whole query
  // string, so a bookmark or a link still pointing here opens the same filtered
  // view. In `beforeLoad`, so the old page is never mounted and the dead path
  // leaves no history entry.
  beforeLoad: ({ search }) => {
    throw redirect({
      to: "/requests",
      search,
      replace: true,
    })
  },
})
