import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/aliases")({
  // Aliases were folded into Routing, which lists and manages them as the
  // one-target policies they are. The old path redirects so bookmarks and any
  // link still pointing at it keep working. In `beforeLoad`, so the page it
  // replaces is never mounted and the dead path leaves no history entry.
  beforeLoad: () => {
    throw redirect({ to: "/routing", replace: true })
  },
})
