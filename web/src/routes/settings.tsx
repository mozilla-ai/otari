import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/settings")({
  // Moved under /deployment with the rest of the deployment's own pages. The
  // old path redirects so bookmarks and links in sent mail keep working.
  beforeLoad: () => {
    throw redirect({ to: "/deployment/settings", search: true, replace: true })
  },
})
