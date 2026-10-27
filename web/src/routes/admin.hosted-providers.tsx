import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/admin/hosted-providers")({
  // Moved under /deployment with the rest of the deployment's own pages. The
  // old path redirects so bookmarks and links in sent mail keep working.
  beforeLoad: () => {
    throw redirect({
      to: "/deployment/hosted-providers",
      search: true,
      replace: true,
    })
  },
})
