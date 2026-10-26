import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/admin/web-search")({
  // Moved under /deployment with the rest of the deployment's own pages. The
  // old path redirects so bookmarks and links in sent mail keep working.
  beforeLoad: () => {
    throw redirect({
      to: "/deployment/web-search-pricing",
      search: true,
      replace: true,
    })
  },
})
