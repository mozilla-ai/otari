import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/organization/pricing")({
  // Model pricing folded into Providers, where an organization's models are
  // offered, priced and switched in one place. The old path redirects so
  // bookmarks and any link still pointing at it keep working, and `search` is
  // forwarded because both of this page's parameters survive the move: `?model=`
  // still opens the deployment rate editor and `?override=` still opens the
  // organization's own. In `beforeLoad`, so the page it replaces is never
  // mounted and the dead path leaves no history entry.
  beforeLoad: ({ search }) => {
    throw redirect({
      to: "/organization/provider-keys",
      search,
      replace: true,
    })
  },
})
