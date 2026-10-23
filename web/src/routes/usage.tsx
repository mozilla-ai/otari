import { createFileRoute, redirect } from "@tanstack/react-router"

import { legacySearch } from "@/features/observe/observeSearch"

export const Route = createFileRoute("/usage")({
  // Retired for the Usage tab of the Overview. The redirect keeps the whole
  // query string, so a bookmark or a link still pointing here opens the same
  // filtered view. In `beforeLoad`, so the old page is never mounted and the dead path
  // leaves no history entry. A link naming a user opens on Everyone, where that
  // filter applies (`legacySearch`).
  beforeLoad: ({ search }) => {
    throw redirect({
      to: "/",
      search: legacySearch(search),
      replace: true,
    })
  },
})
