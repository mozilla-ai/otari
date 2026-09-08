import { createHashHistory, createRouter } from "@tanstack/react-router"
import { PendingPage } from "@/app/PendingPage"
import { routeTree } from "@/routeTree.gen"
import { PageError } from "@/shared/components/feedback/PageError"
import { parseSearch, stringifySearch } from "@/shared/helpers/search"

export const router = createRouter({
  routeTree,
  // Hash routing, as before the TanStack migration: the gateway serves this
  // dashboard from the same origin as its API and hashed assets, so client
  // routes live under `/#/...` and need no server catch-all that could shadow
  // `/v1` or `/assets`.
  history: createHashHistory(),
  parseSearch,
  stringifySearch,
  // Each page is its own chunk, so the first visit to one waits on a fetch.
  // Show the loader as soon as that wait starts and drop it the moment the chunk
  // lands: the defaults hold it back a second and then keep it up for half of
  // one, which turns an instant navigation into a visible stall.
  defaultPendingComponent: PendingPage,
  defaultPendingMs: 0,
  defaultPendingMinMs: 0,
  // The router wraps its whole match tree in a catch boundary already, so a
  // throw inside the shell was never a blank page. What it fell back to was
  // TanStack's built-in `ErrorComponent`, which paints its own inline-styled
  // box and a red `<pre>` of the raw error: outside the design system, and the
  // one thing `feedback.md` says never to render. Same panel as the boundary
  // above the router now, so the two failures look like one product.
  defaultErrorComponent: ({ error }) => (
    <PageError error={error}>
      This page could not finish rendering. Try another destination from the
      sidebar, or reload.
    </PageError>
  ),
})

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
}
