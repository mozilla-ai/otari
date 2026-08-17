import { createHashHistory, createRouter } from "@tanstack/react-router"
import { routeTree } from "@/routeTree.gen"
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
  defaultPendingComponent: () => <div role="status">Loading page…</div>,
  defaultPendingMs: 0,
  defaultPendingMinMs: 0,
})

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
}
