import { createBrowserHistory, createRouter } from "@tanstack/react-router"
import { PendingPage } from "@/app/PendingPage"
import { PageError } from "@/design-system/feedback/PageError"
import { routeTree } from "@/routeTree.gen"
import { BASE_PATH } from "@/shared/helpers/appLocation"
import {
  type DashboardSearch,
  parseSearch,
  stringifySearch,
} from "@/shared/helpers/search"

export const router = createRouter({
  routeTree,
  // Path routing under the base the bundle was built for. The gateway answers
  // a browser opening any path it has no route for with this page
  // (`gateway/dashboard.py`), and only once every route has declined it, so a
  // dashboard path never shadows the API or the assets. A link from before
  // path routing (`/#/keys`) is rewritten by `legacyHashRouteOnLoad`, which
  // `main.tsx` imports ahead of this module.
  history: createBrowserHistory(),
  basepath: BASE_PATH || "/",
  parseSearch,
  stringifySearch,
  // Each page is its own chunk, so the first visit to one waits on a fetch.
  // Show the loader as soon as that wait starts and drop it the moment the chunk
  // lands: the defaults hold it back a second and then keep it up for half of
  // one, which turns an instant navigation into a visible stall.
  defaultPendingComponent: PendingPage,
  defaultPendingMs: 0,
  defaultPendingMinMs: 0,
  // Without this a routed throw lands on TanStack's built-in component, which
  // paints its own inline-styled box and a red `<pre>` of the raw error: the one
  // thing `design/feedback.md` says never to render. `router.test.tsx` covers
  // why setting it is what makes a boundary catch the throw at all.
  defaultErrorComponent: ({ error }) => (
    // No mention of the sidebar: this serves the root match too, and a throw in
    // `AppShell` itself replaces the shell the sidebar lives in.
    <PageError error={error}>
      This page could not finish rendering. Reload to try again.
    </PageError>
  ),
})

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
  interface HistoryState {
    /** The Models list's search a model was opened from, for its link back. */
    catalogSearch?: DashboardSearch
  }
}
