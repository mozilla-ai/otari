import { createFileRoute, redirect } from "@tanstack/react-router"

import { rendersAheadOfRouter } from "@/features/auth/publicAuthPaths"

export const Route = createFileRoute("/$")({
  // A splat over every unmatched path, which sends an unknown route to the
  // overview rather than rendering a not-found page. The dashboard has no
  // user-authored URLs, so an unrecognized one is a stale bookmark from a route
  // that has since moved, and the overview is where it can be found again.
  //
  // Except a page `DeploymentRoot` answers itself: an emailed link opened in a
  // tab that is already signed in reaches the router too, and a redirect would
  // rewrite the URL before `DeploymentRoot` reads it.
  beforeLoad: ({ location }) => {
    if (rendersAheadOfRouter(location.pathname)) {
      return
    }
    throw redirect({ to: "/", replace: true })
  },
})
