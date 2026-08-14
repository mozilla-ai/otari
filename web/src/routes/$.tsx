import { createFileRoute, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/$")({
  // A splat over every unmatched path, which sends an unknown route to the
  // overview rather than rendering a not-found page. The dashboard has no
  // user-authored URLs, so an unrecognised one is a stale bookmark from a route
  // that has since moved, and the overview is where it can be found again.
  beforeLoad: () => {
    throw redirect({ to: "/", replace: true });
  },
});
