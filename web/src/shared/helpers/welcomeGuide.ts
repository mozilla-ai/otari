import type { DeploymentBootstrap } from "@/client"

/**
 * Where the welcome guide is, on a deployment that serves one.
 *
 * `/welcome` is a route on the gateway process (`src/gateway/main.py`) rather
 * than a client route, so it answers on the origin that process serves and
 * nowhere else. In standalone and hybrid that is the dashboard's own origin,
 * because the same process serves the bundle. On hosted it is not: otari.ai
 * builds the bundle under a path prefix and publishes it as static files from
 * its own edge, so a root-absolute `/welcome` leaves the app for an origin that
 * never had the route (otari#1248).
 *
 * Routing the path through at that edge would answer the 404 without fixing
 * anything, which is why this is a link that disappears rather than a path that
 * moves. The page is the self-hosted quickstart, and it opens by telling the
 * reader to copy a bootstrap API key out of the server's logs. A tenant of a
 * hosted deployment has neither the logs nor the key, so the honest answer is
 * the one `bootstrap.ts` already gives for a fact this dashboard cannot
 * establish: offer nothing, claim nothing, link nowhere.
 *
 * Undefined rather than `""`, so a caller cannot accidentally build a link to
 * the page it is already on. `resolveSnippetBaseUrl` withholds its base for the
 * same reason.
 */
export function welcomeGuideHref(
  deployment: Pick<DeploymentBootstrap, "deployment_type">,
): string | undefined {
  return deployment.deployment_type === "hosted" ? undefined : "/welcome"
}
