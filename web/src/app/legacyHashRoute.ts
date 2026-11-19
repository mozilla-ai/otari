/**
 * Moves a link from before path routing (`/#/keys`, `/#/verify-email?token=…`)
 * to the path it names. The server never sees a fragment, so only the page can
 * do this. `replaceState` rather than a navigation: a mailed token goes from
 * one fragment to another without a request carrying it. A link's secret stays
 * in the fragment, as `/verify-email#token=…`, where the gateway now puts it;
 * anything else's query joins the page's own. An OAuth callback's route was
 * renamed with the move and is renamed here.
 *
 * A module rather than an inline script in `index.html`: a deployment whose
 * Content-Security-Policy allows only its own scripts and listed hashes (as
 * otari.ai's does) runs it without a hash to keep in step with its text.
 */

interface Page {
  location: Pick<Location, "pathname" | "search" | "hash">
  history: Pick<History, "replaceState">
  addEventListener(type: "hashchange", listener: () => void): void
}

const SECRET_ROUTES = /^(setup|verify-email|reset-password|accept-invitation)$/

export function moveLegacyHashRoute(page: Page, base: string): void {
  const { hash, pathname } = page.location
  if (!hash.startsWith("#/")) return
  const route = hash.slice(2)
  const queryAt = route.indexOf("?")
  const query = queryAt === -1 ? "" : route.slice(queryAt + 1)
  const path = (queryAt === -1 ? route : route.slice(0, queryAt)).replace(
    /^auth\/([^/]+)\/callback$/,
    "sign-in/$1/callback",
  )
  const secret = SECRET_ROUTES.test(path)
  // The page's own query belongs to the link only where the link was written
  // against the dashboard's root, as every old link was.
  const atRoot =
    `${pathname}/`.startsWith(base) && pathname.length <= base.length
  let search = atRoot ? page.location.search.slice(1) : ""
  if (query && !secret) search = search ? `${search}&${query}` : query
  page.history.replaceState(
    null,
    "",
    `${base}${path}${search ? `?${search}` : ""}${query && secret ? `#${query}` : ""}`,
  )
}

/**
 * Moves the address now and again on `hashchange`, because an old link opened
 * in a tab already on the dashboard changes only the fragment and loads nothing.
 */
export function installLegacyHashRoute(page: Page, base: string): void {
  moveLegacyHashRoute(page, base)
  page.addEventListener("hashchange", () => moveLegacyHashRoute(page, base))
}
