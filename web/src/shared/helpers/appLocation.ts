/**
 * The dashboard's own address within the page's URL.
 *
 * The dashboard routes by path under the base it was built for
 * (`import.meta.env.BASE_URL`: `/` here, `/dashboard/` where a build serves it
 * beside another site), so a "route" below is that path with the base taken
 * off, plus its query and fragment: `/verify-email#token=…`,
 * `/models?provider=openai`.
 *
 * The pages in front of a session render ahead of the router (`App.tsx`'s
 * `DeploymentRoot`), so they read and change the location through here rather
 * than through router hooks they have no context for.
 */

/** Where this build is served, without a trailing slash: empty at an origin's root. */
export const BASE_PATH = import.meta.env.BASE_URL.replace(/\/+$/, "")

/** The route a location names: its path under the base, query and fragment included. */
export function currentRoute(
  location: Pick<Location, "pathname" | "search" | "hash"> = window.location,
): string {
  const { pathname } = location
  const path =
    BASE_PATH &&
    (pathname === BASE_PATH || pathname.startsWith(`${BASE_PATH}/`))
      ? pathname.slice(BASE_PATH.length) || "/"
      : pathname
  return `${path}${location.search}${location.hash}`
}

/** A route's path alone, without its query or fragment. */
export function routePath(route: string): string {
  return route.split(/[?#]/, 1)[0] || "/"
}

/**
 * A parameter a route carries, read from its fragment and then its query.
 *
 * The fragment first because that is where a link puts a secret: a browser
 * never sends it, so a token there reaches no server, access log or `Referer`
 * header. An empty value is malformed rather than present, and reads as null:
 * every caller branches on null to say "this link carries nothing to act on",
 * and an empty string would satisfy neither that branch nor the request it
 * gates, stranding the page on its loading line.
 */
export function routeParam(route: string, name: string): string | null {
  const hashAt = route.indexOf("#")
  const beforeHash = hashAt === -1 ? route : route.slice(0, hashAt)
  const queryAt = beforeHash.indexOf("?")
  const fromFragment =
    hashAt === -1
      ? null
      : new URLSearchParams(route.slice(hashAt + 1)).get(name)
  const value =
    fromFragment ??
    (queryAt === -1
      ? null
      : new URLSearchParams(beforeHash.slice(queryAt + 1)).get(name))
  return value === "" ? null : value
}

/** The href a route has in this build, base included. */
export function routeHref(route: string): string {
  return `${BASE_PATH}${route}`
}

/**
 * Move this tab to another route without a page load.
 *
 * The router's history listens to `history.pushState` and `replaceState`, so a
 * route it serves picks the change up as its own navigation, and `App` re-reads
 * the route for the pages in front of it.
 */
export function navigateTo(route: string, { replace = false } = {}): void {
  const href = routeHref(route)
  if (replace) {
    window.history.replaceState(null, "", href)
  } else {
    window.history.pushState(null, "", href)
  }
}

/**
 * Follow an in-app `<a href>` without a page load, for the plain anchors on
 * the pages in front of a session, where there is no router `Link` to use.
 *
 * A click the browser should keep (another tab, a download, a modified click)
 * is left alone, as is an href outside this build's base.
 */
export function followInAppLink(
  event: Pick<
    MouseEvent,
    | "button"
    | "metaKey"
    | "ctrlKey"
    | "shiftKey"
    | "altKey"
    | "defaultPrevented"
    | "preventDefault"
  > & { currentTarget: { getAttribute(name: string): string | null } },
): void {
  if (
    event.defaultPrevented ||
    event.button !== 0 ||
    event.metaKey ||
    event.ctrlKey ||
    event.shiftKey ||
    event.altKey
  ) {
    return
  }
  const target = event.currentTarget.getAttribute("target")
  if (target && target !== "_self") return
  const route = inAppRoute(event.currentTarget.getAttribute("href") ?? "")
  if (route === null) return
  event.preventDefault()
  navigateTo(route)
}

/** The route an href names in this build, or null when it leaves it. */
export function inAppRoute(href: string): string | null {
  if (!href.startsWith("/") || href.startsWith("//")) return null
  if (BASE_PATH && href !== BASE_PATH && !href.startsWith(`${BASE_PATH}/`)) {
    return null
  }
  return href.slice(BASE_PATH.length) || "/"
}

/** A route's query string, parsed: what follows its `?` and precedes its `#`. */
export function routeSearchParams(route: string): URLSearchParams {
  const beforeHash = route.split("#", 1)[0] ?? ""
  const queryAt = beforeHash.indexOf("?")
  return new URLSearchParams(
    queryAt === -1 ? "" : beforeHash.slice(queryAt + 1),
  )
}

/**
 * A route without its fragment, for keeping somewhere other than the address
 * bar: a link's secret rides in the fragment, so this is the part that holds
 * none.
 */
export function routeWithoutFragment(route: string): string {
  return route.split("#", 1)[0] || "/"
}
