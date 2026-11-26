/**
 * The parts of `window.location` a route sets, for a test that stands a plain
 * object in for the location: spreading the real one copies its values at that
 * moment, so a stub has to carry the route's own.
 */
export function routeLocation(
  route: string,
): Pick<Location, "pathname" | "search" | "hash"> {
  const url = new URL(route, "http://localhost")
  return { pathname: url.pathname, search: url.search, hash: url.hash }
}
