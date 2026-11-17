/**
 * Which providers page a hash names, for the render ahead of a session.
 *
 * `#/providers` is the list, `#/providers/<name>` one provider; anything else is
 * not the public providers pages' to answer. Matched on the path alone, as
 * `publicCatalogPath` is, so a query string cannot make a known path
 * unrecognizable.
 */
export function publicProvidersPath(
  hash: string,
): { name: string | undefined } | null {
  const path = hash.replace(/^#/, "").split("?")[0] ?? ""
  if (path === "/providers" || path === "/providers/") {
    return { name: undefined }
  }
  // An instance name cannot contain a slash, so one segment is the whole name.
  const match = /^\/providers\/([^/]+)\/?$/.exec(path)
  if (!match) return null
  try {
    return { name: decodeURIComponent(match[1] ?? "") }
  } catch {
    return null
  }
}

/** The hash a public providers link points at. */
export function publicProvidersHref(name?: string): string {
  return name ? `#/providers/${encodeURIComponent(name)}` : "#/providers"
}
