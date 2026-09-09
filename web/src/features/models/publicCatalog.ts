/**
 * Which catalog page a hash names, for the render ahead of a session.
 *
 * `#/models` is the list, `#/models/<id>` a selected model; anything else is
 * not the public catalog's to answer. Matched on the path alone, so a query
 * string (`?provider=…`) cannot make a known path unrecognizable, the way
 * `publicAuthPath` reads its own table.
 */
export function publicCatalogPath(
  hash: string,
): { modelId: string | undefined } | null {
  const path = hash.replace(/^#/, "").split("?")[0] ?? ""
  if (path === "/models" || path === "/models/") {
    return { modelId: undefined }
  }
  const match = /^\/models\/([^/]+)$/.exec(path)
  if (!match) return null
  return { modelId: decodeURIComponent(match[1] ?? "") }
}

/** The hash a public catalog link points at. */
export function publicCatalogHref(modelId?: string): string {
  return modelId ? `#/models/${encodeURIComponent(modelId)}` : "#/models"
}
