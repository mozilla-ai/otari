import type { DeploymentBootstrap } from "@/client"

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
  // A model id carries its vendor, `z-ai/glm-5.3`, so the rest of the path is
  // the id whole.
  const match = /^\/models\/(.+)$/.exec(path)
  if (!match) return null
  try {
    return { modelId: decodeURIComponent(match[1] ?? "") }
  } catch {
    // A malformed escape (`#/models/%`) is not a model id; it is not the
    // public catalog's hash at all.
    return null
  }
}

/** The hash a public catalog link points at. */
export function publicCatalogHref(modelId?: string): string {
  return modelId ? `#/models/${modelId}` : "#/models"
}

/**
 * Where the logo on a page reached without an account leads: the deployment's
 * public website, else its public catalog. `""` when it publishes neither, so
 * the only page left would be the one the visitor is on and the logo stays
 * unlinked.
 */
export function siteHomeHref({
  site_url,
  public_catalog,
}: Pick<DeploymentBootstrap, "site_url" | "public_catalog">): string {
  if (site_url) return site_url
  return public_catalog ? publicCatalogHref() : ""
}

// A visitor who asks to use a model is sent to create an account first, and
// that trip runs through an emailed verification link, so the model has to
// survive a new tab: localStorage, not sessionStorage. A day covers a verify
// link opened later; past that the choice is stale.
const REMEMBERED_MODEL_KEY = "otari.catalog.remembered-model"
const REMEMBERED_MODEL_TTL_MS = 24 * 60 * 60 * 1000

/** Remembers the model a visitor chose, for {@link takeRememberedModel} after sign-in. */
export function rememberModel(modelId: string, now = Date.now()): void {
  try {
    window.localStorage.setItem(
      REMEMBERED_MODEL_KEY,
      JSON.stringify({ modelId, at: now }),
    )
  } catch {
    // Storage off: the visitor lands on the dashboard's home instead.
  }
}

/** The remembered model, if one is fresh, forgotten as it is read. */
export function takeRememberedModel(now = Date.now()): string | null {
  try {
    const raw = window.localStorage.getItem(REMEMBERED_MODEL_KEY)
    window.localStorage.removeItem(REMEMBERED_MODEL_KEY)
    if (!raw) return null
    const { modelId, at } = JSON.parse(raw) as {
      modelId?: unknown
      at?: unknown
    }
    if (typeof modelId !== "string" || !modelId || typeof at !== "number") {
      return null
    }
    return now - at <= REMEMBERED_MODEL_TTL_MS ? modelId : null
  } catch {
    return null
  }
}
