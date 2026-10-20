import type { DeploymentBootstrap } from "@/client"

/** Where the dashboard's Documentation links go when `docs_url` is unset. */
export const OTARI_DOCS_URL = "https://docs.otari.ai/"

export function documentationHref({
  docs_url,
}: Pick<DeploymentBootstrap, "docs_url">): string {
  return docs_url ?? OTARI_DOCS_URL
}

/**
 * Where a repository document is rendered, for a link to one section of it.
 * GitHub serves every document at `main`, under the same filename.
 */
export const DOCS_SOURCE_BASE =
  "https://github.com/mozilla-ai/otari/blob/main/docs/"

export function docsSourceHref(doc: string, anchor?: string): string {
  return `${DOCS_SOURCE_BASE}${doc}${anchor ? `#${anchor}` : ""}`
}
