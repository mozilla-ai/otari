import type { DeploymentBootstrap } from "@/client"

/** Where the dashboard's Documentation links go when `docs_url` is unset. */
export const OTARI_DOCS_URL = "https://docs.otari.ai/"

export function documentationHref({
  docs_url,
}: Pick<DeploymentBootstrap, "docs_url">): string {
  return docs_url ?? OTARI_DOCS_URL
}

/**
 * The link a page offers to the getting-started guide.
 *
 * `/welcome` is the gateway's own self-hosted quickstart. otari.ai configures
 * `docs_url` and does not serve `/welcome` at its origin, so where `docs_url` is
 * set the guide is that documentation: an operator who points `docs_url` at
 * their own site gets it instead of `/welcome`. It lives on another origin, so
 * it opens in a new tab. Unset, the same-origin `/welcome` stays as it was.
 */
export function welcomeGuideLink({
  docs_url,
}: Pick<DeploymentBootstrap, "docs_url">): {
  href: string
  label: string
  target?: "_blank"
  rel?: string
} {
  return docs_url
    ? {
        href: docs_url,
        label: "Read the documentation",
        target: "_blank",
        rel: "noopener noreferrer",
      }
    : { href: "/welcome", label: "Open the welcome guide" }
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
