/**
 * Where a repository document is actually rendered.
 *
 * Only `docs/dashboard.md` is bundled into the dashboard, so its own relative
 * links cannot resolve inside the SPA and neither can a link to a sibling
 * document. GitHub is where both go. Pointing a "Docs" link at `/#/docs`
 * instead would land on a guide that has no section by that name.
 */
export const DOCS_SOURCE_BASE =
  "https://github.com/mozilla-ai/otari/blob/main/docs/"

export function docsSourceHref(doc: string, anchor?: string): string {
  return `${DOCS_SOURCE_BASE}${doc}${anchor ? `#${anchor}` : ""}`
}
