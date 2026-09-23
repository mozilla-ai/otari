/**
 * A link the server built into this dashboard, made absolute so it can be
 * pasted into a chat or an email.
 *
 * The server can only make it absolute when it knows its public address; without
 * one it answers `/#/...`. That is resolved against the page the operator is on
 * rather than against the origin, because a hosted bundle is served under a path
 * prefix, and every dashboard route lives in the hash.
 */
export function absoluteDashboardLink(
  link: string,
  location: Pick<Location, "origin" | "pathname"> = window.location,
): string {
  if (/^https?:\/\//i.test(link)) return link
  const hash = link.indexOf("#")
  if (hash === -1) return new URL(link, location.origin).href
  return `${location.origin}${location.pathname}${link.slice(hash)}`
}
