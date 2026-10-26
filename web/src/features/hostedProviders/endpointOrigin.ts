// Whether an edit repoints a hosted provider at another origin, which the API
// refuses unless the key is entered again: the stored key would otherwise go to
// an endpoint chosen without knowing it. Mirrors the server's `origin_of`
// (scheme, host, effective port) so the form can ask for the key before the
// save rather than after; the server still decides.

const DEFAULT_PORTS: Record<string, string> = { "http:": "80", "https:": "443" }

/** `scheme://host:port` with the port made explicit, or `""` for the default. */
export function originOf(apiBase: string): string {
  const trimmed = apiBase.trim()
  if (trimmed === "") return ""
  try {
    const url = new URL(trimmed)
    if (url.hostname === "") return ""
    const port = url.port || DEFAULT_PORTS[url.protocol] || ""
    return `${url.protocol}//${url.hostname.toLowerCase()}:${port}`
  } catch {
    // Not a URL the server reads a host from either.
    return ""
  }
}

/**
 * True when saving `next` over `stored` needs the key retyped. Going back to the
 * provider's default endpoint does not: no caller-chosen host receives the key.
 */
export function movesToAnotherOrigin(stored: string, next: string): boolean {
  const target = originOf(next)
  return target !== "" && target !== originOf(stored)
}
