/**
 * Why a provider's Terms of service or Privacy policy URL would be refused, or
 * `undefined` when it would not. Blank is allowed (no link); anything else has
 * to be an absolute http(s) URL with no username or password, the rule the
 * server applies with a 422.
 */
export function policyUrlError(value: string): string | undefined {
  const trimmed = value.trim()
  if (trimmed === "") return undefined
  // \`new URL\` repairs \`https:example.com\`, which the server, parsing the
  // string as sent, refuses for having no host.
  if (!/^https?:\/\//i.test(trimmed)) {
    return "Enter a full http:// or https:// address."
  }
  let url: URL
  try {
    url = new URL(trimmed)
  } catch {
    return "Enter a full http:// or https:// address."
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return "Enter a full http:// or https:// address."
  }
  // The link is shown to visitors, so a credential in it would be published.
  if (url.username || url.password) return "Leave out a username or password."
  return undefined
}

/** The value a URL field sends: null for blank, so the link is cleared. */
export function policyUrlValue(value: string): string | null {
  return value.trim() || null
}
