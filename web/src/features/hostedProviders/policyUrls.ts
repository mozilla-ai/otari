/**
 * Why a provider's Terms of service or Privacy policy URL would be refused, or
 * `undefined` when it would not. Blank is allowed (no link); anything else has
 * to be an absolute http(s) URL, the rule the server applies with a 422.
 */
export function policyUrlError(value: string): string | undefined {
  const trimmed = value.trim()
  if (trimmed === "") return undefined
  try {
    const url = new URL(trimmed)
    if (url.protocol === "http:" || url.protocol === "https:") return undefined
  } catch {
    // Not a URL at all; the message below covers it.
  }
  return "Enter a full http:// or https:// address."
}

/** The value a URL field sends: null for blank, so the link is cleared. */
export function policyUrlValue(value: string): string | null {
  return value.trim() || null
}
