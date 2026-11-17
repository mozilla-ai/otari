// Addresses pasted as a list, separated by commas, semicolons or whitespace,
// deduplicated without regard to case.
export function parseAddresses(text: string): string[] {
  const seen = new Set<string>()
  return text.split(/[\s,;]+/).filter((address) => {
    const key = address.toLowerCase()
    if (address === "" || seen.has(key)) return false
    seen.add(key)
    return true
  })
}

// A space, an invisible character, or a character that only a display name, comment, quoted text, list or domain literal needs.
const NOT_IN_A_PLAIN_ADDRESS = /[\p{C}\p{Z}"(),:;<>[\\\]]/u

/**
 * Whether the server could accept this as an address: a plain address with one
 * `@`, and a dot inside the domain with no empty label. The same shape
 * `normalized_address` checks, so a form can name a bad address before the
 * server answers with a 422.
 */
export function looksLikeAddress(address: string): boolean {
  // `=?` starts an encoded word.
  if (NOT_IN_A_PLAIN_ADDRESS.test(address) || address.includes("=?")) {
    return false
  }
  const [local, domain, ...rest] = address.split("@")
  if (!local || domain === undefined || rest.length > 0) return false
  const labels = domain.split(".")
  return labels.length > 1 && labels.every((label) => label !== "")
}
