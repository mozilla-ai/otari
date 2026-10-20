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

/**
 * Whether the server could accept this as an address: one `@`, and a dot inside
 * the domain with no empty label. The same shape `normalized_address` checks,
 * so a form can name a bad address before the server answers with a 422.
 */
export function looksLikeAddress(address: string): boolean {
  const [local, domain, ...rest] = address.split("@")
  if (!local || domain === undefined || rest.length > 0) return false
  const labels = domain.split(".")
  return labels.length > 1 && labels.every((label) => label !== "")
}
