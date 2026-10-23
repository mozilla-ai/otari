// The "Supported countries" field holds ISO 3166-1 alpha-2 codes as text
// ("US, DE"), and blank means every country, as null does on the wire. The
// server checks the codes too; this only says so before the save.

export type ParsedCountries =
  | { ok: true; value: string[] | null }
  | { ok: false; invalid: string[] }

export function parseCountries(text: string): ParsedCountries {
  const tokens = text.split(/[\s,]+/).filter(Boolean)
  if (tokens.length === 0) return { ok: true, value: null }
  const invalid = tokens.filter((token) => !/^[A-Za-z]{2}$/.test(token))
  if (invalid.length > 0) return { ok: false, invalid }
  return { ok: true, value: [...new Set(tokens.map((t) => t.toUpperCase()))] }
}

export function formatCountries(
  countries: string[] | null | undefined,
): string {
  return countries?.join(", ") ?? ""
}
