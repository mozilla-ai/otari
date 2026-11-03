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

const NOT_A_LIST =
  'The file is not a JSON list of country codes, such as ["US", "DE"].'

/**
 * The codes in an uploaded file: a JSON list, or one under
 * `supported_countries` as the API and config.yml spell it. The codes are
 * checked once they are in the field, as typed ones are.
 */
export function countriesFromJson(
  text: string,
): { ok: true; value: string[] } | { ok: false; error: string } {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return { ok: false, error: NOT_A_LIST }
  }
  const list =
    parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>).supported_countries
      : parsed
  if (
    !Array.isArray(list) ||
    !list.every((item): item is string => typeof item === "string")
  ) {
    return { ok: false, error: NOT_A_LIST }
  }
  return { ok: true, value: list }
}
