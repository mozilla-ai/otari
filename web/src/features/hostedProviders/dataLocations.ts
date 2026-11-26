// Where an upstream keeps request data, as the API spells it: an ISO 3166-1
// alpha-2 country code, `EU` (which a workspace's list reads as every member
// state too) or `GLOBAL`. Null on the wire is "nobody declared it"; the
// dashboard holds that as "".

import type { HostedProvider, HostedProviderRegion } from "@/client"

export const GLOBAL = "GLOBAL"
export const EU = "EU"

/**
 * Every ISO 3166-1 alpha-2 code, and Kosovo's user-assigned `XK`, which the
 * gateway accepts too: the countries a data location may name.
 */
const COUNTRY_CODES = (
  "AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ " +
  "BL BM BN BO BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CR " +
  "CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR " +
  "GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU " +
  "ID IE IL IM IN IO IQ IR IS IT JE JM JO JP KE KG KH KI KM KN KP KR KW KY KZ " +
  "LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ " +
  "MR MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF " +
  "PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD SE SG SH SI " +
  "SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM TN TO TR " +
  "TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS XK YE YT ZA ZM ZW"
).split(" ")

const regionNames = new Intl.DisplayNames("en-US", {
  type: "region",
  fallback: "code",
})

/** A location's name: "Global", "European Union", or the country's name. */
export function dataLocationLabel(code: string): string {
  if (code === GLOBAL) return "Global"
  if (code === EU) return "European Union"
  try {
    return regionNames.of(code) ?? code
  } catch {
    // A code Intl refuses to read is still the operator's to see.
    return code
  }
}

export interface DataLocationOption {
  value: string
  label: string
  /** The code, shown under a country's name so either can be searched. */
  hint?: string
}

/** Global and the European Union first, then every country by name. */
export const DATA_LOCATION_OPTIONS: readonly DataLocationOption[] = [
  { value: GLOBAL, label: dataLocationLabel(GLOBAL) },
  { value: EU, label: dataLocationLabel(EU), hint: EU },
  ...COUNTRY_CODES.map((code) => ({
    value: code,
    label: dataLocationLabel(code),
    hint: code,
  })).sort((a, b) => a.label.localeCompare(b.label)),
]

/**
 * The options whose name or code contains `query`, ignoring case, capped at
 * `limit`. With no query, `selected` is kept in the list even past the cap, so
 * a picked country far down the alphabet still shows by name.
 */
export function matchDataLocations(
  query: string,
  limit: number = Number.POSITIVE_INFINITY,
  selected = "",
): DataLocationOption[] {
  const term = query.trim().toLowerCase()
  if (term === "") {
    const shown = DATA_LOCATION_OPTIONS.slice(0, limit)
    const picked = DATA_LOCATION_OPTIONS.find(
      (option) => option.value === selected,
    )
    return picked && !shown.includes(picked) ? [picked, ...shown] : shown
  }
  return DATA_LOCATION_OPTIONS.filter(
    (option) =>
      option.label.toLowerCase().includes(term) ||
      option.value.toLowerCase() === term,
  ).slice(0, limit)
}

/** A provider's regions in the order requests try them. */
export function orderedRegions(
  provider: Pick<HostedProvider, "regions">,
): HostedProviderRegion[] {
  return [...(provider.regions ?? [])].sort((a, b) => a.position - b.position)
}

/** The regions requests can reach: the enabled ones, in order. */
export function servingRegions(
  provider: Pick<HostedProvider, "regions">,
): HostedProviderRegion[] {
  return orderedRegions(provider).filter((region) => region.enabled)
}

/**
 * Every distinct place a provider serves from, in region order: each serving
 * region's own location, or the provider's when it has no regions. Null
 * stands for a connection that declares none, which no allowed list admits.
 */
export function providerDataLocations(
  provider: HostedProvider,
): (string | null)[] {
  const locations =
    (provider.regions ?? []).length === 0
      ? [provider.data_location ?? null]
      : servingRegions(provider).map((region) => region.data_location ?? null)
  return [...new Set(locations)]
}

/** EU member states, which an allowed list naming `EU` admits as well. */
export const EU_MEMBER_STATES = new Set(
  (
    "AT BE BG CY CZ DE DK EE ES FI FR GR HR HU IE IT LT LU LV MT NL PL PT RO " +
    "SE SI SK"
  ).split(" "),
)

/**
 * Whether a workspace's allowed list admits a location, as the gateway reads
 * it: no list admits everything, `EU` also admits each member state, `GLOBAL`
 * admits only `GLOBAL`, and an undeclared location matches nothing.
 */
export function admitsLocation(
  allowed: readonly string[] | null,
  location: string | null,
): boolean {
  if (allowed === null) return true
  if (location === null) return false
  return (
    allowed.includes(location) ||
    (allowed.includes(EU) && EU_MEMBER_STATES.has(location))
  )
}

/**
 * Whether a list refuses a provider: none of the places it serves from is on
 * it. A provider listing no location at all declares none.
 */
export function refusesProvider(
  allowed: readonly string[] | null,
  locations: readonly (string | null)[],
): boolean {
  const declared = locations.length === 0 ? [null] : locations
  return !declared.some((location) => admitsLocation(allowed, location))
}

/** Locations as a short list of codes, an undeclared one as "not declared". */
export function formatLocationCodes(
  locations: readonly (string | null)[],
): string {
  const declared = locations.filter((location) => location !== null)
  return declared.length === 0 ? "not declared" : declared.join(", ")
}

/** A region slug as the API accepts it: what pins it in `provider@slug:model`. */
export const REGION_SLUG_PATTERN = /^[a-z0-9][a-z0-9-]{0,31}$/

/**
 * The slug a provider's current connection takes when it becomes a region,
 * as the server defaults it: its data location, else `primary`.
 */
export function primarySlugFor(provider: HostedProvider): string {
  const slug = provider.data_location?.toLowerCase() ?? ""
  return REGION_SLUG_PATTERN.test(slug) ? slug : "primary"
}

/**
 * A region id derived from its name: lower case, a hyphen for every run of
 * anything else, at most 32 characters. Empty when nothing usable is left.
 */
export function slugFromLabel(label: string): string {
  return label
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+/, "")
    .slice(0, 32)
    .replace(/-+$/, "")
}

/** The name that connection is offered under. */
export function primaryLabelFor(provider: HostedProvider): string {
  return provider.data_location
    ? dataLocationLabel(provider.data_location)
    : "Primary"
}
