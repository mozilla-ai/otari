// Where an upstream keeps request data, as the API spells it: an ISO 3166-1
// alpha-2 country code, `EU` (which a workspace's list reads as every member
// state too) or `GLOBAL`. Null on the wire is "nobody declared it"; the
// dashboard holds that as "".

import type { HostedProvider, HostedProviderRegion } from "@/client"

export const GLOBAL = "GLOBAL"
export const EU = "EU"

/** Every ISO 3166-1 alpha-2 code: the countries a data location may name. */
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
  "TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA ZM ZW"
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

/** The options whose name or code contains `query`, ignoring case. */
export function matchDataLocations(query: string): DataLocationOption[] {
  const term = query.trim().toLowerCase()
  if (term === "") return [...DATA_LOCATION_OPTIONS]
  return DATA_LOCATION_OPTIONS.filter(
    (option) =>
      option.label.toLowerCase().includes(term) ||
      option.value.toLowerCase() === term,
  )
}

/** Where a region keeps data: its own declaration, else the provider's; "" when neither declared one. */
export function effectiveDataLocation(
  region: Pick<HostedProviderRegion, "data_location">,
  provider: Pick<HostedProvider, "data_location">,
): string {
  return region.data_location ?? provider.data_location ?? ""
}

/** A provider's regions in the order requests try them. */
export function orderedRegions(
  provider: Pick<HostedProvider, "regions">,
): HostedProviderRegion[] {
  return [...(provider.regions ?? [])].sort((a, b) => a.position - b.position)
}

/**
 * Every distinct place a provider keeps data, in region order: its regions'
 * effective locations, or its own when it serves from one connection. Nothing
 * for a provider nobody declared a location for.
 */
export function providerDataLocations(provider: HostedProvider): string[] {
  const regions = orderedRegions(provider)
  const locations =
    regions.length === 0
      ? [provider.data_location ?? ""]
      : regions.map((region) => effectiveDataLocation(region, provider))
  return [...new Set(locations.filter(Boolean))]
}

/** A region slug as the API accepts it: what pins it in `provider@slug:model`. */
export const REGION_SLUG_PATTERN = /^[a-z0-9][a-z0-9-]{0,31}$/

/** The slug a provider's current connection is offered as when it becomes a region. */
export function primarySlugFor(provider: HostedProvider): string {
  return provider.data_location?.toLowerCase() || "primary"
}

/** The name that connection is offered under. */
export function primaryLabelFor(provider: HostedProvider): string {
  return provider.data_location
    ? dataLocationLabel(provider.data_location)
    : "Primary"
}
