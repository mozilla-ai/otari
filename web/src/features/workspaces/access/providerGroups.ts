/** What the list needs to know about one source to group, filter and count it. */
export interface AccessEntry {
  /** Unique across the list: a key's id, or a hosted provider's name. */
  id: string
  /** The provider it serves, which is what the list groups on. */
  provider: string
  /** What a search matches besides the provider. */
  name: string
  isCustomized: boolean
  isOff: boolean
}

export interface ProviderGroup<T extends AccessEntry> {
  provider: string
  entries: T[]
}

export type AccessFilter = "all" | "customized" | "off"

export const ACCESS_FILTERS: { value: AccessFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "customized", label: "Customized" },
  { value: "off", label: "Off" },
]

/** Past this many providers the list offers a search and a filter. */
export const SEARCH_THRESHOLD = 8

/**
 * One group per provider, in name order, each keeping its entries in the order
 * given. The caller passes keys before hosted providers, so a provider's own
 * keys read first inside its group.
 */
export function groupByProvider<T extends AccessEntry>(
  entries: readonly T[],
): ProviderGroup<T>[] {
  const groups = new Map<string, T[]>()
  for (const entry of entries) {
    const group = groups.get(entry.provider)
    if (group) group.push(entry)
    else groups.set(entry.provider, [entry])
  }
  return [...groups]
    .map(([provider, members]) => ({ provider, entries: members }))
    .sort((a, b) => a.provider.localeCompare(b.provider))
}

/**
 * The groups a search and a filter leave, each narrowed to its matching entries.
 *
 * A query naming the provider keeps the whole group; one naming a key keeps
 * that key. The filter then keeps only customized or switched-off entries, and
 * a group left with none is dropped.
 */
export function filterGroups<T extends AccessEntry>(
  groups: readonly ProviderGroup<T>[],
  query: string,
  filter: AccessFilter,
): ProviderGroup<T>[] {
  const needle = query.trim().toLowerCase()
  const keep = (entry: T) =>
    filter === "all" ||
    (filter === "customized" ? entry.isCustomized : entry.isOff)
  return groups.flatMap((group) => {
    const named = group.provider.toLowerCase().includes(needle)
    const entries = group.entries.filter(
      (entry) =>
        (named || entry.name.toLowerCase().includes(needle)) && keep(entry),
    )
    return entries.length > 0 ? [{ ...group, entries }] : []
  })
}

/** "12 providers · 2 customized · 1 off", leaving out what is zero. */
export function summarizeGroups(
  groups: readonly ProviderGroup<AccessEntry>[],
): string {
  const entries = groups.flatMap((group) => group.entries)
  const customized = entries.filter((entry) => entry.isCustomized).length
  const off = entries.filter((entry) => entry.isOff).length
  return [
    `${groups.length} ${groups.length === 1 ? "provider" : "providers"}`,
    customized > 0 ? `${customized} customized` : "",
    off > 0 ? `${off} off` : "",
  ]
    .filter(Boolean)
    .join(" · ")
}
