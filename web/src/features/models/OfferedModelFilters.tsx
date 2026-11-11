import { SearchField } from "@/design-system/forms/SearchField"
import { useDebounced } from "@/shared/hooks/useDebounced"

import { OfferedKindTabs } from "./OfferedKindTabs"
import type { KindFilter } from "./offeredKinds"

// The routes' `max_length` on `search`: a longer term from a hand-edited URL
// would be refused with a 422 rather than searched.
const MAX_SEARCH_LENGTH = 200

/** What was typed into a provider's model search, as the list endpoint is asked for it. */
export function useModelSearchTerm(typed: string): string {
  return useDebounced(typed.trim().slice(0, MAX_SEARCH_LENGTH))
}

/**
 * A provider's model list filters: the kind tabs, where there is a list to
 * divide, and a search by model name beside them.
 */
export function OfferedModelFilters({
  providerName,
  counts,
  showTabs,
  kind,
  onKindChange,
  search,
  onSearchChange,
}: {
  providerName: string
  counts: Readonly<Record<string, number>> | undefined
  showTabs: boolean
  kind: KindFilter
  onKindChange: (kind: KindFilter) => void
  /** What is typed, as it is typed; the caller debounces it on the way to a query. */
  search: string
  onSearchChange: (search: string) => void
}) {
  return (
    <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
      <div className="min-w-0">
        {showTabs ? (
          <OfferedKindTabs counts={counts} shown={kind} onShow={onKindChange} />
        ) : null}
      </div>
      <SearchField
        label={`Search models on ${providerName}`}
        placeholder="Search models"
        value={search}
        onChange={onSearchChange}
        className="w-full md:w-64 md:shrink-0"
      />
    </div>
  )
}
