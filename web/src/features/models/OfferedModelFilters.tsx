import { SearchField } from "@/design-system/forms/SearchField"
import { Segmented } from "@/design-system/navigation/Segmented"
import { useDebounced } from "@/shared/hooks/useDebounced"

// The routes' `max_length` on `search`: a longer term from a hand-edited URL
// would be refused with a 422 rather than searched.
const MAX_SEARCH_LENGTH = 200

/** Which models a provider's list shows by whether each is served; empty for both. */
export type ServingFilter = "" | "serving" | "withheld"

const SERVING_OPTIONS: { value: ServingFilter; label: string }[] = [
  { value: "", label: "All" },
  { value: "serving", label: "Serving" },
  { value: "withheld", label: "Not serving" },
]

/** A URL value read as a serving filter, and anything else as both. */
export function servingFilterOf(value: string): ServingFilter {
  return value === "serving" || value === "withheld" ? value : ""
}

/** The list endpoint's `enabled` for a serving filter; undefined for both. */
export function servedParam(filter: ServingFilter): boolean | undefined {
  return filter === "" ? undefined : filter === "serving"
}

/** What an empty table says under a search or a serving filter; undefined for the default. */
export function filteredEmptyContent(
  term: string,
  serving: ServingFilter,
): string | undefined {
  if (term) return "No models match this search."
  if (serving === "withheld") return "Every model here is served."
  if (serving === "serving") return "No model here is served."
  return undefined
}

/** What was typed into a provider's model search, as the list endpoint is asked for it. */
export function useModelSearchTerm(typed: string): string {
  return useDebounced(typed.trim().slice(0, MAX_SEARCH_LENGTH))
}

/**
 * A provider's model list filters that sit beside its actions: whether a model
 * is served, and a search by its name. The kind tabs have the row above the
 * table to themselves, since a provider can offer a dozen kinds.
 */
export function OfferedModelFilters({
  providerName,
  serving,
  onServingChange,
  search,
  onSearchChange,
}: {
  providerName: string
  serving: ServingFilter
  onServingChange: (serving: ServingFilter) => void
  /** What is typed, as it is typed; the caller debounces it on the way to a query. */
  search: string
  onSearchChange: (search: string) => void
}) {
  return (
    // Pushed to the end of the actions' row, and onto a row of its own end
    // when the two do not fit side by side.
    <div className="flex w-full flex-col gap-2 md:ml-auto md:w-auto md:flex-row md:items-center">
      <Segmented
        label={`Serving models on ${providerName}`}
        size="md"
        options={SERVING_OPTIONS}
        value={serving}
        onChange={(next) => onServingChange(servingFilterOf(next))}
      />
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
