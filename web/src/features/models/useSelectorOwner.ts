import type { CatalogModelSummary } from "@/client"
import { isPrefixedSelector } from "@/features/models/modelKey"
import { findOfferingOwner } from "@/features/models/offerings"
import { useCatalogSearch } from "@/shared/api/models"
import { useDebounced } from "@/shared/hooks/useDebounced"

// Searching by a whole selector matches its siblings too (`openai:gpt-5`
// finds `openai:gpt-5-mini`), so the page has room for them.
const LOOKUP_LIMIT = 25

/**
 * The catalog model that offers `selector`, once the catalog has answered.
 *
 * The catalog search matches selectors, so one bounded read names the model
 * behind a stored target. Undefined for text that is not a selector, while the
 * read is in flight, and for a selector no model in the caller's catalog offers.
 */
export function useSelectorOwner(
  selector: string,
  enabled = true,
): CatalogModelSummary | undefined {
  const term = useDebounced(selector.trim())
  const lookup = useCatalogSearch(
    term,
    LOOKUP_LIMIT,
    enabled && isPrefixedSelector(term),
  )
  return findOfferingOwner(lookup.data?.models ?? [], selector)
}
