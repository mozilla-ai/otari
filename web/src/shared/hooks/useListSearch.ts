import { useUrlState } from "@/shared/helpers/urlState"
import { useDebounced } from "@/shared/hooks/useDebounced"

/** The longest term the gateway's `search` params accept (`max_length=200`). */
export const SEARCH_MAX_LENGTH = 200

/**
 * A list's search term, kept in the URL as `?q=` so a narrowed list survives a
 * reload, the back button and a shared link.
 *
 * `value` and `onChange` belong to the `SearchField`; a change sends the list
 * back to its first page. `term` is what to search with: trimmed, capped, and
 * settled for a pause in typing, so it can go into a query key without a
 * request per keystroke, or into `matchesSearch` for rows already in memory.
 *
 * A list that shares its page with another names its own `param`, and the
 * `pageParam` its pager keeps, so the two searches and pagers stay apart.
 */
export function useListSearch({
  param = "q",
  pageParam = "page",
}: {
  param?: string
  pageParam?: string
} = {}) {
  const url = useUrlState({ [param]: "", [pageParam]: "0" })
  const value = url.get(param)
  const term = useDebounced(value.trim().slice(0, SEARCH_MAX_LENGTH))
  return {
    value,
    onChange: (next: string) => url.patch({ [param]: next, [pageParam]: 0 }),
    term,
  }
}

/**
 * Whether any of `fields` contains `term`, case-insensitively. An empty term
 * matches everything.
 *
 * The in-memory half of a list search, for a list the page already holds in
 * full. It matches the way the gateway's `search` params do (a lowercased
 * substring, nothing treated as a wildcard), so a term finds the same rows
 * whichever side searches. A caller passes an absent field as `""`.
 */
export function matchesSearch(term: string, ...fields: string[]): boolean {
  if (term === "") return true
  const needle = term.toLowerCase()
  return fields.some((field) => field.toLowerCase().includes(needle))
}
