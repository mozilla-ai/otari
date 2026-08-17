import { useNavigate, useSearch } from "@tanstack/react-router"
import { useCallback } from "react"

import type { DashboardSearch } from "@/shared/helpers/search"

// Keep table filter/pagination state in the URL query string, so a filtered view
// is shareable and survives the back button. Values equal to their default are
// removed to keep the URL clean, and every update is a single `navigate` call:
// the router's functional updater is based on the current location, so several
// separate calls in one tick would clobber each other rather than compose.
// `patch` therefore takes all the keys to change at once.

// `strict: false` because these hooks are shared by every page rather than bound
// to one route; the shape is the root route's, which every route inherits.
function useSearchRecord(): DashboardSearch {
  return useSearch({ strict: false }) as DashboardSearch
}

// The first value of a key, whether it was written once or repeated. `undefined`
// means the key is absent, which is what lets a default apply; a key present but
// blank ("?source=") is a cleared filter and reads as "".
function first(raw: string | string[] | undefined): string | undefined {
  return Array.isArray(raw) ? raw[0] : raw
}

/** Read one search param, for state that is only seeded from the URL. */
export function useUrlValue(key: string, defaultValue = ""): string {
  return first(useSearchRecord()[key]) ?? defaultValue
}

export interface UrlState<K extends string> {
  get: (key: K) => string
  /** Every value of a repeatable key (`?model=a&model=b`), empty when it is absent. */
  getAll: (key: K) => string[]
  getNumber: (key: K) => number
  /**
   * Apply several key changes in one history entry; "" or the default drops a key.
   * An array writes the key once per value, and an empty array drops it.
   */
  patch: (updates: Partial<Record<K, string | number | string[]>>) => void
}

export function useUrlState<K extends string>(
  defaults: Record<K, string>,
): UrlState<K> {
  const search = useSearchRecord()
  const navigate = useNavigate()

  const get = useCallback(
    (key: K) => first(search[key]) ?? defaults[key],
    [search, defaults],
  )

  // Values are trimmed and blanks dropped, so `?model=` or `?model=%20` reads as no
  // filter rather than a filter on whitespace (which would match nothing and look
  // like an empty result set). The default applies only when the key is absent
  // entirely: present-but-blank is a cleared filter, the same reading `get` gives it.
  const getAll = useCallback(
    (key: K) => {
      const raw = search[key]
      if (raw === undefined) {
        return defaults[key] ? [defaults[key]] : []
      }
      return (Array.isArray(raw) ? raw : [raw])
        .map((value) => value.trim())
        .filter((value) => value !== "")
    },
    [search, defaults],
  )

  const getNumber = useCallback(
    (key: K) => {
      // A present but non-numeric param (e.g. a hand-edited `?size=abc`) must fall
      // back to the key's default, not 0: a 0 page size would send `limit=0` and 422.
      const parsed = Number.parseInt(first(search[key]) ?? "", 10)
      if (!Number.isNaN(parsed)) {
        return parsed
      }
      const fallback = Number.parseInt(defaults[key], 10)
      return Number.isNaN(fallback) ? 0 : fallback
    },
    [search, defaults],
  )

  const patch = useCallback(
    (updates: Partial<Record<K, string | number | string[]>>) => {
      navigate({
        to: ".",
        search: (prev) => {
          const next: DashboardSearch = { ...(prev as DashboardSearch) }
          for (const [key, raw] of Object.entries(updates)) {
            if (Array.isArray(raw)) {
              // Rewritten wholesale rather than appended to: the caller passes the
              // filter's complete value set, so a removed value has to disappear.
              const values = raw.filter((value) => value !== "")
              if (values.length === 0) {
                delete next[key]
              } else {
                next[key] = values
              }
              continue
            }
            const value = String(raw)
            if (value === "" || value === defaults[key as K]) {
              delete next[key]
            } else {
              next[key] = value
            }
          }
          return next
        },
        replace: true,
      })
    },
    [navigate, defaults],
  )

  return { get, getAll, getNumber, patch }
}
