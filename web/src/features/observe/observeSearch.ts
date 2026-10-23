import type { DashboardSearch } from "@/shared/helpers/search"
import {
  ACTIVITY_DEFAULT_KEY,
  ACTIVITY_PRESETS,
  type ChartBucket,
  findPreset,
  isoAgo,
  type RangePreset,
  SESSION_PRESETS,
  USAGE_DEFAULT_KEY,
  USAGE_PRESETS,
  YEAR_SPAN_S,
} from "@/shared/helpers/timeRange"

export type ObserveView = "usage" | "requests" | "sessions"

export const OBSERVE_PATHS = {
  usage: "/",
  requests: "/requests",
  sessions: "/sessions",
} as const satisfies Record<ObserveView, string>

/**
 * The query keys every view reads under the same name, and so the only ones a
 * tab switch carries. Pagination, the open session and a view's own controls
 * (the chart metric, a breakdown) describe one view and stay behind.
 */
export const SHARED_KEYS = [
  "scope",
  "range",
  "start_date",
  "end_date",
  "model",
  "user_id",
  "api_key_id",
  "status",
] as const

interface ViewRanges {
  presets: RangePreset<ChartBucket>[]
  defaultKey: string
}

// Sessions opens on the same default window as the request log.
const RANGES: Record<ObserveView, ViewRanges> = {
  usage: { presets: USAGE_PRESETS, defaultKey: USAGE_DEFAULT_KEY },
  requests: { presets: ACTIVITY_PRESETS, defaultKey: ACTIVITY_DEFAULT_KEY },
  sessions: { presets: SESSION_PRESETS, defaultKey: ACTIVITY_DEFAULT_KEY },
}

export function viewForPath(pathname: string): ObserveView {
  if (pathname.startsWith(OBSERVE_PATHS.requests)) return "requests"
  if (pathname.startsWith(OBSERVE_PATHS.sessions)) return "sessions"
  return "usage"
}

const firstValue = (value: string | string[] | undefined): string =>
  (Array.isArray(value) ? value[0] : value) ?? ""

/**
 * The query string a tab opens on: the shared filters as they stand, with the
 * window restated in terms the destination understands.
 *
 * The window is read the way `windowFor` reads it on the source view, so the
 * tab opens on what the source was showing: explicit bounds travel as they are,
 * and an absent, unrecognized or bound-less `custom` range is the source's
 * default (Usage 30 days, the other two 24 hours), pinned rather than carried
 * as absent. A preset the destination offers travels by name. One it lacks
 * becomes the destination's preset of the same span if there is one, else an
 * explicit start: 90 days on Usage opens Requests on those 90 days. The
 * unbounded "All" is a year, as `windowFor` frames it, so on Usage it is 12mo.
 */
export function carrySearch(
  from: ObserveView,
  to: ObserveView,
  search: DashboardSearch,
  now: number = Date.now(),
): DashboardSearch {
  const next: DashboardSearch = {}
  for (const key of SHARED_KEYS) {
    if (search[key] !== undefined) next[key] = search[key]
  }
  if (next.start_date || next.end_date) return next
  delete next.range

  const source = RANGES[from]
  const target = RANGES[to]
  const preset =
    findPreset(source.presets, firstValue(search.range)) ??
    findPreset(source.presets, source.defaultKey)
  const sourceKey = preset?.key ?? source.defaultKey
  const seconds = preset?.seconds ?? YEAR_SPAN_S
  const match =
    findPreset(target.presets, sourceKey) ??
    target.presets.find((one) => one.seconds === seconds)
  if (match) {
    if (match.key !== target.defaultKey) next.range = match.key
    return next
  }
  next.start_date = isoAgo(seconds, now)
  return next
}

/**
 * The window a view is showing, from the URL: explicit bounds win, then the
 * view's own preset, then its default (an unrecognized or `custom` range with
 * no bounds reads as the default, as each page does). A preset with no start
 * ("All") becomes a year, because the summary reads an omitted start as 30 days.
 * An absent bound is "".
 */
export function windowFor(
  view: ObserveView,
  search: DashboardSearch,
  now: number = Date.now(),
): { start: string; end: string } {
  const start = firstValue(search.start_date)
  const end = firstValue(search.end_date)
  if (start || end) return { start, end }

  const { presets, defaultKey } = RANGES[view]
  const preset =
    findPreset(presets, firstValue(search.range)) ??
    findPreset(presets, defaultKey)
  return { start: isoAgo(preset?.seconds ?? YEAR_SPAN_S, now), end: "" }
}

// Sessions have an outcome rather than a request status: one with an
// unrecovered error, or one without.
const SESSION_STATUSES = ["error", "success"]

/**
 * Whether a view can honor a status filter. Requests read every status,
 * Sessions only error and success, and Usage none.
 */
export function statusApplies(view: ObserveView, status: string): boolean {
  if (view === "requests") return true
  return view === "sessions" && SESSION_STATUSES.includes(status)
}

/**
 * The query a retired route (`/usage`, `/activity`) forwards to its Overview
 * tab. A link from before the scope switch that names a user means that
 * user, which Mine would ignore for a member, so it opens on Everyone unless
 * it already says which.
 */
export function legacySearch(search: DashboardSearch): DashboardSearch {
  return firstValue(search.user_id) && !firstValue(search.scope)
    ? { ...search, scope: "everyone" }
    : search
}
