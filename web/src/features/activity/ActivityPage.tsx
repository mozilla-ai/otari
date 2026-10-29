import { Button } from "@heroui/react"
import { useNavigate } from "@tanstack/react-router"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import type { SummaryDimension, UsageEntry, UsageFilters } from "@/client"
import { RefreshButton } from "@/design-system/actions/RefreshButton"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import {
  PAGE_SIZE_OPTIONS,
  TablePagination,
} from "@/design-system/data/TablePagination"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { useIsMobile } from "@/design-system/hooks/useIsMobile"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import {
  type FilterChip,
  FilterChips,
} from "@/design-system/navigation/FilterChips"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import {
  type PricingUnit,
  pricingUnitForEndpoint,
} from "@/features/models/pricingUnit"
import {
  type ManualRates,
  SetPriceDialog,
} from "@/features/models/SetPriceDialog"
import { useObserveScope } from "@/features/observe/useObserveScope"
import { useMemberAttributionLabels } from "@/features/organization/attribution"
import { userDisplay } from "@/features/users/userDisplay"
import { useSetPricing } from "@/shared/api/pricing"
import {
  useInFlightRequests,
  useLiveUsageCount,
  useRequestGroups,
  useUsageCount,
  useUsageLogs,
  useUsageScope,
  useUsageSummary,
} from "@/shared/api/usage"
import { formatNumber } from "@/shared/helpers/format"
import type { DashboardSearch } from "@/shared/helpers/search"
import {
  ACTIVITY_DEFAULT_KEY,
  ACTIVITY_PRESETS,
  bucketForWindow,
  CUSTOM_KEY,
  findPreset,
  type RangePreset,
} from "@/shared/helpers/timeRange"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { ActivityTimeline } from "./ActivityTimeline"
import {
  describeSource,
  findGroupsMissingEarlierAttempts,
  formatLatencyCell,
  getActivityRowClassName,
  getActivityRowKey,
  indexGroupOutcomes,
  PRICED_OPTIONS,
  resolveExtentWindow,
  resolveWindow,
  TOOL_OPTIONS,
} from "./activityModel"
import { InFlightControl } from "./InFlightControl"
import { RequestPeek } from "./RequestPeek"
import { RoutingCell } from "./RoutingCell"
import {
  API_KEY_COLUMN,
  COST_COLUMN,
  INPUT_TOKENS_COLUMN,
  MODEL_COLUMN,
  OUTPUT_TOKENS_COLUMN,
  STATUS_COLUMN,
  TIME_COLUMN,
} from "./requestColumns"

// The only breakdown this page asks the summary for: whether the window contains
// gateway-run tool calls, which decides if the Tool filter is worth offering.
const TOOL_BREAKDOWN: SummaryDimension[] = ["tool"]

const DEFAULT_PAGE_SIZE = 50

// The source picker's option list: the sources with usage in the window.
const SOURCE_BREAKDOWN: SummaryDimension[] = ["source"]

// All filter + pagination state, with defaults, kept in the URL.
const URL_DEFAULTS = {
  range: ACTIVITY_DEFAULT_KEY,
  start_date: "",
  end_date: "",
  status: "",
  model: "",
  user_id: "",
  api_key_id: "",
  priced: "",
  source: "",
  source_label: "",
  endpoint: "",
  provider: "",
  tool: "",
  page: "0",
  size: String(DEFAULT_PAGE_SIZE),
  // The usage row open in the panel.
  request: "",
} as const

/** Inside the Overview the page waits for its scope, so it never fetches the wrong one. */
export function ActivityPage({
  embedded = false,
}: {
  embedded?: boolean
} = {}) {
  const observe = useObserveScope(embedded)
  return observe.isResolved ? (
    <ActivityPageContent embedded={embedded} observe={observe} />
  ) : (
    <PageLoading />
  )
}

function ActivityPageContent({
  embedded = false,
  observe,
}: {
  embedded?: boolean
  observe: ReturnType<typeof useObserveScope>
}) {
  // Who the people behind the owner ids are. The log already carries the alias
  // the gateway was told; the roster is who the person is, so it wins.
  const memberLabels = useMemberAttributionLabels()
  // Filter + pagination state lives in the URL, so a filtered view is shareable
  // and survives the back button. `patch` batches related changes into one entry.
  const url = useUrlState(URL_DEFAULTS)
  const range = url.get("range")
  const startParam = url.get("start_date")
  const endParam = url.get("end_date")
  const statusFilter = url.get("status")
  // The three entity filters hold sets: each is repeatable in the URL and on the
  // wire, so a comparison ("these two models") is one view rather than several, and
  // a drill-down from the analytics page can carry its whole selection across.
  const modelFilters = url.getAll("model")
  // Inside the Overview the scope decides this; elsewhere it is the URL's own.
  const { userIds: userFilters, scope: observeScope } = observe
  const apiKeyFilters = url.getAll("api_key_id")
  const pricedFilter = url.get("priced")
  // Provenance: gateway traffic vs an imported agent source. Set by its own select,
  // or by a drill-down (the pricing alarm links here scoped to gateway traffic).
  const sourceFilter = url.get("source")
  // The Usage breakdowns (session / endpoint / provider) drill in the same way:
  // no select of their own, carried as a chip so the scoping is visible and one
  // click removes it.
  const sessionFilter = url.get("source_label")
  const endpointFilter = url.get("endpoint")
  const providerFilter = url.get("provider")
  const toolFilter = url.get("tool")
  const page = Math.max(0, url.getNumber("page"))
  // Snap URL-supplied sizes to the nearest offered option: an old bookmark
  // with size=500 must not render a page of 500 rows, and a hand-edited size=0
  // or size=-5 must not reach the API as an invalid limit (or leave the
  // rows-per-page select showing a value it does not offer).
  const rawPageSize = url.getNumber("size")
  const pageSize = PAGE_SIZE_OPTIONS.reduce((best, option) =>
    Math.abs(option - rawPageSize) < Math.abs(best - rawPageSize)
      ? option
      : best,
  )

  // Snapshot the window so a rolling preset does not recompute "now" every render
  // (which would churn the query key). Re-anchored when the range selection changes,
  // and by re-picking the active preset (see `pickPreset`).
  //
  // Both re-anchor effects skip their first run. The `useState` initializers have
  // already snapshotted the window, so re-anchoring on mount only moves `start` by
  // the milliseconds since, which changes `filters`, which trips the page reset
  // below: a bookmarked or shared `?page=3` URL would silently open on page 1.
  const selectionKey = `${range}|${startParam}|${endParam}`
  // One clock reading shared by both initial windows. Taken separately they differ
  // by the milliseconds between the two lines, and since `extentWin` is read second
  // its rolling start is the later one, which made the strict `<` in
  // `winOutsideExtent` true on an ordinary load: the page then framed the window
  // rather than the preset, so no preset was highlighted and a 24h extent bucketed
  // by day instead of by hour. Whether that happened came down to machine load.
  const [mountClock] = useState(Date.now)
  const [win, setWin] = useState(() =>
    resolveWindow(range, startParam, endParam, mountClock),
  )
  // The preset extent (ignoring any brushed bounds) that the timeline histogram
  // spans. Snapshotted like `win`, re-anchored when the preset changes: a brushed
  // sub-window must leave the extent alone, or zooming in would drag the frame
  // (and refetch the histogram) along with it.
  const [extentWin, setExtentWin] = useState(() =>
    resolveExtentWindow(range, mountClock),
  )
  // Both re-anchors share one effect so they also share one clock reading. As two
  // effects they ran in declaration order on a range change, leaving the list
  // window a millisecond behind the extent containing it, which is the same drift
  // described above. The guards are per-window and unchanged: `win` re-anchors for
  // any selection change, `extentWin` only when the preset itself moves.
  const prevSelectionKey = useRef(selectionKey)
  const prevRange = useRef(range)
  useEffect(() => {
    const clock = Date.now()
    if (prevSelectionKey.current !== selectionKey) {
      prevSelectionKey.current = selectionKey
      setWin(resolveWindow(range, startParam, endParam, clock))
    }
    if (prevRange.current !== range) {
      prevRange.current = range
      setExtentWin(resolveExtentWindow(range, clock))
    }
  }, [selectionKey, range, startParam, endParam])

  // An unrecognized `range` resolves to the default window (see `resolveWindow`),
  // so the URL has to be corrected to say what the page is actually showing. The
  // case that produces it is a Usage-page key: `90d` and `12mo` are presets there
  // and not here, so a hand-edited or hand-copied URL carries one across and the
  // address bar then claims ninety days over a list showing one. `patch` replaces
  // rather than pushes, because this is a correction to the entry rather than a
  // navigation, and writing the default drops the key entirely, which is the same
  // URL the page produces when it is opened with no range at all.
  //
  // Bounds win over a preset, so a window carried in `start_date`/`end_date` is
  // left alone: the range is not lying there, it is not being read.
  const patch = url.patch
  useEffect(() => {
    if (startParam || endParam) return
    if (range === CUSTOM_KEY || findPreset(ACTIVITY_PRESETS, range)) return
    patch({ range: ACTIVITY_DEFAULT_KEY })
  }, [range, startParam, endParam, patch])

  const priced =
    pricedFilter === "true"
      ? true
      : pricedFilter === "false"
        ? false
        : undefined

  const { selected: workspace } = useSelectedWorkspace()
  const filters: UsageFilters = useMemo(
    () => ({
      // From the sidebar's switcher, not from a control on this page: it scopes
      // the whole shell, and the operator's own filters sit below it.
      workspace_id: workspace?.workspace_id,
      start_date: win.start,
      end_date: win.end,
      status: statusFilter || undefined,
      model: modelFilters.length > 0 ? modelFilters : undefined,
      user_id: userFilters.length > 0 ? userFilters : undefined,
      api_key_id: apiKeyFilters.length > 0 ? apiKeyFilters : undefined,
      source: sourceFilter || undefined,
      source_label: sessionFilter || undefined,
      endpoint: endpointFilter || undefined,
      provider: providerFilter || undefined,
      tool: (toolFilter || undefined) as UsageFilters["tool"],
      priced,
    }),
    [
      workspace,
      win,
      toolFilter,
      statusFilter,
      modelFilters,
      userFilters,
      apiKeyFilters,
      sourceFilter,
      sessionFilter,
      endpointFilter,
      providerFilter,
      priced,
    ],
  )

  // Any change to the filter set returns to the first page, but not on mount,
  // so a shared URL keeps its page.
  const filtersKey = JSON.stringify(filters)
  const prevFiltersKey = useRef(filtersKey)
  useEffect(() => {
    if (prevFiltersKey.current !== filtersKey) {
      prevFiltersKey.current = filtersKey
      url.patch({ page: 0, request: "" })
    }
  }, [filtersKey, url])

  const usage = useUsageLogs(filters, page, pageSize)
  const count = useUsageCount(filters)

  // Which usage surface this caller reads, and with it the two things on this
  // page that stay the deployment operator's: the in-flight strip and the
  // per-row "price this model" button. The log and its aggregates are
  // organization-scoped for everyone else (otari#837), so the page is theirs;
  // these two are not.
  const scope = useUsageScope()

  // Requests in progress, read unfiltered: the endpoint has no filters, because a
  // request that has not finished has no outcome, cost, or token count to filter
  // on. Reported gateway-wide beside the refresh control, not as rows.
  //
  // Deployment-wide and left that way: its registry entries carry no workspace,
  // so there is nothing in them to scope to a tenant yet. A non-operator is not
  // shown a strip reporting somebody else's traffic, and does not ask for one.
  const inFlight = useInFlightRequests(scope.isDeploymentWide)

  // How far behind the frozen page has fallen. Polled while the count it is
  // compared against is not (see `useUsageCount`), so the difference is "rows that
  // have landed since this page was drawn".
  //
  // Only asked for where it can be acted on. On page 2 onward, refreshing does not
  // bring newer rows into view (they land at the top of page 1), so a badge
  // offering to load them would be a promise the button does not keep; a window
  // that ends in the past can gain no rows at all, so the poll would be pure cost.
  const newRowsRelevant = page === 0 && !filters.end_date
  const liveCount = useLiveUsageCount(filters, newRowsRelevant)

  // Source options: the sources with usage in the window. This must ignore the
  // source filter itself, or picking Claude Code would hide every other source
  // and switching would mean clearing first. The session/endpoint/provider
  // drill-down chips are not applied either, so the list stays complete (and the
  // picker useful) while a drill-down narrows everything else. The model, user
  // and key pickers are the Overview's, which fetches their options.
  const sourceSuggestFilters: UsageFilters = useMemo(
    () => ({
      workspace_id: workspace?.workspace_id,
      start_date: win.start,
      end_date: win.end,
      status: statusFilter || undefined,
      model: modelFilters.length > 0 ? modelFilters : undefined,
      user_id: userFilters.length > 0 ? userFilters : undefined,
      api_key_id: apiKeyFilters.length > 0 ? apiKeyFilters : undefined,
    }),
    [
      workspace?.workspace_id,
      win,
      statusFilter,
      modelFilters,
      userFilters,
      apiKeyFilters,
    ],
  )
  const sourceSummary = useUsageSummary(
    sourceSuggestFilters,
    "day",
    SOURCE_BREAKDOWN,
  )
  // A drill-down can name a source with no rows in the window; keep it listed so
  // the select shows the filter that is actually applied.
  const sourceBreakdown = sourceSummary.data?.by_source
  const sourceOptions = useMemo(() => {
    const seen = (sourceBreakdown ?? [])
      .filter((group) => !group.is_other && group.key !== null)
      .map((group) => group.key as string)
    return sourceFilter && !seen.includes(sourceFilter)
      ? [sourceFilter, ...seen]
      : seen
  }, [sourceBreakdown, sourceFilter])

  // The timeline histogram spans the whole preset *extent* (the rolling preset
  // window, independent of any brushed sub-window), so the brush always has
  // context to zoom back out into. For the unbounded "All", `extentWin` carries an
  // explicit year-long start (see `resolveExtentWindow`) so the bars span a
  // deterministic window instead of the summary endpoint's hidden 30-day default;
  // the list stays all-time and the caption reflects the true list window, and the
  // brush still narrows it. Entity filters carry over so the bars match what's shown.
  const extentPreset =
    findPreset(ACTIVITY_PRESETS, range) ??
    findPreset(ACTIVITY_PRESETS, ACTIVITY_DEFAULT_KEY)
  // A window reaching outside the preset extent (a drill-down from the Usage
  // page carries its own bounds while the URL's `range` still holds a default)
  // cannot be framed by that extent: the histogram would show unrelated bars
  // and the preset would read as active. Frame the window itself instead, with
  // no preset highlighted; zoom-out falls back to the smallest broader preset.
  const winOutsideExtent = Boolean(
    win.start &&
      extentWin.start &&
      new Date(win.start).getTime() < new Date(extentWin.start).getTime(),
  )
  const extentKey = winOutsideExtent ? CUSTOM_KEY : range
  const extentBucket = winOutsideExtent
    ? bucketForWindow(win.start as string, win.end)
    : (extentPreset?.bucket ?? "day")
  const contextFilters: UsageFilters = useMemo(
    () => ({
      workspace_id: workspace?.workspace_id,
      start_date: winOutsideExtent ? win.start : extentWin.start,
      end_date: winOutsideExtent ? win.end : undefined,
      status: statusFilter || undefined,
      model: modelFilters.length > 0 ? modelFilters : undefined,
      user_id: userFilters.length > 0 ? userFilters : undefined,
      api_key_id: apiKeyFilters.length > 0 ? apiKeyFilters : undefined,
      source: sourceFilter || undefined,
      source_label: sessionFilter || undefined,
      endpoint: endpointFilter || undefined,
      provider: providerFilter || undefined,
      tool: (toolFilter || undefined) as UsageFilters["tool"],
      priced,
    }),
    [
      workspace?.workspace_id,
      winOutsideExtent,
      toolFilter,
      win,
      extentWin,
      statusFilter,
      modelFilters,
      userFilters,
      apiKeyFilters,
      sourceFilter,
      sessionFilter,
      endpointFilter,
      providerFilter,
      priced,
    ],
  )
  // The timeline reads `series`; the tool dimension is requested so the Tool filter
  // knows whether this window contains any gateway-run tool calls. With
  // NO_BREAKDOWNS the server returns `by_tool: []` by contract, which left the
  // selector permanently hidden unless a tool filter was already in the URL.
  const contextSummary = useUsageSummary(
    contextFilters,
    extentBucket,
    TOOL_BREAKDOWN,
  )
  const timelineSeries = (contextSummary.data?.series ?? []).map((point) => ({
    bucketStart: point.bucket_start,
    requests: point.requests,
    // Failed requests render as a red segment on the strip, so dropped traffic
    // shows up while browsing, not only after filtering to status=error.
    errors: point.errors ?? 0,
  }))

  const rows = usage.data ?? []

  // What the live control may report. A failed poll leaves the list unknown, not
  // unchanged: TanStack keeps the last successful payload, so without the
  // `isError` arm the count would sit there with its waits climbing against a
  // frozen anchor, asserting that work is running which may have landed minutes
  // ago. That is the state `useInFlightRequests` already refuses to cache across
  // mounts, so it must not be reachable by this route either. The failure reaches
  // the operator through the page's error banner instead.
  const liveNow = inFlight.isError ? undefined : inFlight.data

  // What served each routed request on this page. Built from the page itself where
  // possible (a group's attempts are written milliseconds apart, so they are
  // usually adjacent in a newest-first list), and looked up for the groups whose
  // outcome row is missing: a page boundary splits a group, and filtering to the
  // `absorbed` status (the way an operator investigates fallovers) hides every
  // outcome row by construction, which is precisely when the answer is wanted.
  // A served row is looked up as well when its earlier attempts are off the page,
  // so it can still say what it spilled over from.
  const { pageOutcomes, unresolvedGroupIds } = useMemo(() => {
    const known = indexGroupOutcomes(rows)
    const missing = new Set([
      ...rows.flatMap((row) =>
        row.status === "absorbed" &&
        row.request_group_id &&
        !known.has(row.request_group_id)
          ? [row.request_group_id]
          : [],
      ),
      // A served row whose skipped or failed attempts fell on the next page.
      ...findGroupsMissingEarlierAttempts(rows),
    ])
    return { pageOutcomes: known, unresolvedGroupIds: [...missing] }
  }, [rows])
  const unresolvedGroups = useRequestGroups(unresolvedGroupIds)
  const groupOutcomes = useMemo(() => {
    if (!unresolvedGroups.data?.length) return pageOutcomes
    return new Map([
      ...pageOutcomes,
      ...indexGroupOutcomes(unresolvedGroups.data),
    ])
  }, [pageOutcomes, unresolvedGroups.data])

  const totalIsExact = count.isSuccess && !count.isPlaceholderData
  const total = totalIsExact ? (count.data?.total ?? 0) : null

  // Rows that have landed since this page was drawn, as the gap between the polled
  // count and the pinned one. Clamped at zero because the gap can close from the
  // other side: a delete makes the live figure the smaller one, and "-14 new"
  // is not a thing to show an operator.
  //
  // Gated on the same condition as the poll: `page` is not part of the live count's
  // key, so disabling the query on page 2 stops it refetching but still hands back
  // the payload it cached on page 1. Without this the badge would follow the
  // operator forward and offer rows that pressing it cannot bring into view.
  const newRows =
    newRowsRelevant && total != null && liveCount.data
      ? Math.max(0, liveCount.data.total - total)
      : 0

  // Whether the page can still tell newer rows from none. The badge's absence
  // otherwise reads as "nothing has landed", which on a table that no longer moves
  // by itself makes a flooded gateway look identical to an idle one. Said in the
  // control strip rather than the error banner: the count failing costs the
  // operator a hint, not the log they came for, and `retry: false` means a single
  // blip would otherwise raise a page-level alarm the next poll silently clears.
  const newRowsUnknown = newRowsRelevant && liveCount.isError
  // Neither the default preset nor the unbounded "All" is itself a filter: only an
  // explicit sub-window or a bounded non-default preset narrows the window, so a
  // brand-new gateway reads "never used" on both its 24h default and on "All",
  // and only a real narrowing reads "filtered empty". (UsagePage can mirror this
  // with a bare `!== default` because it has no unbounded preset; Activity does.)
  const rangePreset = findPreset(ACTIVITY_PRESETS, range)
  const timeFiltered =
    Boolean(startParam || endParam) ||
    (range !== ACTIVITY_DEFAULT_KEY && rangePreset?.seconds != null)
  const anyFilter = Boolean(
    statusFilter ||
      modelFilters.length ||
      (observeScope === "everyone" && userFilters.length) ||
      apiKeyFilters.length ||
      pricedFilter ||
      sourceFilter ||
      sessionFilter ||
      endpointFilter ||
      providerFilter ||
      toolFilter ||
      timeFiltered,
  )

  // This page's own filters as removable chips (time is driven by the timeline,
  // and model, user, key and status are the Overview's). Values show the human label
  // where one exists.
  const labelFrom = (
    options: { value: string; label: string }[],
    value: string,
  ) => options.find((option) => option.value === value)?.label ?? value
  const filterChips: FilterChip[] = [
    ...(pricedFilter
      ? [
          {
            key: "priced",
            label: "Priced",
            value: labelFrom(PRICED_OPTIONS, pricedFilter),
            onClear: () => url.patch({ priced: "" }),
          },
        ]
      : []),
    ...(sourceFilter
      ? [
          {
            key: "source",
            label: "Source",
            value: describeSource(sourceFilter),
            onClear: () => url.patch({ source: "" }),
          },
        ]
      : []),
    ...(sessionFilter
      ? [
          {
            key: "session",
            label: "Session",
            value: sessionFilter,
            onClear: () => url.patch({ source_label: "" }),
          },
        ]
      : []),
    ...(endpointFilter
      ? [
          {
            key: "endpoint",
            label: "Endpoint",
            value: endpointFilter,
            onClear: () => url.patch({ endpoint: "" }),
          },
        ]
      : []),
    ...(providerFilter
      ? [
          {
            key: "provider",
            label: "Provider",
            value: providerFilter,
            onClear: () => url.patch({ provider: "" }),
          },
        ]
      : []),
    ...(toolFilter
      ? [
          {
            key: "tool",
            label: "Tool",
            value: labelFrom(TOOL_OPTIONS, toolFilter),
            onClear: () => url.patch({ tool: "" }),
          },
        ]
      : []),
  ]

  const setModelPrice = useSetPricing()
  // Bumped on every open of the price dialog and used as its key, so the
  // rates are cleared on the way in. Clearing them on close would blank the
  // fields while the dialog is still animating away, and these values set
  // money.
  const [priceOpenCount, setPriceOpenCount] = useState(0)
  // The model selector whose price is being set from a request detail, and
  // unset when that dialog is closed.
  const [modelPriceKey, setModelPriceKey] = useState<string>()
  const [modelPriceUnit, setModelPriceUnit] = useState<PricingUnit>("tokens")
  const priceModel = scope.isDeploymentWide
    ? (model: string, endpoint: string) => {
        setPriceOpenCount((count) => count + 1)
        setModelPriceUnit(pricingUnitForEndpoint(endpoint))
        setModelPriceKey(model)
      }
    : undefined

  // The open request lives in the URL, so it is linkable. Opening one from the
  // list is a history entry, so Back closes it; walking and closing replace it.
  const navigate = useNavigate()
  const requestId = url.get("request")
  const openIndex = rows.findIndex((row) => row.id === requestId)
  const openEntry = openIndex >= 0 ? rows[openIndex] : undefined
  const openRequest = (id: string) => {
    if (id === requestId) return
    navigate({
      to: ".",
      search: (prev) => ({ ...(prev as DashboardSearch), request: id }),
    })
  }
  // Previous and next walk the loaded page only: J/K stop at its edges.
  const previousEntry = openIndex > 0 ? rows[openIndex - 1] : undefined
  const nextEntry =
    openIndex >= 0 && openIndex < rows.length - 1
      ? rows[openIndex + 1]
      : undefined
  // A link to a row that is not on this page opens nothing; say so rather than
  // ignore it. Only once the page has settled, or a loading page would flash it.
  const requestMissing =
    requestId !== "" &&
    openEntry === undefined &&
    !usage.isLoading &&
    !usage.isPlaceholderData
  // Below md the panel covers the page, which then takes no focus or presses.
  const isMobile = useIsMobile()
  // Stable per open row, so DataTable's row cache holds while the page polls.
  // `is-peeked` washes the cells (see globals.css) and composes with the row's
  // own failure tint.
  const rowClassName = useCallback(
    (entry: UsageEntry) =>
      [
        getActivityRowClassName(entry),
        entry.id === requestId ? "is-peeked" : undefined,
      ]
        .filter(Boolean)
        .join(" ") || undefined,
    [requestId],
  )

  // Sets the model's own price (a ModelPricing row), which is what future
  // requests are billed at. Deliberately does not touch the rows already logged:
  // a gateway row's cost is what it was served at, and rewriting history from an
  // activity view would move spend that budgets were already enforced against.
  // Awaited by the dialog, which owns the pending and error state below its own
  // key: a refusal cannot then greet the next open. `mutateAsync`, so a
  // rejection reaches it rather than being reported from page state that
  // outlives the form.
  const onSetModelPrice = (rates: ManualRates, modelKey: string) =>
    setModelPrice.mutateAsync(
      {
        model_key: modelKey,
        input_price_per_million: rates.input_price_per_million,
        output_price_per_million: rates.output_price_per_million,
        cache_read_price_per_million:
          rates.cache_read_price_per_million ?? null,
        cache_write_price_per_million:
          rates.cache_write_price_per_million ?? null,
        unit: rates.unit ?? "tokens",
      },
      { onSuccess: () => setModelPriceKey(undefined) },
    )

  // Refresh means "same view, newer rows", so it deliberately does *not* re-anchor
  // a rolling preset's window. Re-anchoring recomputed "now", which changed
  // `filters`, which tripped the page reset below: pressing refresh on page 12
  // dropped the operator back onto page 1, making deep browsing unusable. The
  // window is instead re-anchored when the range selection changes (see
  // `pickPreset`). Every window-scoped query is refetched explicitly (see the
  // UsagePage refresh note), since the keys are unchanged by design here.
  const refresh = () => {
    void usage.refetch()
    void count.refetch()
    // Re-read alongside the count it is compared against, or the badge would keep
    // offering rows the refresh just loaded. Guarded for the same reason as the
    // source summary below: refetch() ignores `enabled`.
    if (newRowsRelevant) {
      void liveCount.refetch()
    }
    void inFlight.refetch()
    void contextSummary.refetch()
    void sourceSummary.refetch()
  }

  // A rolling preset clears any explicit bounds; a timeline selection sets them
  // (the preset key is left as-is, since it still names the extent).
  const pickPreset = (preset: RangePreset) => {
    // Re-picking the preset that is already active is the explicit "re-anchor to
    // now" gesture. It leaves the URL untouched, so the effect that snapshots the
    // window never fires; do it here instead. (Going *to* a different preset, or
    // off an explicit range, changes the URL and that effect handles it, so
    // re-anchoring here as well would fire a second query for the same view.)
    if (preset.key === range && !startParam && !endParam) {
      // Both windows off one reading, for the same reason as at mount: two reads
      // would leave the list window a millisecond behind the extent it sits in.
      const clock = Date.now()
      setWin(resolveWindow(preset.key, "", "", clock))
      setExtentWin(resolveExtentWindow(preset.key, clock))
      return
    }
    url.patch({ range: preset.key, start_date: "", end_date: "" })
  }
  const pickCustom = (startIso: string, endIso: string) =>
    url.patch({ start_date: startIso, end_date: endIso })

  // Memoized on its per-render inputs (the roster labels and the routing
  // outcomes, both themselves memoized) so DataTable's per-row cache holds: a
  // fresh array every render would rebuild all rows per click.
  const columns = useMemo<DataTableColumn<UsageEntry>[]>(
    () => [
      TIME_COLUMN,
      {
        id: "user",
        header: "User",
        cell: (entry) => {
          if (entry.user_id === null) return "—"
          const { label } = userDisplay(
            entry.user_id,
            entry.user_alias,
            memberLabels,
          )
          return (
            <span className="block truncate" title={label}>
              {label}
            </span>
          )
        },
      },
      MODEL_COLUMN,
      {
        id: "routing",
        header: "Routing",
        // The policy the caller named, plus where this row sits in its plan and how
        // that turned out. The Model column keeps meaning the model that actually
        // ran (it is the join key for filters and for spend-by-model), so this is
        // additive: together they answer "what did I ask for, and what served it".
        cell: (entry) => (
          <RoutingCell
            entry={entry}
            outcome={groupOutcomes.get(entry.request_group_id ?? "") ?? null}
          />
        ),
      },
      API_KEY_COLUMN,
      INPUT_TOKENS_COLUMN,
      OUTPUT_TOKENS_COLUMN,
      COST_COLUMN,
      {
        id: "latency",
        header: "Total time",
        align: "end",
        cell: (entry) => formatLatencyCell(entry.latency_ms),
      },
      STATUS_COLUMN,
    ],
    [groupOutcomes, memberLabels],
  )

  return (
    <div className="flex flex-col">
      <div
        className="flex flex-col"
        inert={openEntry !== undefined && isMobile}
      >
        {/* Inside the Overview the layout owns the heading. */}
        {embedded ? null : (
          <PageIntro title="Activity">
            A per-request log of what the gateway served: tokens, cost, latency,
            and failures. No request or response content is stored.
          </PageIntro>
        )}

        {/* The timeline's summary error is included so a failed series request
          reads as a failure, not as an empty "No activity in this range" strip. */}
        {/* The in-flight error is last: a failure to read the log itself is the more
          important thing to say. It is here at all so a live view that has gone
          quiet is distinguishable from a gateway that has, since the rows are
          dropped on failure rather than left to go stale. */}
        <ErrorBanner
          error={
            usage.error ?? count.error ?? contextSummary.error ?? inFlight.error
          }
        />

        <div className="flex flex-col gap-3">
          <ActivityTimeline
            presets={ACTIVITY_PRESETS}
            extentKey={extentKey}
            onPreset={pickPreset}
            onSelectRange={pickCustom}
            onSelectFull={() =>
              extentPreset ? pickPreset(extentPreset) : undefined
            }
            series={timelineSeries}
            bucket={extentBucket}
            windowStart={win.start}
            windowEnd={win.end}
            loading={contextSummary.isLoading}
            ariaLabel="Activity request volume over the selected window"
            action={
              <span className="inline-flex items-center gap-2">
                {/* Both live signals sit here, beside the control that acts on them:
                  the table below never moves on its own, so this strip is the only
                  place the page says anything is still happening. */}
                {/* Rendered whenever the poll is answering at all, not only when it
                  reports traffic: the control hides itself while idle, but has to
                  stay mounted to keep an open list open across the moment the last
                  request lands. It still goes on a failed poll, where `liveNow` is
                  undefined, since there is nothing trustworthy left to report. */}
                {liveNow ? (
                  <InFlightControl
                    data={liveNow}
                    updatedAt={inFlight.dataUpdatedAt}
                  />
                ) : null}
                {newRows > 0 ? (
                  <Button
                    size="sm"
                    variant="ghost"
                    onPress={refresh}
                    isDisabled={usage.isFetching}
                  >
                    {formatNumber(newRows)} new · load
                  </Button>
                ) : null}
                {newRowsUnknown ? (
                  <span
                    className="text-caption"
                    title="The row count could not be read, so this page cannot tell whether newer requests have landed. Refresh to load whatever is there."
                  >
                    Newer rows unknown
                  </span>
                ) : null}
                <RefreshButton
                  onRefresh={refresh}
                  isFetching={usage.isFetching}
                  updatedAt={usage.dataUpdatedAt}
                />
              </span>
            }
          />
          <FilterChips chips={filterChips} toggleLabel="More filters">
            <FilterSelect
              label="Priced?"
              value={pricedFilter}
              onChange={(value) => url.patch({ priced: value })}
              options={PRICED_OPTIONS}
            />
            {/* Only offered once the window actually contains tool usage, following the
              source select below: a filter whose every option returns nothing is
              noise on the majority of gateways, which run no tools at all. */}
            {toolFilter || contextSummary.data?.by_tool?.length ? (
              <FilterSelect
                label="Tool"
                value={toolFilter}
                onChange={(value) => url.patch({ tool: value })}
                options={TOOL_OPTIONS}
              />
            ) : null}
            {/* Provenance only earns a select once there is more than one source
              to choose between: most gateways see only their own traffic, and a
              filter with a single option is noise. A drill-down that arrives
              with a source applied keeps the select so it stays clearable. */}
            {sourceOptions.length > 1 || sourceFilter ? (
              <FilterSelect
                label="Source"
                value={sourceFilter}
                onChange={(value) => url.patch({ source: value })}
                options={[
                  { value: "", label: "All" },
                  ...sourceOptions.map((source) => ({
                    value: source,
                    label: describeSource(source),
                  })),
                ]}
              />
            ) : null}
          </FilterChips>
        </div>

        <div className="flex flex-col gap-3">
          {requestMissing ? (
            <div className="flex flex-wrap items-center gap-3">
              <InfoBanner>
                That request is not in the current results.
              </InfoBanner>
              <Button
                size="sm"
                variant="ghost"
                onPress={() => url.patch({ request: "" })}
              >
                Dismiss
              </Button>
            </div>
          ) : null}
          <TableScrollFrame className="otari-activity-table">
            <DataTable
              mobileLayout="cards"
              ariaLabel="Activity log"
              columns={columns}
              rows={rows}
              getRowKey={getActivityRowKey}
              isLoading={usage.isLoading}
              emptyContent={
                anyFilter
                  ? "No requests match these filters."
                  : "No requests recorded yet."
              }
              onRowAction={openRequest}
              rowClassName={rowClassName}
            />
          </TableScrollFrame>

          <TablePagination
            page={page}
            pageSize={pageSize}
            total={total}
            rowsOnPage={rows.length}
            // Paging re-reads the count as well as the rows. The total is a property of
            // the filters, not of the page, so it is not in the count's key and a frozen
            // page would carry whichever value it loaded with. That understates a table
            // traffic has grown, and `TablePagination` derives `isLast` from the total
            // whenever it has one, so the operator hits a wall short of the real end and
            // the oldest rows sit past it, unreachable until a manual refresh. Paging is
            // a deliberate act, so re-reading here keeps the total describing a set the
            // operator can actually navigate, without putting a self-moving number under
            // a table that deliberately holds still.
            onPageChange={(next) => {
              url.patch({ page: next, request: "" })
              void count.refetch()
            }}
            onPageSizeChange={(size) =>
              url.patch({ size, page: 0, request: "" })
            }
            isFetching={usage.isFetching}
            hasNextFallback={rows.length === pageSize}
          />
        </div>
      </div>

      {openEntry ? (
        <RequestPeek
          entry={openEntry}
          onPriceModel={priceModel}
          onClose={() => url.patch({ request: "" })}
          onPrevious={
            previousEntry
              ? () => url.patch({ request: previousEntry.id })
              : undefined
          }
          onNext={
            nextEntry ? () => url.patch({ request: nextEntry.id }) : undefined
          }
        />
      ) : null}

      <SetPriceDialog
        key={`model-${priceOpenCount}`}
        isOpen={modelPriceKey !== undefined}
        onOpenChange={(open) =>
          setModelPriceKey(open ? (modelPriceKey ?? "") : undefined)
        }
        onSubmit={onSetModelPrice}
        submitLabel="Price this model"
        collectModelKey
        initialModelKey={modelPriceKey ?? ""}
        chooseUnit
        initialUnit={modelPriceUnit}
        title="Price this model"
        description="Set what this model costs, taken from the request you were looking at. Requests from now on are costed at these rates and counted against budgets; rows already logged keep the cost they were served with."
      />
    </div>
  )
}
