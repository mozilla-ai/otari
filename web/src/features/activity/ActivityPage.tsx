import { Button } from "@heroui/react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import type {
  SummaryDimension,
  UsageEntry,
  UsageFilters,
  UsageGroupRow,
  UsageMutationSelection,
} from "@/client"
import { RefreshButton } from "@/design-system/actions/RefreshButton"
import { BulkActionBar } from "@/design-system/data/BulkActionBar"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import {
  PAGE_SIZE_OPTIONS,
  TablePagination,
} from "@/design-system/data/TablePagination"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Dot } from "@/design-system/indicators/Dot"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import {
  type FilterChip,
  FilterChips,
} from "@/design-system/navigation/FilterChips"
import { FilterMultiComboBox } from "@/design-system/navigation/FilterMultiComboBox"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import {
  type ManualRates,
  SetPriceDialog,
} from "@/features/models/SetPriceDialog"
import { useMemberAttributionLabels } from "@/features/organization/attribution"
import { userDisplay } from "@/features/users/userDisplay"
import { useSetPricing } from "@/shared/api/pricing"
import {
  useDeleteUsage,
  useInFlightRequests,
  useLiveUsageCount,
  useRequestGroups,
  useSetUsagePrice,
  useUsageCount,
  useUsageLogs,
  useUsageScope,
  useUsageSummary,
} from "@/shared/api/usage"
import {
  formatDateTime,
  formatNumber,
  formatRelative,
} from "@/shared/helpers/format"
import {
  resolveSelectedIds,
  useTableSelection,
} from "@/shared/helpers/tableSelection"
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
  formatLatencyCell,
  formatToolUsage,
  formatUSD,
  getActivityRowClassName,
  getActivityRowKey,
  indexGroupOutcomes,
  listToolUsage,
  PRICED_OPTIONS,
  resolveExtentWindow,
  resolveWindow,
  STATUS_OPTIONS,
  TOOL_OPTIONS,
} from "./activityModel"
import { InFlightControl } from "./InFlightControl"
import { RequestDetail } from "./RequestDetail"
import { RoutingCell } from "./RoutingCell"
import { StatusMark } from "./StatusMark"
import { TokenBar } from "./TokenBar"

// The only breakdown this page asks the summary for: whether the window contains
// gateway-run tool calls, which decides if the Tool filter is worth offering.
const TOOL_BREAKDOWN: SummaryDimension[] = ["tool"]

const DEFAULT_PAGE_SIZE = 50

// The only breakdown this page reads: the in-window models behind the typeahead.
// The typeahead reads `by_model`; the source picker's option list piggybacks on
// the same query's `by_source` while no source is picked (see the source
// suggestion note below), so both breakdowns ride one request.
const MODEL_AND_SOURCE_BREAKDOWNS: SummaryDimension[] = ["model", "source"]

// The user and key pickers read these two. by_user and by_api_key carry each
// entity's display name, resolved server-side in the same GROUP BY, so naming an
// option costs nothing beyond the breakdown itself. The alternative is paging
// the whole users and api_keys tables on every visit.
const ENTITY_BREAKDOWNS: SummaryDimension[] = ["user", "api_key"]
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
} as const

export function ActivityPage() {
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
  const userFilters = url.getAll("user_id")
  const apiKeyFilters = url.getAll("api_key_id")
  const pricedFilter = url.get("priced")
  // Provenance: gateway traffic vs an imported agent source. Set by its own select,
  // or by a drill-down (the pricing alarm links here scoped to gateway traffic).
  const sourceFilter = url.get("source")
  // The Usage-page secondary breakdowns (session / endpoint / provider) drill in
  // the same way: no select of their own, carried as a chip so the scoping is
  // visible and one click removes it.
  const sessionFilter = url.get("source_label")
  const endpointFilter = url.get("endpoint")
  const providerFilter = url.get("provider")
  const toolFilter = url.get("tool")
  const page = Math.max(0, url.getNumber("page"))
  // Snap URL-supplied sizes to the nearest offered option: selection latency
  // grows linearly with rows on the page, so an old bookmark with size=500
  // must not resurrect second-long checkbox clicks, and a hand-edited size=0
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

  const selection = useTableSelection()

  // Any change to the filter set returns to the first page and drops the
  // selection, but not on mount, so a shared URL keeps its page.
  const filtersKey = JSON.stringify(filters)
  const prevFiltersKey = useRef(filtersKey)
  useEffect(() => {
    if (prevFiltersKey.current !== filtersKey) {
      prevFiltersKey.current = filtersKey
      url.patch({ page: 0 })
      selection.clear()
    }
  }, [filtersKey, url, selection])

  const usage = useUsageLogs(filters, page, pageSize)
  const count = useUsageCount(filters)

  // Which usage surface this caller reads, and with it the three things on this
  // page that stay the deployment operator's: the in-flight strip, the bulk
  // delete / reprice controls, and the per-row "price this model" button. The
  // log and its aggregates are organization-scoped for everyone else
  // (otari#837), so the page is theirs; these three are not.
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

  // Model suggestions: models with usage in the window (other filters applied, the
  // model filter omitted so the full list stays offered).
  const modelSuggestFilters: UsageFilters = useMemo(
    () => ({
      workspace_id: workspace?.workspace_id,
      start_date: win.start,
      end_date: win.end,
      status: statusFilter || undefined,
      user_id: userFilters.length > 0 ? userFilters : undefined,
      api_key_id: apiKeyFilters.length > 0 ? apiKeyFilters : undefined,
      source: sourceFilter || undefined,
      source_label: sessionFilter || undefined,
      endpoint: endpointFilter || undefined,
      provider: providerFilter || undefined,
      tool: (toolFilter || undefined) as UsageFilters["tool"],
    }),
    [
      workspace?.workspace_id,
      win,
      statusFilter,
      userFilters,
      apiKeyFilters,
      sourceFilter,
      sessionFilter,
      endpointFilter,
      providerFilter,
      toolFilter,
    ],
  )
  // Two breakdowns are read here (model typeahead, source picker); the rest are
  // not requested.
  const modelSummary = useUsageSummary(
    modelSuggestFilters,
    "day",
    MODEL_AND_SOURCE_BREAKDOWNS,
  )
  const realGroups = (rows: UsageGroupRow[] | undefined) =>
    (rows ?? []).filter((group) => !group.is_other && group.key !== null)
  const modelOptions = realGroups(modelSummary.data?.by_model).map(
    (group) => group.key as string,
  )

  // The user and key pickers need their own window: each must keep offering the
  // *other* values of its own dimension, so both entity filters come off. That
  // cannot share the model/source query above, which has to keep them applied,
  // or filtering Activity to one user would make the typeahead suggest only the
  // models other users called and picking one would return an empty table.
  const entitySuggestFilters: UsageFilters = useMemo(
    () => ({ ...filters, user_id: undefined, api_key_id: undefined }),
    [filters],
  )
  const entitySummary = useUsageSummary(
    entitySuggestFilters,
    "day",
    ENTITY_BREAKDOWNS,
  )
  const keyOptions = realGroups(entitySummary.data?.by_api_key).map(
    (group) => ({
      value: group.key as string,
      label: group.label ?? `${(group.key as string).slice(0, 8)}…`,
    }),
  )

  // Source options: the sources with usage in the window. Like the model
  // suggestions, this must ignore the source filter itself, or picking Claude Code
  // would hide every other source and switching would mean clearing first.
  //
  // While no source is picked the model-suggestion summary is already computed
  // without one, so its provenance breakdown is that full list and no second query
  // is needed. Only a picked source needs its own, so it is fetched then and only
  // then (each summary runs four grouped aggregations plus the series).
  // Deliberately broader than modelSuggestFilters: the session/endpoint/provider
  // drill-down chips are not applied here, so the source list stays complete
  // (and the picker stays useful) while a drill-down narrows everything else.
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
    Boolean(sourceFilter),
  )
  const sourceBreakdown = (
    sourceFilter ? sourceSummary.data : modelSummary.data
  )?.by_source
  // A drill-down can name a source with no rows in the window; keep it listed so
  // the select shows the filter that is actually applied.
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
  const { pageOutcomes, unresolvedGroupIds } = useMemo(() => {
    const known = indexGroupOutcomes(rows)
    const missing = new Set(
      rows.flatMap((row) =>
        row.status === "absorbed" &&
        row.request_group_id &&
        !known.has(row.request_group_id)
          ? [row.request_group_id]
          : [],
      ),
    )
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
  // other side: a bulk delete makes the live figure the smaller one, and "-14 new"
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
      userFilters.length ||
      apiKeyFilters.length ||
      pricedFilter ||
      sourceFilter ||
      sessionFilter ||
      endpointFilter ||
      providerFilter ||
      toolFilter ||
      timeFiltered,
  )

  // Active entity filters as removable chips (time is driven by the timeline, so
  // it is not a chip). Values show the human label where one exists.
  const labelFrom = (
    options: { value: string; label: string }[],
    value: string,
  ) => options.find((option) => option.value === value)?.label ?? value
  // Name first, id in parentheses: the id is what the filter submits, and two
  // people can share a name. Resolved the way the User column resolves it, so
  // the same person reads the same in the picker, the chip and the row.
  const userOptionsList = realGroups(entitySummary.data?.by_user).map(
    (group) => {
      const name = userDisplay(group.key as string, group.label, memberLabels)
      return {
        value: group.key as string,
        label: name.id ? `${name.label} (${name.id})` : name.label,
      }
    },
  )
  const clearEntityFilters = () =>
    url.patch({
      status: "",
      priced: "",
      model: [],
      user_id: [],
      api_key_id: [],
      source: "",
      source_label: "",
      endpoint: "",
      provider: "",
      tool: "",
    })
  // One chip per picked value of a repeatable filter, each clearing only itself.
  const valueChips = (
    dimension: string,
    label: string,
    param: "model" | "user_id" | "api_key_id",
    values: string[],
    display: (value: string) => string,
  ): FilterChip[] =>
    values.map((value) => ({
      key: `${dimension}:${value}`,
      label,
      value: display(value),
      // Several chips share a dimension, so the value has to be part of the name.
      clearLabel: `Remove ${label} filter ${display(value)}`,
      onClear: () =>
        url.patch({
          [param]: values.filter((optionValue) => optionValue !== value),
        }),
    }))
  const filterChips: FilterChip[] = [
    ...(statusFilter
      ? [
          {
            key: "status",
            label: "Status",
            value: labelFrom(STATUS_OPTIONS, statusFilter),
            onClear: () => url.patch({ status: "" }),
          },
        ]
      : []),
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
    ...valueChips("user", "User", "user_id", userFilters, (value) =>
      labelFrom(userOptionsList, value),
    ),
    ...valueChips("model", "Model", "model", modelFilters, (value) => value),
    ...valueChips("key", "API key", "api_key_id", apiKeyFilters, (value) =>
      labelFrom(keyOptions, value),
    ),
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

  // Selection targets imported rows only. `bulk_editable` is the server's own answer
  // to "can a bulk delete or set-price reach this row", so the checkbox and the
  // mutation's WHERE clause cannot disagree; deriving it here from
  // `counts_toward_budget` alone offered a checkbox for budget-exempt gateway traffic
  // that the delete then silently skipped (#781).
  const selectableKeys = useMemo(
    () => rows.filter((entry) => entry.bulk_editable).map((entry) => entry.id),
    [rows],
  )
  const disabledKeys = useMemo(
    () => rows.filter((entry) => !entry.bulk_editable).map((entry) => entry.id),
    [rows],
  )
  const selectedIds = resolveSelectedIds(selection.selectedKeys, selectableKeys)
  const pageSelectedCount = selectedIds.length
  const hasSelection = selection.allMatching || pageSelectedCount > 0

  // Total imported rows matching the filter, for the "select all N" affordance
  // and the bulk-op copy; only fetched once there is a selection.
  const importedFilters = useMemo<UsageFilters>(
    () => ({ ...filters, counts_toward_budget: false }),
    [filters],
  )
  const importedCount = useUsageCount(importedFilters, hasSelection)
  const matchingTotal = importedCount.isSuccess
    ? (importedCount.data?.total ?? null)
    : null
  const allPageSelected =
    selectableKeys.length > 0 && pageSelectedCount === selectableKeys.length
  const canSelectAllMatching =
    allPageSelected &&
    matchingTotal != null &&
    matchingTotal > pageSelectedCount
  const effectiveCount = selection.allMatching
    ? (matchingTotal ?? pageSelectedCount)
    : pageSelectedCount
  // Only offer the selection column when something on the page can actually be
  // selected. A deployment with no imported usage has none: every checkbox would
  // render disabled, which reads as a broken control rather than as "these rows
  // are not eligible". Kept while "all matching" is live so the affordance does
  // not vanish under an operator mid-bulk-op.
  // Deleting and repricing usage rows are deployment-wide writes, so a caller
  // who cannot make them is not offered the selection that leads to them: the
  // bulk bar is rendered off `hasSelection`, which this gates.
  const showSelection =
    scope.isDeploymentWide &&
    (selectableKeys.length > 0 || selection.allMatching)

  const deleteUsage = useDeleteUsage()
  const setPrice = useSetUsagePrice()
  const setModelPrice = useSetPricing()
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [priceOpen, setPriceOpen] = useState(false)
  // Bumped on every open of either price dialog and used as its key, so the
  // rates are cleared on the way in. Clearing them on close would blank the
  // fields while the dialog is still animating away, and these values set
  // money.
  const [priceOpenCount, setPriceOpenCount] = useState(0)
  // The model selector whose price is being set from a request detail, and
  // unset when that dialog is closed. Distinct from `priceOpen` above, which
  // reprices already-logged imported rows rather than setting a model's price.
  const [modelPriceKey, setModelPriceKey] = useState<string>()
  const [expandedId, setExpandedId] = useState<string>()

  // Inline accordion panel under the clicked row (DataTable renderDetail).
  // Was a setter-only closure with empty dependencies, so the row cache held for
  // the life of the page; see the DataTable docstring. It now depends on the
  // operator flag as well, which is not a setter and cannot be read from the
  // closure: it is false until the organization context resolves, so a stale one
  // would leave a real operator without the price control for the session. It
  // flips at most once, so the cache it invalidates is rebuilt once.
  const renderDetail = useCallback(
    (entry: UsageEntry) => (
      <div>
        <div className="flex items-center justify-between border-b border-border px-4 py-2">
          <span className="text-body">Request detail</span>
          <Button
            size="sm"
            variant="ghost"
            onPress={() => setExpandedId(undefined)}
          >
            Close
          </Button>
        </div>
        <RequestDetail
          entry={entry}
          onPriceModel={
            scope.isDeploymentWide
              ? (model) => {
                  setPriceOpenCount((count) => count + 1)
                  setModelPriceKey(model)
                }
              : null
          }
        />
      </div>
    ),
    [scope.isDeploymentWide],
  )

  // A bulk op targets either the current page selection (ids) or, once the operator
  // opted into "all matching", the filter itself (by_filter). The server scopes
  // either to imported rows.
  //
  // Every filter that scopes the table has to be forwarded here. "All matching" is
  // counted client-side from the full filter set but re-derived server-side from
  // this body, so any filter left out widens the delete/reprice past the rows the
  // operator was shown (a session drill-down would wipe every other session).
  //
  // The entity filters travel as the sets they are: the selection body takes the same
  // repeatable form as the read filters, so "all N matching" targets exactly the rows
  // the count was taken over (see UsageSelection).
  const selectionBody = (): UsageMutationSelection =>
    selection.allMatching
      ? {
          by_filter: true,
          // First, because it is the widest of them: the switcher scopes the
          // whole shell, so leaving it out would delete or reprice every other
          // workspace's imported rows from a view the operator had narrowed to
          // one. `UsageSelection.workspace_id` exists on the server for exactly
          // this case.
          workspace_id: filters.workspace_id,
          model: filters.model,
          user_id: filters.user_id,
          api_key_id: filters.api_key_id,
          status: filters.status,
          source: filters.source,
          source_label: filters.source_label,
          endpoint: filters.endpoint,
          provider: filters.provider,
          tool: filters.tool,
          start_date: filters.start_date,
          end_date: filters.end_date,
          priced: filters.priced,
        }
      : { ids: selectedIds }

  const onDeleteConfirm = () => {
    deleteUsage.mutate(selectionBody(), {
      onSuccess: () => {
        setDeleteOpen(false)
        selection.clear()
      },
    })
  }

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
      },
      { onSuccess: () => setModelPriceKey(undefined) },
    )

  const onSetPrice = (rates: ManualRates) =>
    setPrice.mutateAsync(
      { ...selectionBody(), ...rates },
      {
        onSuccess: () => {
          setPriceOpen(false)
          selection.clear()
        },
      },
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
    void modelSummary.refetch()
    void entitySummary.refetch()
    // Guarded because refetch() ignores `enabled`: without a picked source the
    // query is disabled by design and refetching it would fire a pointless
    // extra summary request.
    if (sourceFilter) {
      void sourceSummary.refetch()
    }
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
  const columns = useMemo<DataTableColumn<UsageEntry>[]>(() => {
    const apiKeyLabel = (entry: UsageEntry): string =>
      entry.api_key_id === null
        ? "—"
        : (entry.api_key_name ?? `${entry.api_key_id.slice(0, 8)}…`)
    return [
      {
        id: "time",
        header: "Time",
        // Relative time reads better in a scan than a full timestamp; the
        // absolute value stays available as a tooltip.
        cell: (entry) => (
          <span title={formatDateTime(entry.timestamp)} className="text-muted">
            {formatRelative(entry.timestamp)}
          </span>
        ),
      },
      {
        id: "user",
        header: "User",
        cell: (entry) =>
          entry.user_id === null
            ? "—"
            : userDisplay(entry.user_id, entry.user_alias, memberLabels).label,
      },
      {
        id: "model",
        header: "Model",
        isRowHeader: true,
        // The tool marker lives here rather than in a column of its own: a ninth
        // column would compete with the token bar for the row's only graphic slot
        // and push the failure-forward Status pill off a narrow viewport. Text, not
        // color alone, so it survives the same accessibility bar as TokenBar.
        cell: (entry) => {
          const tools = listToolUsage(entry)
          if (!tools.length) return entry.model
          const calls = tools.reduce(
            (sum, tool) => sum + tool.billed + tool.errors,
            0,
          )
          const detail = tools.map(formatToolUsage).join(" \u00b7 ")
          return (
            <span className="inline-flex items-center gap-1.5">
              {entry.model}
              {/* A marker, not a badge: an accent dot and the count in mono
                  uppercase, on the same terms as every other marker in the
                  product. A generic span does not reliably expose aria-label,
                  so it takes the img role; the label is the whole meaning and
                  the count inside is a summary of it. */}
              <span
                role="img"
                className="inline-flex items-center gap-1.5 text-mono-overline text-muted"
                title={detail}
                aria-label={`Gateway tools: ${detail}`}
              >
                <Dot className="bg-accent" />
                {calls} {calls === 1 ? "tool" : "tools"}
              </span>
            </span>
          )
        },
      },
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
      {
        id: "api_key",
        header: "API key",
        cell: (entry) => (
          <span className="text-muted">{apiKeyLabel(entry)}</span>
        ),
      },
      {
        id: "tokens",
        header: "Tokens",
        align: "end",
        cell: (entry) => <TokenBar entry={entry} />,
      },
      {
        id: "cost",
        header: "Cost",
        align: "end",
        cell: (entry) => formatUSD(entry.cost),
      },
      {
        id: "latency",
        header: "Total time",
        align: "end",
        cell: (entry) => formatLatencyCell(entry.latency_ms),
      },
      {
        id: "status",
        header: "Status",
        cell: (entry) => <StatusMark status={entry.status} />,
      },
    ]
  }, [groupOutcomes, memberLabels])

  return (
    <div className="flex flex-col">
      <PageIntro title="Activity">
        A per-request log of what the gateway served: tokens, cost, latency, and
        failures. No request or response content is stored.
      </PageIntro>

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
        <FilterChips chips={filterChips} onClearAll={clearEntityFilters}>
          <FilterSelect
            label="Status"
            value={statusFilter}
            onChange={(value) => url.patch({ status: value })}
            options={STATUS_OPTIONS}
          />
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
          {/* allowsCustom on all three: the options are the in-window top spenders
              (a breakdown capped at 100), so an entity that exists but ranks below
              that, or has no traffic in the window, is not offered. Enter commits a
              pasted id anyway, the way the Model box already accepts a name the
              suggestions do not cover. */}
          <FilterMultiComboBox
            label="API key"
            values={apiKeyFilters}
            onChange={(values) => url.patch({ api_key_id: values })}
            allowsCustom
            placeholder="All keys"
            options={keyOptions}
          />
          <FilterMultiComboBox
            label="User"
            values={userFilters}
            onChange={(values) => url.patch({ user_id: values })}
            allowsCustom
            placeholder="All users"
            options={userOptionsList}
          />
          <FilterMultiComboBox
            label="Model"
            values={modelFilters}
            onChange={(values) => url.patch({ model: values })}
            allowsCustom
            placeholder="Any model"
            options={modelOptions.map((model) => ({
              value: model,
              label: model,
            }))}
          />
        </FilterChips>
      </div>

      {/* Both, not just `hasSelection`: that reads the selection state, which
          does not clear when the operator answer flips. An operator who selects
          rows and is then found not to be one (the context refetches on its own
          cadence) would keep Delete and Set price on screen over a hidden
          checkbox column. The server refuses either way, so this is a confusing
          control rather than an escalation, but it is one nobody should be
          offered. */}
      {showSelection && hasSelection ? (
        <BulkActionBar
          selectedCount={effectiveCount}
          allMatching={selection.allMatching}
          matchingTotal={matchingTotal}
          canSelectAllMatching={canSelectAllMatching}
          onSelectAllMatching={selection.enableAllMatching}
          onClear={selection.clear}
        >
          <Button
            size="sm"
            variant="primary"
            onPress={() => {
              setPriceOpenCount((count) => count + 1)
              setPriceOpen(true)
            }}
          >
            Set price
          </Button>
          <Button
            size="sm"
            variant="danger"
            onPress={() => setDeleteOpen(true)}
          >
            Delete
          </Button>
        </BulkActionBar>
      ) : null}

      <TableScrollFrame className="otari-activity-table">
        <DataTable
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
          selectionMode={showSelection ? "multiple" : "none"}
          selectedKeys={selection.selectedKeys}
          onSelectionChange={selection.onSelectionChange}
          disabledKeys={disabledKeys}
          onRowAction={(key) =>
            setExpandedId((current) => (current === key ? undefined : key))
          }
          rowClassName={getActivityRowClassName}
          detailKey={expandedId}
          renderDetail={renderDetail}
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
          url.patch({ page: next })
          void count.refetch()
        }}
        onPageSizeChange={(size) => url.patch({ size, page: 0 })}
        isFetching={usage.isFetching}
        hasNextFallback={rows.length === pageSize}
      />

      <ConfirmDialog
        isOpen={deleteOpen}
        onOpenChange={setDeleteOpen}
        heading="Delete usage rows"
        body={`Delete ${formatNumber(effectiveCount)} imported ${
          effectiveCount === 1 ? "row" : "rows"
        }? Only imported rows are removed, and this cannot be undone.`}
        confirmLabel="Delete"
        isPending={deleteUsage.isPending}
        error={deleteUsage.error}
        onConfirm={onDeleteConfirm}
      />

      {/* Keyed on the open count, so each open remounts a blank form. */}
      <SetPriceDialog
        key={`repricing-${priceOpenCount}`}
        isOpen={priceOpen}
        onOpenChange={setPriceOpen}
        targetCount={effectiveCount}
        onSubmit={onSetPrice}
        submitLabel="Set price"
        // The object rather than the verb: the submit and its trigger are both
        // "Set price", so the default title was a third copy of that string.
        title="Imported row costs"
      />

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
        title="Price this model"
        description={() =>
          "Set what this model costs, taken from the request you were looking at. Requests from now on are costed at these rates and counted against budgets; rows already logged keep the cost they were served with."
        }
      />
    </div>
  )
}
