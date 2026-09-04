import { Button } from "@heroui/react"
import type { LinkProps } from "@tanstack/react-router"
import { Link, useNavigate } from "@tanstack/react-router"
import type { ReactNode } from "react"
import { useEffect, useMemo, useState } from "react"
import type { UsageEntry } from "@/client"
import { SetupGuideCard } from "@/features/onboarding/SetupGuideCard"
import { isDeploymentOperator } from "@/features/organization/roles"
import {
  budgetHealth,
  errorRateHealth,
  providerHealthStatus,
  toStatStatus,
} from "@/features/overview/overview"
import {
  NO_BREAKDOWNS,
  useBudgets,
  useKeys,
  useOrganizationContext,
  useOrganizationMembers,
  useProviderHealth,
  useProviders,
  useUsageLogs,
  useUsageSummary,
  useUsers,
} from "@/shared/api/hooks"
import { Sparkline } from "@/shared/components/charts"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import {
  Dot,
  KpiCell,
  KpiStrip,
  Section,
  SpendMeter,
} from "@/shared/components/surface"
import { TrendChip } from "@/shared/components/TrendChip"
import {
  ErrorBanner,
  PageHeader,
  PageLoading,
  RefreshButton,
  // Still reached by `UsageStatTiles`, which the organization overview seats
  // beside its own tiles. The operator overview's KPI strip replaced its use
  // of these, the organization one's has not been rebuilt yet.
  StatCard,
} from "@/shared/components/ui"
import {
  deltaFraction,
  formatNumber,
  formatPct,
  formatRelative,
  formatUsd,
} from "@/shared/helpers/format"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useSurfaces } from "@/shared/hooks/useDeployment"

const DAY_MS = 86_400_000
const PERIOD_DAYS = 30

// The operator's current local date, as a stable key. Used to hold the window
// bounds steady within a day (so query keys don't churn every render) while still
// letting them advance across midnight.
export function localDayKey(): string {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`
}

// Window bounds for the summary queries. Held stable within a local day so the
// query keys don't churn (and the vs-prev delta compares a fixed baseline), but
// advanced when the tab is refocused on a new day, so a tab left open overnight
// does not keep aggregating yesterday's "today" or a stale 30-day window.
// "Today" is LOCAL midnight rendered to an absolute instant, so the operator's
// wall-clock day is what the server aggregates, not UTC's.
function useWindows() {
  const [dayKey, setDayKey] = useState(localDayKey)
  useEffect(() => {
    const check = () => {
      if (document.visibilityState === "visible") {
        const next = localDayKey()
        setDayKey((prev) => (prev === next ? prev : next))
      }
    }
    document.addEventListener("visibilitychange", check)
    window.addEventListener("focus", check)
    return () => {
      document.removeEventListener("visibilitychange", check)
      window.removeEventListener("focus", check)
    }
  }, [])
  // dayKey drives the refresh: same day -> same bounds; new day -> re-derive.
  // Nothing inside reads it, so the rule sees it as surplus; it is the whole
  // point of the memo.
  // biome-ignore lint/correctness/useExhaustiveDependencies: dayKey is the cache key, not an input
  return useMemo(() => {
    const now = Date.now()
    const d = new Date(now)
    return {
      today: new Date(d.getFullYear(), d.getMonth(), d.getDate()).toISOString(),
      periodStart: new Date(now - PERIOD_DAYS * DAY_MS).toISOString(),
      prevStart: new Date(now - 2 * PERIOD_DAYS * DAY_MS).toISOString(),
    }
  }, [dayKey])
}

// The queries every overview variant is built on: the three summary windows
// behind the usage tiles, and the newest few requests. Every filter carries the
// shell's selected workspace, so the tiles, the sparklines and the
// recent-requests strip narrow together with whatever sits beside them. The
// usage hooks pick the surface for the caller (deployment-wide for an operator,
// organization-scoped otherwise; see useUsageScope), which is what lets the
// operator page and the organization page share this derivation.
function useUsageOverview() {
  const w = useWindows()
  const { selected: workspace } = useSelectedWorkspace()
  const scope = workspace?.workspace_id

  const todayFilters = useMemo(
    () => ({ workspace_id: scope, start_date: w.today }),
    [scope, w],
  )
  const periodFilters = useMemo(
    () => ({ workspace_id: scope, start_date: w.periodStart }),
    [scope, w],
  )
  // Bounded previous window ([-60d, -30d)) so it does not overlap the current one.
  const prevFilters = useMemo(
    () => ({
      workspace_id: scope,
      start_date: w.prevStart,
      end_date: w.periodStart,
    }),
    [scope, w],
  )
  const recentFilters = useMemo(() => ({ workspace_id: scope }), [scope])

  // Tiles and the sparkline read only `totals` and `series`, so all three windows
  // opt out of every breakdown rather than making the server run a GROUP BY per
  // dimension three times over for numbers this page never shows.
  const today = useUsageSummary(todayFilters, "hour", NO_BREAKDOWNS)
  const period = useUsageSummary(periodFilters, "day", NO_BREAKDOWNS)
  const previous = useUsageSummary(prevFilters, "day", NO_BREAKDOWNS)
  const recent = useUsageLogs(recentFilters, 0, 5)

  return { scope, today, period, previous, recent }
}

// The severity a cell states under its number, keyed off the derived Health.
// Paired with a dot as well as the color, so the judgment never rides on hue
// alone. Restored after the teardown briefly dropped them: the number alone
// says 3.8% and leaves the reader to decide whether that is fine, which is the
// one thing an at-a-glance page should not make them do.
const ERROR_WORDS = { ok: "Healthy", warn: "Elevated", alert: "High" } as const
const BUDGET_WORDS = {
  ok: "On track",
  warn: "Near limit",
  alert: "Over budget",
} as const

// The four usage tiles both overviews open with: spend today, spend and request
// volume over the window, and the window's error rate. A fragment rather than
// its own grid, so each page seats them beside its own tiles.
function UsageStatTiles({
  today,
  period,
  previous,
}: {
  today: ReturnType<typeof useUsageSummary>
  period: ReturnType<typeof useUsageSummary>
  previous: ReturnType<typeof useUsageSummary>
}) {
  const todayTotals = today.data?.totals
  const periodTotals = period.data?.totals
  const prevTotals = previous.data?.totals

  // The 30-day daily series is already on the wire (used for tile sparklines).
  // A single point has no trend to draw, so sparklines only appear with 2+ days.
  const periodSeries = period.data?.series ?? []
  const hasTrend = periodSeries.length > 1

  const err = errorRateHealth(periodTotals)
  const errPrev = errorRateHealth(prevTotals)
  // Each delta is its own const so the tile below can gate its chip on the
  // fraction rather than on the query: `deltaFraction` also returns null once
  // the current window has landed but the previous one has not, and when the
  // previous value is 0. TrendChip renders nothing for a null fraction, but the
  // *element* is truthy, and StatCard reserves the aside row for whatever it is
  // handed, so an ungated chip costs a tile 42px of dead space.
  const costDelta = periodTotals
    ? deltaFraction(periodTotals.cost, prevTotals?.cost)
    : null
  const requestDelta = periodTotals
    ? deltaFraction(periodTotals.request_count, prevTotals?.request_count)
    : null
  const errDelta =
    err.rate !== null && errPrev.rate !== null
      ? deltaFraction(err.rate, errPrev.rate)
      : null

  return (
    <>
      <StatCard
        label="Spend today"
        value={todayTotals ? formatUsd(todayTotals.cost) : "—"}
      />
      <StatCard
        label="Spend, last 30 days"
        value={periodTotals ? formatUsd(periodTotals.cost) : "—"}
        // Spend falling is the improvement, so a rise paints danger while the
        // arrow keeps telling the truth about which way it went.
        trend={
          costDelta !== null ? (
            <TrendChip
              fraction={costDelta}
              polarity="down-is-good"
              caption="vs prev"
            />
          ) : null
        }
        chart={
          hasTrend ? (
            <Sparkline
              values={periodSeries.map((p) => p.cost)}
              ariaLabel="Spend trend over the last 30 days"
            />
          ) : undefined
        }
      />
      <StatCard
        label="Requests, last 30 days"
        value={periodTotals ? formatNumber(periodTotals.request_count) : "—"}
        // Volume, so `neutral`: more traffic through the gateway is neither a
        // win nor a regression on its own, and the error rate tile beside it
        // is what carries the judgment.
        trend={
          requestDelta !== null ? (
            <TrendChip fraction={requestDelta} caption="vs prev" />
          ) : null
        }
        chart={
          hasTrend ? (
            <Sparkline
              values={periodSeries.map((p) => p.requests)}
              ariaLabel="Request volume trend over the last 30 days"
            />
          ) : undefined
        }
      />
      <StatCard
        label="Error rate, last 30 days"
        value={err.rate === null ? "—" : formatPct(err.rate)}
        status={toStatStatus(err.status)}
        statusLabel={
          err.status === "neutral" ? undefined : ERROR_WORDS[err.status]
        }
        // Errors falling is the improvement, as with spend.
        trend={
          errDelta !== null ? (
            <TrendChip
              fraction={errDelta}
              polarity="down-is-good"
              caption="vs prev"
            />
          ) : null
        }
      />
    </>
  )
}

/**
 * The landing route, which is the one page nobody navigates to on purpose.
 *
 * The operator page's status panels read deployment-wide endpoints (provider
 * health, budgets, keys, accounts), and since otari-ai#1880 those answer 403
 * to anyone who is not an operator of the deployment. The gate is here rather
 * than on the queries because the hooks do not all take an `enabled` flag, and
 * a component that is not rendered runs none of them: that is what keeps a
 * tenant's first page from being a row of failed requests.
 *
 * Decided from the organization context, which is where `useUsageScope` and the
 * nav rail take the same answer from, so nothing on screen can hold a different
 * one (otari-ai#1936). Two sources compose into a page that reports nothing
 * while being wrong: the deployment-wide panels beside usage tiles quietly
 * reading `/v1/organizations/me/usage`, every number disagreeing with its
 * neighbor and each query happy with its own. It also costs no request, since
 * the shell reads this context before it paints.
 *
 * So it fails toward the scoped page, the direction the hooks behind those tiles
 * already fail in: the narrower surface understates rather than refusing, and
 * reports its own error where this page can show it. An errored context is an
 * answer, not a wait, for `useUsageScope`'s reason: holding the page on a
 * loading state that no retry clears renders nothing at all.
 */
export function OverviewIndex() {
  const context = useOrganizationContext()

  // `isFetched`, not `isPending`: a query that errored with no data goes back to
  // pending on its next fetch, and the page below this line mounts a second
  // observer of the same context (the usage hooks), whose mount is what asks for
  // that fetch. Reading the transient state would swing the page back to this
  // spinner, unmount the observer, and start the round again forever.
  if (!context.isFetched) {
    return <PageLoading />
  }
  if (!isDeploymentOperator(context.data)) {
    return <OrganizationOverview />
  }
  return <OperatorOverviewIndex />
}

/**
 * The landing page for a signed-in caller who does not operate the deployment.
 *
 * The 11 Aug roles matrix has Overview as "my usage" for every role
 * (otari-ai#1946), so this is a real page rather than a card apologizing for
 * the operator one (otari-ai#1929): the same usage tiles and recent-request
 * preview the operator page opens with, served by the scope-aware usage hooks,
 * which read `/v1/organizations/me/usage` for this caller. The server narrows
 * those rows to what the caller may read (their organization for an admin, the
 * workspaces they belong to for a member; otari#837), so nothing here
 * re-derives roles. The deployment-wide panels (provider health, budgets, keys,
 * accounts) stay on the operator page, whose endpoints refuse this caller.
 */
function OrganizationOverview() {
  const { today, period, previous, recent } = useUsageOverview()

  // Recent activity is excluded for the operator page's reason: it renders its
  // own inline banner, so including it here would double-report. The previous
  // window is included: its only reader is the trend chips, so its failure
  // would otherwise just silently strip them.
  const loadError = today.error ?? period.error ?? previous.error

  const refresh = () => {
    void today.refetch()
    void period.refetch()
    void previous.refetch()
    void recent.refetch()
  }
  const isRefreshing =
    today.isFetching ||
    period.isFetching ||
    previous.isFetching ||
    recent.isFetching

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Overview"
        description="At-a-glance spend, traffic, and recent activity in your organization."
        action={
          <RefreshButton
            onRefresh={refresh}
            isFetching={isRefreshing}
            updatedAt={period.dataUpdatedAt}
          />
        }
      />

      <ErrorBanner error={loadError} />

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
        <UsageStatTiles today={today} period={period} previous={previous} />
      </div>

      <RecentActivity
        entries={recent.data ?? []}
        loading={recent.isLoading}
        error={recent.error}
      />
    </div>
  )
}

// The operator index uses the provider list to make the empty gateway a
// useful getting-started page. The providers query is operator-gated, so this
// only runs once the gate above has answered.
function OperatorOverviewIndex() {
  const providers = useProviders()
  if (providers.isLoading) {
    // A visible wait beats a blank screen while the master-key-gated providers
    // query resolves on first paint.
    return <PageLoading />
  }
  // A failed providers query leaves the setup state unknown, so it stays neutral
  // (no getting-started block) and the error is reported instead of swallowed.
  // Its refetch goes down with it: the query is cached for minutes and does not
  // refetch on focus, so without this the page's Refresh could not clear the
  // banner and a recovered backend would need a full reload to be noticed.
  return (
    <OverviewPage
      needsSetup={providers.isSuccess && providers.data.providers.length === 0}
      hasProviders={providers.isSuccess && providers.data.providers.length > 0}
      setupError={providers.error}
      refreshSetup={providers.refetch}
      setupFetching={providers.isFetching}
    />
  )
}

/**
 * The one page nobody navigates to on purpose, rebuilt as a divided surface.
 *
 * There are no cards here any more. The page is one ground partitioned by
 * hairlines: a header, an optional attention strip, a KPI strip of five cells
 * divided by vertical rules, a subordinate spend chart, and a lower split of
 * the activity table against a workspace rail. Each section is full-bleed
 * between `border-border` rules, which is what `-mx-4 md:-mx-6` is doing at
 * every one of them: `<main>` pads its column, and a rule that stops short of
 * the column edge reads as a card's top border rather than as a division of
 * the page.
 */
export function OverviewPage({
  needsSetup = false,
  hasProviders = false,
  setupError,
  refreshSetup,
  setupFetching = false,
}: {
  needsSetup?: boolean
  /**
   * Whether a provider is configured, which is the *other* answer the same
   * query gives: `needsSetup` is only true once it has succeeded and found
   * none, so its negation would also cover a failed query. The setup guide
   * needs the positive form, and takes it from here rather than asking again
   * (see `SetupGuideCard`).
   */
  hasProviders?: boolean
  setupError?: unknown
  refreshSetup?: () => void
  setupFetching?: boolean
}) {
  const usage = useUsageOverview()
  const { today, period, previous, recent } = usage
  const health = useProviderHealth()
  const budgets = useBudgets()
  // Same scope as the API keys page this tile links to, so the count and the
  // table behind it cannot disagree.
  const keys = useKeys(usage.scope)
  const users = useUsers()
  const members = useOrganizationMembers()

  const todayTotals = today.data?.totals
  const periodTotals = period.data?.totals
  const prevTotals = previous.data?.totals

  // The 30-day daily series is already on the wire, and so is today's hourly
  // one: the today window is requested at `"hour"` granularity for exactly this.
  // A single point has no trend to draw, so a sparkline needs two.
  const periodSeries = period.data?.series ?? []
  const todaySeries = today.data?.series ?? []
  const hasTrend = periodSeries.length > 1
  const hasHourlyTrend = todaySeries.length > 1

  // The status strip reads the window's error rate too; errorRateHealth is
  // pure, so it is derived here again rather than threaded out of the tiles.
  const err = errorRateHealth(periodTotals)
  const errPrev = errorRateHealth(prevTotals)
  // Each delta is its own const so the cell below can gate its chip on the
  // fraction rather than on the query: `deltaFraction` also returns null once
  // the current window has landed but the previous one has not, and when the
  // previous value is 0.
  const costDelta = periodTotals
    ? deltaFraction(periodTotals.cost, prevTotals?.cost)
    : null
  const requestDelta = periodTotals
    ? deltaFraction(periodTotals.request_count, prevTotals?.request_count)
    : null
  const errDelta =
    err.rate !== null && errPrev.rate !== null
      ? deltaFraction(err.rate, errPrev.rate)
      : null

  const budget = budgetHealth(budgets.data ?? [])
  const providerHealth = providerHealthStatus(health.data)

  const activeKeys = (keys.data ?? []).filter((k) => k.is_active).length
  const activeMembers = (members.data ?? []).filter(
    (member) => member.status === "active",
  ).length

  // The getting-started state is an onboarding empty state: the gateway has no
  // providers AND no recorded usage. Imported OTLP usage lands in the usage
  // tables (with counts_toward_budget=false) through a budget-exempt key and no
  // provider config, so "no providers" alone no longer means "nothing has
  // happened". `recent` is the unfiltered, all-time log query
  // `useUsageOverview` already loads; gate on it having resolved so the strip
  // never flashes in then hides.
  const hasAnyUsage = (recent.data?.length ?? 0) > 0
  const isEmpty = needsSetup && recent.isSuccess && !hasAnyUsage

  // Surface the first load error across the cell queries so a broken master key
  // or backend does not just leave a wall of "—". Recent activity is excluded: it
  // renders its own inline banner, so including it here would double-report. The
  // provider-list error from the index comes first: it is the query that decides
  // whether this page is a getting-started screen, so its failure is the most
  // load-bearing thing to tell the operator about.
  const loadError =
    setupError ??
    today.error ??
    period.error ??
    previous.error ??
    health.error ??
    budgets.error ??
    keys.error ??
    users.error ??
    members.error

  // Manual refresh for the whole page; the windows already advance across
  // midnight on focus, but the numbers within a day are only as fresh as the
  // last fetch, so give the operator a way to pull the latest.
  const refresh = () => {
    refreshSetup?.()
    void today.refetch()
    void period.refetch()
    void previous.refetch()
    void health.refetch()
    void budgets.refetch()
    void keys.refetch()
    void users.refetch()
    void members.refetch()
    void recent.refetch()
  }
  const isRefreshing =
    setupFetching ||
    today.isFetching ||
    period.isFetching ||
    previous.isFetching ||
    health.isFetching ||
    budgets.isFetching ||
    keys.isFetching ||
    users.isFetching ||
    members.isFetching ||
    recent.isFetching

  return (
    <div className="flex flex-col">
      <OverviewHeader
        refresh={refresh}
        isRefreshing={isRefreshing}
        updatedAt={period.dataUpdatedAt}
      />

      {isEmpty ? <GetStartedStrip /> : null}

      {/* The step after that one: a provider exists, so the guide can hand out a
          key and watch for the first request. It decides for itself whether to
          render, including holding back while there is no provider, which is
          when the strip above is the right guide instead. */}
      <SetupGuideCard hasProviders={hasProviders} />

      <ErrorBanner error={loadError} />

      <AttentionStrip
        providerHealth={providerHealth}
        healthy={health.data?.healthy ?? 0}
        degraded={health.data?.degraded ?? 0}
        total={health.data?.total ?? 0}
        budget={budget}
        errStatus={err.status}
        errRate={err.rate}
        // The strip evaluates health, budgets, and error rate only after all
        // three load successfully, avoiding transient or false alerts.
        ready={health.isSuccess && budgets.isSuccess && period.isSuccess}
        failed={health.isError || budgets.isError || period.isError}
      />

      <KpiStrip empty={isEmpty}>
        <KpiCell
          label="Spend today"
          // A real zero where zero is a fact, an em dash where the value is
          // unknown: a failed query must not read as "you spent nothing".
          value={todayTotals ? formatUsd(todayTotals.cost) : "—"}
          // "midnight" and not "00:00 UTC": `useWindows` builds this window from
          // the operator's *local* midnight, deliberately and with a comment
          // saying so, so a UTC string would be false for everyone not on it.
          subline={
            isEmpty
              ? "no prior spend"
              : todayTotals
                ? "since midnight"
                : "no data"
          }
          graphic={
            !isEmpty && hasHourlyTrend ? (
              <Sparkline
                values={todaySeries.map((p) => p.cost)}
                ariaLabel="Spend by hour today"
                height={40}
              />
            ) : undefined
          }
        />
        <KpiCell
          label="Spend, last 30 days"
          value={periodTotals ? formatUsd(periodTotals.cost) : "—"}
          subline={
            isEmpty ? "no prior spend" : periodTotals ? undefined : "no data"
          }
          // Spend falling is the improvement, so a rise paints danger while the
          // arrow keeps telling the truth about which way it went.
          delta={
            costDelta !== null ? (
              <TrendChip
                fraction={costDelta}
                polarity="down-is-good"
                caption="vs prev"
              />
            ) : undefined
          }
          graphic={
            !isEmpty && hasTrend ? (
              <Sparkline
                values={periodSeries.map((p) => p.cost)}
                ariaLabel="Spend trend over the last 30 days"
                height={40}
              />
            ) : undefined
          }
        />
        <KpiCell
          label="Requests, last 30 days"
          value={periodTotals ? formatNumber(periodTotals.request_count) : "—"}
          subline={
            isEmpty ? "no prior traffic" : periodTotals ? undefined : "no data"
          }
          // Volume, so no polarity: more traffic through the gateway is neither
          // a win nor a regression on its own, and the error rate beside it is
          // what carries the judgment.
          delta={
            requestDelta !== null ? (
              <TrendChip fraction={requestDelta} caption="vs prev" />
            ) : undefined
          }
          graphic={
            !isEmpty && hasTrend ? (
              <Sparkline
                values={periodSeries.map((p) => p.requests)}
                ariaLabel="Request volume trend over the last 30 days"
                height={40}
              />
            ) : undefined
          }
        />
        <KpiCell
          label="Error rate, last 30 days"
          value={err.rate === null ? "—" : formatPct(err.rate)}
          // The word is present whenever there is a rate to judge, "Healthy"
          // included: a line that appears only when something is wrong makes
          // its absence ambiguous with a page that has not loaded.
          severity={
            err.status === "neutral"
              ? undefined
              : { status: err.status, word: ERROR_WORDS[err.status] }
          }
          subline={
            err.rate === null
              ? isEmpty
                ? "NO REQUESTS YET"
                : "no requests in range"
              : undefined
          }
          // Errors falling is the improvement, as with spend.
          delta={
            errDelta !== null ? (
              <TrendChip
                fraction={errDelta}
                polarity="down-is-good"
                caption="vs prev"
              />
            ) : undefined
          }
          graphic={
            !isEmpty && periodTotals ? (
              <span className="text-xs text-muted">
                {`${formatNumber(periodTotals.error_count)} of ${formatNumber(periodTotals.request_count)} requests`}
              </span>
            ) : undefined
          }
        />
        <KpiCell
          label="Budget health"
          value={
            budgets.data && budget.worst ? formatPct(budget.worst.pct) : "—"
          }
          severity={
            budgets.data && budget.worst && budget.status !== "neutral"
              ? { status: budget.status, word: BUDGET_WORDS[budget.status] }
              : undefined
          }
          // The defect this rule was written for: without it, a deployment with
          // no budgets showed a bare em dash on the populated page and only
          // explained itself in the empty state.
          subline={
            budgets.data && budget.worst
              ? undefined
              : isEmpty
                ? "NO BUDGETS SET"
                : budgets.data
                  ? "no budgets set"
                  : "no data"
          }
          // `SpendMeter`, not the plain accent `Meter` it used to be, and this
          // was the gap: the strip named the state in words and drew it in one
          // colour, so a budget 37% past its limit and one comfortably inside
          // it produced the same teal bar. Same component as the Budgets
          // table's cell now, so the two cannot say different things about the
          // same budget.
          graphic={
            !isEmpty && budgets.data && budget.worst ? (
              <SpendMeter
                spent={budget.worst.spent}
                allocated={budget.worst.allocated}
                ariaLabel={`Worst budget usage: ${budget.worst.name}`}
              />
            ) : undefined
          }
        />
      </KpiStrip>

      {/* Subordinate to the strip above it, which is the point: the numbers are
          the answer and the shape of the month is the context. Absent entirely
          in the empty state, where there is no shape to show. */}
      {isEmpty ? null : (
        <SpendChart series={periodSeries} ready={period.isSuccess} />
      )}

      <div className="flex flex-col lg:flex-row lg:items-stretch">
        <div className="min-w-0 flex-1">
          <RecentActivity
            entries={recent.data ?? []}
            loading={recent.isLoading}
            error={recent.error}
          />
        </div>
        <div className="border-border lg:w-[300px] lg:shrink-0 lg:border-l">
          <WorkspaceRail
            activeKeys={keys.data ? activeKeys : null}
            activeMembers={members.data ? activeMembers : null}
          />
        </div>
      </div>
    </div>
  )
}

function OverviewHeader({
  refresh,
  isRefreshing,
  updatedAt,
}: {
  refresh: () => void
  isRefreshing: boolean
  updatedAt: number
}) {
  return (
    <header className="flex flex-col gap-4 pb-6 sm:flex-row sm:items-start sm:justify-between">
      <div>
        <h1 className="text-display">Overview</h1>
        {/* ~620px rather than `max-w-prose`: this paragraph sits beside the meta
            block, so its measure is set by the room the two share. */}
        <p className="mt-1 max-w-[620px] text-sm text-muted">
          At-a-glance spend, traffic, and health across the gateway.
        </p>
      </div>
      <div className="flex shrink-0 items-center gap-3">
        <span className="text-overline">Last 30 days</span>
        <RefreshButton
          onRefresh={refresh}
          isFetching={isRefreshing}
          updatedAt={updatedAt}
        />
      </div>
    </header>
  )
}

// Where "add a provider credential" lives on this deployment. A standalone one
// serves the process-wide page; a hosted one serves the organization-scoped
// page in its place and does not report `providers` at all, so naming
// `/providers` unconditionally would point an operator at the shell's "not
// available here" panel.
//
// Only correct for *adding* one, which is why provider health does not use it:
// `/v1/providers/health` reports on `config.providers`, the process-global
// table, so on a hosted deployment an unreachable instance is not a row the
// organization page could show. `AttentionStrip` drops the link there rather
// than sending somebody to a page the instance is not on.
function useAddProviderRoute(): "/providers" | "/organization/provider-keys" {
  const serves = useSurfaces()
  return serves("providers") ? "/providers" : "/organization/provider-keys"
}

function GetStartedStrip() {
  const navigate = useNavigate()
  const addProviderRoute = useAddProviderRoute()

  return (
    <Section
      // Its top rule only, for the reason `AttentionStrip` gives below: whatever
      // follows this band in the empty state (the attention strip, or the KPI
      // strip under it) draws the seam from its own side.
      className="border-t border-border py-5"
      contentClassName="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
    >
      <div className="flex items-start gap-3">
        <Dot className="mt-2.5 bg-accent" />
        <div>
          <h2 className="text-display-sub">Get started with Otari</h2>
          <p className="mt-1 text-sm text-muted">
            Add a provider to begin serving models. Once it is configured, this
            page will show your gateway&rsquo;s traffic, spend, and health.
          </p>
        </div>
      </div>
      <Button
        variant="primary"
        className="shrink-0"
        onPress={() => navigate({ to: addProviderRoute })}
      >
        Add your first provider
      </Button>
    </Section>
  )
}

// A neutral, hue-free line for a failed status source. Its details are also
// surfaced in the ErrorBanner, but this preserves context at the status area.
function NeutralStrip({ text }: { text: string }) {
  return (
    <Section
      role="status"
      className="border-t border-border py-3"
      contentClassName="text-sm text-muted"
    >
      {text}
    </Section>
  )
}

function AttentionStrip({
  providerHealth,
  healthy,
  degraded,
  total,
  budget,
  errStatus,
  errRate,
  ready,
  failed,
}: {
  providerHealth: "ok" | "warn" | "alert" | "neutral"
  healthy: number
  degraded: number
  total: number
  budget: ReturnType<typeof budgetHealth>
  errStatus: "ok" | "warn" | "alert" | "neutral"
  errRate: number | null
  ready: boolean
  failed: boolean
}) {
  // Provider health is about `config.providers`, which a hosted deployment
  // serves no page for, so those two entries state the problem without offering
  // a destination there. Naming one would be worse than naming none: both
  // candidates are wrong, `/providers` because the shell answers it with "not
  // available here" and the organization page because the instance is not on it.
  const providerProblemsAreReachable = useSurfaces()("providers")
  // A failed source deserves a visible status message; while loading, wait for
  // actionable information instead of reserving space for a transient banner.
  if (failed) {
    return <NeutralStrip text="Some status data could not be loaded." />
  }
  if (!ready) {
    return null
  }

  const problems: {
    text: string
    // Absent where this deployment hosts no page for the problem; the entry
    // renders as plain text rather than as a link nobody can follow.
    to?: LinkProps["to"]
    search?: LinkProps["search"]
  }[] = []
  if ((providerHealth === "warn" || providerHealth === "alert") && total > 0) {
    // A degraded provider answers no model listing but is not down, so it is
    // reported separately rather than folded into "unreachable" (issue #447).
    const down = total - healthy - degraded
    if (down > 0) {
      problems.push({
        text: `${down} provider${down === 1 ? "" : "s"} unreachable`,
        to: providerProblemsAreReachable ? "/providers" : undefined,
      })
    }
    if (degraded > 0) {
      problems.push({
        text: `${degraded} provider${degraded === 1 ? "" : "s"} without model discovery`,
        to: providerProblemsAreReachable ? "/providers" : undefined,
      })
    }
  }
  if (budget.overCount > 0) {
    problems.push({
      text: `${budget.overCount} budget${budget.overCount === 1 ? "" : "s"} over limit`,
      to: "/budgets",
    })
  } else if (budget.nearCount > 0) {
    problems.push({
      text: `${budget.nearCount} budget${budget.nearCount === 1 ? "" : "s"} near limit`,
      to: "/budgets",
    })
  }
  if (errStatus === "alert" && errRate !== null) {
    problems.push({
      text: `error rate ${formatPct(errRate)}`,
      to: "/activity",
      search: { status: "error" },
    })
  }

  if (problems.length === 0) {
    return null
  }

  // No fill and no radius any more: the strip is a band of the page under a
  // rule, and the square danger dot is what carries the urgency the tinted
  // attention fill used to.
  //
  // Its top rule only. The band below it declares its own top rule, and two
  // rules meeting on one line paint 1px twice: measured on this page, that seam
  // came out two rows of pixels where every other rule was one, which is what
  // read as a heavier rule. The seam belongs to the band below, which always
  // renders; this band does not always.
  return (
    <Section
      role="alert"
      className="border-t border-border py-3"
      contentClassName="flex flex-col gap-2 text-sm sm:flex-row sm:flex-wrap sm:items-center"
    >
      <span className="flex items-center gap-2 font-semibold text-foreground">
        <Dot className="bg-danger" />
        Needs attention
      </span>
      {problems.map((p, i) => (
        <span key={p.text} className="flex items-center gap-2 text-muted">
          {i > 0 ? (
            <span aria-hidden className="text-subtle">
              ·
            </span>
          ) : null}
          {p.to ? (
            <Link
              to={p.to}
              search={p.search}
              // Thicken the underline on hover rather than lightening the text:
              // the color here is already the one tuned to clear AA.
              className="underline underline-offset-2 hover:decoration-2"
            >
              {p.text}
            </Link>
          ) : (
            <span>{p.text}</span>
          )}
        </span>
      ))}
    </Section>
  )
}

/**
 * The month's shape, under the numbers that answer the question. Bars rather
 * than an area, square-topped, on a baseline rule.
 *
 * Subordinate to the strip above it, and subordinate means a shorter plot and
 * fewer x labels. It does **not** mean fewer axes: a bar chart with no y-axis
 * cannot be read quantitatively, and a chart that cannot be read is decoration
 * rather than a supporting detail. The y-axis was missing here and this is the
 * fix; five steps in a 44px lane, nine x labels at every fourth day.
 */
function SpendChart({
  series,
  ready,
}: {
  series: { bucket_start: string; cost: number }[]
  ready: boolean
}) {
  const [hovered, setHovered] = useState<number | null>(null)
  if (!ready || series.length < 2) {
    return null
  }
  const peak = Math.max(...series.map((p) => p.cost), 0)
  const hoveredPoint = hovered === null ? null : (series[hovered] ?? null)
  // A rounded ceiling rather than the peak itself, so the top label is a number
  // somebody would say out loud and the steps between are even.
  const top = niceCeiling(peak)
  // Top-down, which is the order they are drawn in.
  const steps = [1, 0.75, 0.5, 0.25, 0].map((f) => top * f)
  return (
    <Section className="border-b border-border py-5" contentClassName="">
      <div className="flex items-baseline justify-between">
        <h2 className="text-title">Spend, last 30 days</h2>
        <Link
          to="/usage"
          className="text-sm text-muted underline underline-offset-2 hover:text-foreground"
        >
          View usage →
        </Link>
      </div>
      <div className="mt-4 flex">
        {/* The lane is fixed so the plot's left edge does not move as the
            numbers change width. */}
        <div
          aria-hidden
          className="flex h-[180px] w-11 shrink-0 flex-col justify-between pr-2 text-right text-mono-micro text-subtle"
        >
          {steps.map((value) => (
            <span key={value} className="leading-none">
              {formatAxisUsd(value)}
            </span>
          ))}
        </div>
        <div className="min-w-0 flex-1">
          <div
            role="img"
            aria-label={`Daily spend over the last 30 days, peaking at ${formatUsd(peak)}`}
            className="relative flex h-[180px] items-end gap-[3px] border-b border-border"
            onPointerLeave={() => setHovered(null)}
          >
            {series.map((point, i) => (
              // The hit area is the whole column, not the drawn bar: a day with
              // almost no spend is two pixels tall, and a tooltip you can only
              // reach by hitting two pixels is one nobody reaches.
              <button
                type="button"
                key={point.bucket_start}
                aria-label={`${shortDate(point.bucket_start)}: ${formatUsd(point.cost)}`}
                className="group flex min-w-px flex-1 items-end self-stretch"
                onPointerEnter={() => setHovered(i)}
                onFocus={() => setHovered(i)}
                onBlur={() => setHovered(null)}
              >
                <span
                  className="w-full bg-accent group-hover:bg-accent-hover"
                  style={{
                    height:
                      top > 0
                        ? `${Math.max(1, (point.cost / top) * 100)}%`
                        : "1px",
                  }}
                />
              </button>
            ))}
            {hoveredPoint ? (
              <ChartHoverCard
                date={shortDate(hoveredPoint.bucket_start)}
                value={formatUsd(hoveredPoint.cost)}
                atPercent={((hovered as number) + 0.5) / series.length}
                barPercent={top > 0 ? hoveredPoint.cost / top : 0}
              />
            ) : null}
          </div>
          {/* Every fourth day, which lands nine labels across a thirty-day
              window: enough to date a bar, few enough not to become a second
              row of text under a chart that is not the headline. */}
          <div className="mt-1.5 flex text-mono-micro text-subtle">
            {series.map((point, i) => (
              <span
                key={point.bucket_start}
                className="min-w-px flex-1 text-center whitespace-nowrap"
              >
                {i % 4 === 0 ? shortDate(point.bucket_start) : "\u00a0"}
              </span>
            ))}
          </div>
        </div>
      </div>
      <p className="mt-2 text-xs text-muted">
        Daily totals for the selected workspace. Unpriced requests are recorded
        at zero.
      </p>
    </Section>
  )
}

/**
 * The chart's hover card: the same small divided surface the Usage chart's
 * tooltip is, on the floating rule (surface fill, control edge, square, no
 * shadow). A mono date over a rule, then the value.
 *
 * Anchored to the top of the bar under the pointer, horizontally and
 * vertically. It used to sit at `bottom-full`, above the whole plot, which put
 * it a full plot-height away from a bar near the baseline: the card said one
 * thing and the pointer was somewhere else entirely. `bottom` is now the bar's
 * own height as a share of the plot, so the card rides up and down with what it
 * is describing, which is what the Usage chart's tooltip does by following the
 * pointer.
 *
 * Clamped at both ends. Horizontally by `translate`, so a card at either edge
 * stays inside the plot; vertically at 72%, so a bar near the top does not push
 * the card out through the plot's ceiling.
 */
function ChartHoverCard({
  date,
  value,
  atPercent,
  barPercent,
}: {
  date: string
  value: string
  atPercent: number
  /** The hovered bar's height as a share of the plot, 0 to 1. */
  barPercent: number
}) {
  return (
    <span
      aria-hidden
      className="pointer-events-none absolute z-10 mb-2 min-w-[7.5rem] border border-control-border bg-surface text-xs"
      style={{
        left: `${atPercent * 100}%`,
        bottom: `${Math.min(72, Math.max(0, barPercent * 100))}%`,
        transform: `translateX(-${Math.min(90, Math.max(10, atPercent * 100))}%)`,
      }}
    >
      <span className="block border-b border-border px-2.5 py-1.5 text-mono-micro text-subtle">
        {date}
      </span>
      <span className="flex items-center gap-2 px-2.5 py-1.5">
        <Dot className="bg-accent" />
        <span className="text-muted">Spend</span>
        <span className="ml-auto text-mono-caption text-foreground tabular-nums">
          {value}
        </span>
      </span>
    </span>
  )
}

/**
 * The next 1, 2 or 5 times a power of ten at or above a value, so an axis tops
 * out at a number a reader would say rather than at the tallest bar.
 */
function niceCeiling(value: number): number {
  if (!(value > 0)) return 1
  const magnitude = 10 ** Math.floor(Math.log10(value))
  for (const step of [1, 2, 5, 10]) {
    if (value <= step * magnitude) return step * magnitude
  }
  return 10 * magnitude
}

/** Axis money: whole dollars once the scale is past them, cents below. */
function formatAxisUsd(value: number): string {
  return value >= 10 || value === 0
    ? `$${Math.round(value)}`
    : `$${value.toFixed(2)}`
}

function shortDate(bucket: string | undefined): string {
  if (!bucket) return ""
  const d = new Date(bucket)
  return Number.isNaN(d.getTime())
    ? ""
    : d.toLocaleDateString(undefined, { month: "short", day: "numeric" })
}

// "absorbed" is kept verbatim rather than folded into "ok": it is an attempt a
// routing policy recovered from, so calling it "ok" would present a failed attempt
// as a served request, and this preview is unfiltered so those rows do appear here.
// It is still not an error, which is why it does not read as one.
function statusWord(status: string): string {
  if (status === "error") return "error"
  return status === "absorbed" ? "absorbed" : "ok"
}

/**
 * Status as a family on a square dot plus a severity in text, never as a fill.
 * The dot says which family (served, failed, recovered) and the word says what
 * happened, so neither hue alone nor shape alone has to carry it.
 */
function StatusMark({ status }: { status: string }) {
  const word = statusWord(status)
  const dot =
    word === "error"
      ? "bg-danger"
      : word === "absorbed"
        ? "bg-text-subtle"
        : "bg-success"
  const text = word === "error" ? "text-danger" : "text-muted"
  return (
    <span className={`flex items-center gap-2 text-mono-caption ${text}`}>
      <Dot className={dot} />
      {word.toUpperCase()}
    </span>
  )
}

// Newest few requests, as an at-a-glance preview. Rows are read-only; a single
// "View all" link opens the full Activity log. Cost and tokens are nullable per
// row, and a failed request has neither, which reads as an em dash rather than
// as a zero it did not spend.
function RecentActivity({
  entries,
  loading,
  error,
}: {
  entries: UsageEntry[]
  loading: boolean
  error: unknown
}) {
  const columns: DataTableColumn<UsageEntry>[] = [
    {
      id: "time",
      header: "Time",
      cell: (entry) => (
        <span
          className="text-mono-caption text-muted"
          title={new Date(entry.timestamp).toLocaleString()}
        >
          {formatRelative(entry.timestamp)}
        </span>
      ),
    },
    {
      id: "model",
      header: "Model",
      isRowHeader: true,
      cell: (entry) => (
        <span className="text-mono-caption text-foreground">{entry.model}</span>
      ),
    },
    {
      id: "key",
      header: "Key",
      cell: (entry) => (
        <span className="text-mono-caption text-muted">
          {entry.api_key_name ?? "—"}
        </span>
      ),
    },
    {
      id: "tokens",
      header: "Tokens",
      align: "end",
      cell: (entry) => (
        <span className="text-mono-caption tabular-nums">
          {entry.total_tokens === null ? "—" : formatNumber(entry.total_tokens)}
        </span>
      ),
    },
    {
      id: "cost",
      header: "Cost",
      align: "end",
      cell: (entry) => (
        <span className="text-mono-caption tabular-nums">
          {entry.cost === null ? "—" : formatUsd(entry.cost)}
        </span>
      ),
    },
    {
      id: "status",
      header: "Status",
      cell: (entry) => <StatusMark status={entry.status} />,
    },
  ]

  return (
    <section className="flex flex-col pt-6 lg:pr-6">
      <div className="flex items-baseline justify-between pb-3">
        <h2 className="text-title">Recent activity</h2>
        <Link
          to="/activity"
          className="text-sm text-muted underline underline-offset-2 hover:text-foreground"
        >
          View all →
        </Link>
      </div>
      <ErrorBanner error={error} />
      {/* The lane widths and the row pitch are in globals.css, keyed on the
          column ids below: they have to reach `.table__column`, which is
          HeroUI's DOM and not addressable from here. */}
      <div className="otari-overview-activity">
        <DataTable
          ariaLabel="Recent activity"
          columns={columns}
          rows={entries}
          getRowKey={(entry) => entry.id}
          isLoading={loading}
          emptyContent="No requests yet. Once the gateway serves traffic, it appears here."
        />
      </div>
    </section>
  )
}

/**
 * The workspace's own numbers, beside the deployment-wide table rather than in
 * it. A rail and not two more KPI cells: these count things that belong to one
 * workspace, and the strip above counts what the gateway did.
 */
function WorkspaceRail({
  activeKeys,
  activeMembers,
}: {
  activeKeys: number | null
  activeMembers: number | null
}) {
  return (
    <section className="flex flex-col gap-6 pt-6 lg:pl-6">
      <div>
        <h2 className="text-title">This workspace</h2>
        <dl className="mt-3 flex gap-8 lg:flex-col lg:gap-4">
          <RailStat
            label="Active keys"
            value={activeKeys === null ? "—" : formatNumber(activeKeys)}
          />
          <RailStat
            label="Active members"
            value={activeMembers === null ? "—" : formatNumber(activeMembers)}
          />
        </dl>
      </div>
      <div>
        <h2 className="text-overline">Go to</h2>
        <ul className="mt-1 flex flex-col">
          <RailLink to="/keys">API keys</RailLink>
          <RailLink to="/organization/members">Members</RailLink>
          <RailLink to="/budgets">Budgets</RailLink>
          <RailLink to="/activity">Activity</RailLink>
        </ul>
      </div>
    </section>
  )
}

function RailStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-overline">{label}</dt>
      <dd className="text-mono-figure font-normal text-foreground">{value}</dd>
    </div>
  )
}

function RailLink({
  to,
  children,
}: {
  to: LinkProps["to"]
  children: ReactNode
}) {
  return (
    <li className="border-b border-separator-secondary last:border-b-0">
      <Link
        to={to}
        className="flex items-center justify-between py-2.5 text-sm text-muted hover:text-foreground"
      >
        {children}
        <span aria-hidden>→</span>
      </Link>
    </li>
  )
}
