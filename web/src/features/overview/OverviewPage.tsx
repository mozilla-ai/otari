import { Button, Card } from "@heroui/react"
import type { LinkProps } from "@tanstack/react-router"
import { Link, useNavigate } from "@tanstack/react-router"
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
import { TrendChip } from "@/shared/components/TrendChip"
import {
  ErrorBanner,
  PageHeader,
  PageLoading,
  RefreshButton,
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

// A status tile's short word (paired with the color so status never rides on hue
// alone), keyed off the derived Health.
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

  // The status strip reads the window's error rate too; errorRateHealth is
  // pure, so it is derived here again rather than threaded out of the tiles.
  const err = errorRateHealth(period.data?.totals)

  const budget = budgetHealth(budgets.data ?? [])
  const providerHealth = providerHealthStatus(health.data)

  const activeKeys = (keys.data ?? []).filter((k) => k.is_active).length
  const activeMembers = (members.data ?? []).filter(
    (member) => member.status === "active",
  ).length

  // The getting-started banner is an onboarding empty state: show it only when the
  // gateway has no providers AND no recorded usage. Imported OTLP usage lands in
  // the usage tables (with counts_toward_budget=false) through a budget-exempt key
  // and no provider config, so "no providers" alone no longer means "nothing has
  // happened". `recent` is the all-time log query useUsageOverview already loads;
  // gate on it having resolved so the banner never flashes in then hides.
  const hasAnyUsage = (recent.data?.length ?? 0) > 0
  const showGettingStarted = needsSetup && recent.isSuccess && !hasAnyUsage

  // Surface the first load error across the tile queries so a broken master key
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
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Overview"
        description="At-a-glance spend, traffic, and health across the gateway."
        action={
          <RefreshButton
            onRefresh={refresh}
            isFetching={isRefreshing}
            updatedAt={period.dataUpdatedAt}
          />
        }
      />

      {showGettingStarted ? <GettingStartedPanel /> : null}
      {/* The step after that one: a provider exists, so the guide can hand out a
          key and watch for the first request. It decides for itself whether to
          render, including holding back while there is no provider, which is
          when the panel above is the right guide instead. */}
      <SetupGuideCard hasProviders={hasProviders} />

      <ErrorBanner error={loadError} />

      <SystemStatusStrip
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

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
        {/* Tiles gate on data presence, not isLoading, so a failed query reads as
            "—" (unknown) rather than a misleading real zero. */}
        <UsageStatTiles today={today} period={period} previous={previous} />
        <StatCard
          label="Budget health"
          value={
            budgets.data
              ? budget.worst
                ? formatPct(budget.worst.pct)
                : "—"
              : "—"
          }
          status={budgets.data ? toStatStatus(budget.status) : undefined}
          statusLabel={
            budgets.data && budget.status !== "neutral"
              ? BUDGET_WORDS[budget.status]
              : undefined
          }
          hint={
            budgets.data
              ? budget.worst
                ? `${budget.label} · worst: ${budget.worst.name}`
                : budget.label
              : undefined
          }
          to="/budgets"
        />
        <StatCard
          label="Active keys"
          value={keys.data ? formatNumber(activeKeys) : "—"}
          to="/keys"
        />
        <StatCard
          label="Active members"
          value={members.data ? formatNumber(activeMembers) : "—"}
          to="/organization/members"
        />
      </div>

      <RecentActivity
        entries={recent.data ?? []}
        loading={recent.isLoading}
        error={recent.error}
      />
    </div>
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
// organization page could show. `SystemStatusStrip` drops the link there rather
// than sending somebody to a page the instance is not on.
function useAddProviderRoute(): "/providers" | "/organization/provider-keys" {
  const serves = useSurfaces()
  return serves("providers") ? "/providers" : "/organization/provider-keys"
}

function GettingStartedPanel() {
  const navigate = useNavigate()
  const addProviderRoute = useAddProviderRoute()

  return (
    <Card>
      <Card.Content className="flex flex-col gap-3 p-6">
        <div>
          <h2 className="text-heading">Get started with Otari</h2>
          <p className="mt-1 text-sm text-muted">
            Add a provider to begin serving models. Once it is configured, this
            page will show your gateway&rsquo;s traffic, spend, and health.
          </p>
        </div>
        <div>
          <Button
            variant="primary"
            onPress={() => navigate({ to: addProviderRoute })}
          >
            Add your first provider
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

// A neutral, hue-free strip for a failed status source. Its details are also
// surfaced in the ErrorBanner, but this preserves context at the status area.
function NeutralStrip({ text }: { text: string }) {
  return (
    <div
      role="status"
      className="flex items-center gap-2 rounded-xl border border-border bg-surface-alt px-4 py-3 text-sm text-muted"
    >
      {text}
    </div>
  )
}

function SystemStatusStrip({
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

  // The attention family rather than warning: every entry here names something
  // to go and do, and "look here" is what separates attention from caution in
  // the foundation's two warm roles.
  return (
    <div
      role="alert"
      className="flex flex-col gap-2 rounded-xl border border-attention-border bg-attention-subtle px-4 py-3 text-sm text-attention sm:flex-row sm:flex-wrap sm:items-center"
    >
      <span className="font-medium">Needs attention:</span>
      {problems.map((p, i) => (
        <span key={p.text} className="flex items-center gap-2">
          {i > 0 ? (
            <span aria-hidden className="opacity-60">
              ·
            </span>
          ) : null}
          {p.to ? (
            <Link
              to={p.to}
              search={p.search}
              // Thicken the underline on hover rather than lightening the text:
              // the color here is already the one tuned to clear AA on this fill.
              className="underline underline-offset-2 hover:decoration-2"
            >
              {p.text}
            </Link>
          ) : (
            <span>{p.text}</span>
          )}
        </span>
      ))}
    </div>
  )
}

// "absorbed" is kept verbatim rather than folded into "ok": it is an attempt a
// routing policy recovered from, so calling it "ok" would present a failed attempt
// as a served request, and this preview is unfiltered so those rows do appear here.
// It is still not an error, which is why it does not read as one.
function statusWord(status: string): string {
  if (status === "error") return "error"
  return status === "absorbed" ? "absorbed" : "ok"
}

// Newest few requests, as an at-a-glance preview. Rows are read-only; a single
// "View all" link opens the full Activity log. Cost is nullable per row.
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
          className="text-muted"
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
      cell: (entry) => <span className="text-foreground">{entry.model}</span>,
    },
    {
      id: "cost",
      header: "Cost",
      align: "end",
      cell: (entry) => (entry.cost === null ? "—" : formatUsd(entry.cost)),
    },
    {
      id: "status",
      header: "Status",
      cell: (entry) => (
        <span
          className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${
            entry.status === "error"
              ? "border-danger bg-danger-subtle text-danger"
              : entry.status === "absorbed"
                ? "border-warning bg-warning-subtle text-warning"
                : "border-border bg-primary-subtle text-primary-subtle-foreground"
          }`}
        >
          {statusWord(entry.status)}
        </span>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h2 className="text-title">Recent activity</h2>
        <Link
          to="/activity"
          className="text-sm text-link hover:text-link-hover hover:underline"
        >
          View all →
        </Link>
      </div>
      <ErrorBanner error={error} />
      <DataTable
        ariaLabel="Recent activity"
        columns={columns}
        rows={entries}
        getRowKey={(entry) => entry.id}
        isLoading={loading}
        emptyContent="No requests yet. Once the gateway serves traffic, it appears here."
      />
    </div>
  )
}
