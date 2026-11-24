import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { ChartLegend, TrendChart } from "@/design-system/metrics/charts"
import { KpiCell } from "@/design-system/metrics/KpiCell"
import { KpiStrip } from "@/design-system/metrics/KpiStrip"
import { Segmented } from "@/design-system/navigation/Segmented"
import { useCurrentOwner } from "@/features/organization/attribution"
import {
  AgentTypeBreakdown,
  SwapBreakdown,
} from "@/features/routerSavings/SavingsBreakdowns"
import {
  chartPoints,
  coverageLine,
  formatSaved,
  pricedCount,
  SAVED_KEY,
  SPENT_KEY,
  spentOn,
} from "@/features/routerSavings/savingsModel"
import {
  DEFAULT_PERIOD,
  figureLabel,
  periodFromUrl,
  usageRange,
  windowOf,
  windowWords,
} from "@/features/spend/period"
import { SpendPeriodPicker } from "@/features/spend/SpendPeriodPicker"
import { usePageClock } from "@/features/spend/usePageClock"
import { useSubagentSavings } from "@/shared/api/subagentSavings"
import {
  formatBucketTick,
  formatNumber,
  formatSpend,
} from "@/shared/helpers/format"
import { useUrlState } from "@/shared/helpers/urlState"
import {
  useNarrowingWorkspace,
  useSelectedWorkspace,
} from "@/shared/hooks/SelectedWorkspace"

const WHO = [
  { value: "", label: "Everyone" },
  { value: "me", label: "Just me" },
]

const URL_DEFAULTS = { period: DEFAULT_PERIOD, who: "" }

const WORKSPACE_UNREAD = new Error(
  "Your workspaces could not be read, so the savings cannot be shown. Reload the page to try again.",
)
const OWNER_UNREAD = new Error(
  "Your own usage could not be read, so your subagents cannot be told apart. Reload the page to try again.",
)

const CHART_SERIES = [
  { key: SPENT_KEY, label: "Spent", color: "var(--color-chart-cat-1)" },
  { key: SAVED_KEY, label: "Saved", color: "var(--color-chart-cat-2)" },
]

/**
 * What the subagent router saved: the tokens routed subagents consumed, priced
 * at the models they ran on and at the models they would have started on, less
 * what the recommendations cost. The server decides whose decisions the caller
 * reads, within the selected workspace; "Just me" narrows them to the caller's
 * own. A view of Sessions, so the title above it is the layout's.
 */
export function RouterSavingsPage() {
  // One clock for the page's life, so its window and query key hold still.
  const [now, refreshClock] = usePageClock()
  const { get, patch } = useUrlState(URL_DEFAULTS)
  const owner = useCurrentOwner()
  const workspaceId = useNarrowingWorkspace()
  const { isUnknown: isWorkspaceUnknown } = useSelectedWorkspace()

  const period = periodFromUrl(get("period"))
  const window = windowOf(period, now)
  const isMine = get("who") === "me"
  const savings = useSubagentSavings(
    {
      ...usageRange(window),
      bucket: window.bucket,
      workspace_id: workspaceId || undefined,
      user_id: isMine ? owner.ownerId || undefined : undefined,
    },
    // A read waits for the selected workspace, and "Just me" for the caller's
    // id: without either it is wider than the page says.
    workspaceId !== null && (!isMine || owner.ownerId !== ""),
  )

  // A view of Sessions, under its title and tab row: what the view shows and
  // its two controls, with no heading of its own.
  const intro = (
    <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
      <p className="max-w-[38.75rem] text-sm text-muted">
        What routed subagents cost against what they would have cost on the
        model they would have started on, less what the recommendations cost.
      </p>
      <div className="flex shrink-0 flex-wrap items-center gap-3">
        <Segmented
          label="Whose subagents"
          size="md"
          options={WHO}
          value={isMine ? "me" : ""}
          onChange={(next) => patch({ who: next })}
        />
        <SpendPeriodPicker
          value={period}
          onChange={(next) => {
            refreshClock()
            patch({ period: next })
          }}
        />
      </div>
    </div>
  )

  // The read waits for a workspace that will not come: say so, not "Loading".
  if (workspaceId === null && isWorkspaceUnknown) {
    return (
      <div className="flex flex-col gap-6">
        {intro}
        <ErrorBanner error={WORKSPACE_UNREAD} />
      </div>
    )
  }

  // "Just me" with no id to narrow by: the read is held back, so it would
  // otherwise load forever. A roster that could not be read is not a caller
  // with nothing of their own.
  if (isMine && owner.ownerId === "") {
    return (
      <div className="flex flex-col gap-6">
        {intro}
        {owner.isPending ? (
          <PageLoading />
        ) : owner.isFailed ? (
          <ErrorBanner error={OWNER_UNREAD} />
        ) : (
          <EmptyState
            title="Nothing is recorded under your name"
            description="Your account has no requests of its own here yet, so there are no subagents to count as yours."
          />
        )}
      </div>
    )
  }

  const data = savings.data
  if (savings.isPending && !data) {
    return (
      <div className="flex flex-col gap-6">
        {intro}
        <PageLoading />
      </div>
    )
  }
  if (!data) {
    return (
      <div className="flex flex-col gap-6">
        {intro}
        <ErrorBanner error={savings.error} />
      </div>
    )
  }

  const { totals } = data
  const words = windowWords(window)
  if (totals.decisions === 0) {
    return (
      <div className="flex flex-col gap-6">
        {intro}
        <ErrorBanner error={savings.error} />
        {data.recommender_available ? (
          <EmptyState
            title={`No recommendations ${words}`}
            description="Install the otari-router plugin in Claude Code, or pi-otari in pi, and each subagent it starts shows up here with what it saved."
          />
        ) : (
          <EmptyState
            title="No subagent router is switched on"
            description="A deployment operator switches one on from the Subagent router page. Until then nothing answers a recommendation, and there is nothing to save."
          />
        )}
      </div>
    )
  }

  const priced = pricedCount(totals)
  const hasMoney = priced > 0
  const isEstimate = totals.estimated > 0
  const points = chartPoints(data.series)

  return (
    <div className="flex flex-col gap-6">
      {intro}
      <ErrorBanner error={savings.error} />
      {totals.with_outcome === 0 ? (
        <InfoBanner>
          No subagent has reported what it consumed {words}, so there is no
          saving to show yet. The otari-router plugin reports from version
          0.2.0.
        </InfoBanner>
      ) : null}
      <KpiStrip isEmpty={!hasMoney} columns={4}>
        <KpiCell
          label={figureLabel("Saved", window)}
          value={hasMoney ? formatSaved(totals.net_saved) : "—"}
          severity={
            hasMoney && totals.net_saved < 0
              ? { status: "alert", word: "Loss" }
              : undefined
          }
          subline={
            !hasMoney
              ? "No priced outcome yet"
              : isEstimate
                ? "Estimated"
                : "Measured"
          }
        />
        <KpiCell
          label="Would have spent"
          value={hasMoney ? formatSpend(totals.baseline_cost) : "—"}
          subline="On the baseline model"
        />
        <KpiCell
          label="Spent"
          value={formatSpend(spentOn(totals))}
          subline={`${formatSpend(totals.router_cost)} on the router`}
        />
        <KpiCell
          label="Recommendations"
          value={formatNumber(totals.decisions)}
          subline={`${formatNumber(priced)} priced`}
        />
      </KpiStrip>
      {/* Whole, on a line of its own: a cell's subline truncates, and this is
          what says which recommendations the money leaves out. */}
      <p className="text-sm text-muted">
        Of {formatNumber(totals.decisions)}{" "}
        {totals.decisions === 1 ? "recommendation" : "recommendations"}:{" "}
        {coverageLine(totals)}.
        {totals.router_unanswered > 0
          ? ` ${formatNumber(totals.router_unanswered)} more ${totals.router_unanswered === 1 ? "request" : "requests"} went unanswered.`
          : null}
      </p>
      {hasMoney ? (
        <section aria-label="Over time" className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-title">Over time</h2>
            <ChartLegend series={CHART_SERIES} />
          </div>
          <TrendChart
            data={points}
            series={CHART_SERIES}
            formatValue={formatSpend}
            formatXTick={(x) => formatBucketTick(x, window.bucket)}
            ariaLabel={`What routed subagents cost and saved per ${window.bucket}`}
            showYAxis
          />
          <p className="text-sm text-muted">
            A bar's height is what the subagents would have cost. A{" "}
            {window.bucket} that lost money shows what was spent alone.
          </p>
        </section>
      ) : null}
      <AgentTypeBreakdown rows={data.by_agent_type} />
      <SwapBreakdown rows={data.by_swap} windowWords={words} />
      <section
        aria-label="How this is counted"
        className="flex max-w-prose flex-col gap-2 text-sm text-muted"
      >
        <h2 className="text-title">How this is counted</h2>
        <p>
          The baseline is the model the subagent would have started on without a
          recommendation: the model the caller asked for, else the parent's. A
          subagent type that sets a model of its own would not have run on the
          parent's, so its saving can read high.
        </p>
        <p>
          The same tokens are priced at both models from this deployment's
          rates. A different model would not have used exactly the same tokens,
          tokenizers differ between vendors, and long-context prices are not
          applied, so a figure marked estimated is an estimate.
        </p>
        <p>
          Every recommendation's cost is taken off, also where the subagent
          never reported or could not be priced, so a saving can be negative.
        </p>
      </section>
    </div>
  )
}
