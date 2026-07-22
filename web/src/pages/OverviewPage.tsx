import { useMemo } from "react";
import { NavLink, Navigate } from "react-router-dom";

import {
  useBudgets,
  useKeys,
  useProviderHealth,
  useProviders,
  useUsageLogs,
  useUsageSummary,
  useUsers,
} from "@/api/hooks";
import type { UsageEntry } from "@/api/types";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { DeltaHint, ErrorBanner, PageHeader, StatCard } from "@/components/ui";
import { deltaFraction, formatNumber, formatPct, formatRelative, formatUsd } from "@/lib/format";
import { budgetHealth, errorRateHealth, providerHealthStatus, toStatStatus } from "@/lib/overview";

const DAY_MS = 86_400_000;
const PERIOD_DAYS = 30;

// How often the health tile re-dials providers in the background. Reachability
// does not move second to second, so a minute keeps the overview live without
// hammering every upstream.
const HEALTH_REFRESH_MS = 60_000;

// Freeze the window bounds once, so the summary queries get a stable queryKey and
// do not refetch every render (and the vs-prev delta compares a fixed baseline).
// "Today" is LOCAL midnight rendered to an absolute instant, so the operator's
// wall-clock day is what the server aggregates, not UTC's.
function useWindows() {
  return useMemo(() => {
    const now = Date.now();
    const d = new Date(now);
    return {
      today: new Date(d.getFullYear(), d.getMonth(), d.getDate()).toISOString(),
      periodStart: new Date(now - PERIOD_DAYS * DAY_MS).toISOString(),
      prevStart: new Date(now - 2 * PERIOD_DAYS * DAY_MS).toISOString(),
    };
  }, []);
}

// A status tile's short word (paired with the color so status never rides on hue
// alone), keyed off the derived Health.
const ERROR_WORDS = { ok: "Healthy", warn: "Elevated", alert: "High" } as const;
const PROVIDER_WORDS = { ok: "All up", warn: "Degraded", alert: "All down" } as const;
const BUDGET_WORDS = { ok: "On track", warn: "Near limit", alert: "Over budget" } as const;

// The authenticated index. On a fresh gateway with no provider configured, a new
// admin is sent to Providers to add one (an all-zeros overview is a poor first
// run); gating here means the overview's query fan-out never fires on that
// redirect. The providers query is master-key gated, so this only runs once
// authenticated.
export function OverviewIndex() {
  const providers = useProviders();
  if (providers.isLoading) {
    return null;
  }
  // Never strand the index on a blank screen: a failed providers query falls back
  // to Providers (where the error surfaces) rather than rendering nothing.
  if (!providers.isSuccess || providers.data.providers.length === 0) {
    return <Navigate to="/providers" replace />;
  }
  return <OverviewPage />;
}

export function OverviewPage() {
  const w = useWindows();

  const todayFilters = useMemo(() => ({ start_date: w.today }), [w]);
  const periodFilters = useMemo(() => ({ start_date: w.periodStart }), [w]);
  // Bounded previous window ([-60d, -30d)) so it does not overlap the current one.
  const prevFilters = useMemo(() => ({ start_date: w.prevStart, end_date: w.periodStart }), [w]);

  const today = useUsageSummary(todayFilters, "hour");
  const period = useUsageSummary(periodFilters, "day");
  const previous = useUsageSummary(prevFilters, "day");
  const health = useProviderHealth(HEALTH_REFRESH_MS);
  const budgets = useBudgets();
  const keys = useKeys();
  const users = useUsers();
  const recent = useUsageLogs({}, 0, 5);

  const todayTotals = today.data?.totals;
  const periodTotals = period.data?.totals;
  const prevTotals = previous.data?.totals;

  const err = errorRateHealth(periodTotals);
  const errPrev = errorRateHealth(prevTotals);
  const errDelta = err.rate !== null && errPrev.rate !== null ? deltaFraction(err.rate, errPrev.rate) : null;

  const budget = budgetHealth(budgets.data ?? []);
  const providerHealth = providerHealthStatus(health.data);

  const activeKeys = (keys.data ?? []).filter((k) => k.is_active).length;
  const activeUsers = (users.data ?? []).filter((u) => !u.blocked).length;

  // Surface the first load error across the tile queries so a broken master key
  // or backend does not just leave a wall of "—".
  const loadError =
    today.error ?? period.error ?? health.error ?? budgets.error ?? keys.error ?? users.error ?? recent.error;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader title="Overview" description="At-a-glance spend, traffic, and health across the gateway." />

      <ErrorBanner error={loadError} />

      <SystemStatusStrip
        providerHealth={providerHealth}
        healthy={health.data?.healthy ?? 0}
        total={health.data?.total ?? 0}
        budget={budget}
        errStatus={err.status}
        errRate={err.rate}
      />

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
        <StatCard label="Spend today" value={today.isLoading ? "—" : formatUsd(todayTotals?.cost ?? 0)} />
        <StatCard
          label="Spend, last 30 days"
          value={period.isLoading ? "—" : formatUsd(periodTotals?.cost ?? 0)}
          hint={periodTotals ? <DeltaHint fraction={deltaFraction(periodTotals.cost, prevTotals?.cost)} /> : null}
        />
        <StatCard
          label="Requests, last 30 days"
          value={period.isLoading ? "—" : formatNumber(periodTotals?.request_count ?? 0)}
          hint={
            periodTotals ? (
              <DeltaHint fraction={deltaFraction(periodTotals.request_count, prevTotals?.request_count)} />
            ) : null
          }
        />
        <StatCard
          label="Error rate, last 30 days"
          value={period.isLoading ? "—" : err.rate === null ? "—" : formatPct(err.rate)}
          status={toStatStatus(err.status)}
          statusLabel={err.status === "neutral" ? undefined : ERROR_WORDS[err.status]}
          hint={err.rate !== null ? <DeltaHint fraction={errDelta} /> : null}
        />
        <StatCard
          label="Budget health"
          value={budgets.isLoading ? "—" : budget.worst ? formatPct(budget.worst.pct) : "—"}
          status={toStatStatus(budget.status)}
          statusLabel={budget.status === "neutral" ? undefined : BUDGET_WORDS[budget.status]}
          hint={budget.worst ? `${budget.label} · worst: ${budget.worst.name}` : budget.label}
          to="/budgets"
        />
        <StatCard
          label="Providers healthy"
          value={health.isLoading ? "—" : `${health.data?.healthy ?? 0}/${health.data?.total ?? 0}`}
          status={toStatStatus(providerHealth)}
          statusLabel={providerHealth === "neutral" ? undefined : PROVIDER_WORDS[providerHealth]}
          hint={
            health.data?.checked_at
              ? `checked ${formatRelative(health.data.checked_at)}`
              : health.isLoading
                ? undefined
                : "not checked yet"
          }
          to="/providers"
        />
        <StatCard label="Active keys" value={keys.isLoading ? "—" : formatNumber(activeKeys)} to="/keys" />
        <StatCard label="Active users" value={users.isLoading ? "—" : formatNumber(activeUsers)} to="/users" />
      </div>

      <RecentActivity entries={recent.data ?? []} loading={recent.isLoading} error={recent.error} />
    </div>
  );
}

// A calm one-liner when all clear; otherwise only the things that need attention,
// each linking to the page that fixes it. This is the attention-router: the
// operator should not have to scan every tile to learn whether anything is wrong.
function SystemStatusStrip({
  providerHealth,
  healthy,
  total,
  budget,
  errStatus,
  errRate,
}: {
  providerHealth: "ok" | "warn" | "alert" | "neutral";
  healthy: number;
  total: number;
  budget: ReturnType<typeof budgetHealth>;
  errStatus: "ok" | "warn" | "alert" | "neutral";
  errRate: number | null;
}) {
  const problems: { text: string; to: string }[] = [];
  if ((providerHealth === "warn" || providerHealth === "alert") && total > 0) {
    const down = total - healthy;
    problems.push({ text: `${down} provider${down === 1 ? "" : "s"} unreachable`, to: "/providers" });
  }
  if (budget.overCount > 0) {
    problems.push({ text: `${budget.overCount} budget${budget.overCount === 1 ? "" : "s"} over limit`, to: "/budgets" });
  } else if (budget.nearCount > 0) {
    problems.push({ text: `${budget.nearCount} budget${budget.nearCount === 1 ? "" : "s"} near limit`, to: "/budgets" });
  }
  if (errStatus === "alert" && errRate !== null) {
    problems.push({ text: `error rate ${formatPct(errRate)}`, to: "/activity?status=error" });
  }

  if (problems.length === 0) {
    const providerBit = total > 0 ? `${healthy}/${total} providers healthy` : "no providers configured";
    return (
      <div
        role="status"
        className="flex items-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800"
      >
        <span aria-hidden>✓</span>
        <span>All systems normal · {providerBit} · budgets within limit</span>
      </div>
    );
  }

  return (
    <div
      role="alert"
      className="flex flex-col gap-2 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 sm:flex-row sm:flex-wrap sm:items-center"
    >
      <span className="font-medium">Needs attention:</span>
      {problems.map((p, i) => (
        <span key={p.to + p.text} className="flex items-center gap-2">
          {i > 0 ? <span aria-hidden className="text-amber-400">·</span> : null}
          <NavLink to={p.to} className="underline underline-offset-2 hover:text-amber-950">
            {p.text}
          </NavLink>
        </span>
      ))}
    </div>
  );
}

function statusWord(status: string): string {
  return status === "error" ? "error" : "ok";
}

// Newest few requests, as an at-a-glance preview. Rows are read-only; a single
// "View all" link opens the full Activity log. Cost is nullable per row.
function RecentActivity({ entries, loading, error }: { entries: UsageEntry[]; loading: boolean; error: unknown }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-[var(--otari-ink)]">Recent activity</h2>
        <NavLink to="/activity" className="text-sm text-[var(--otari-brand-dark)] hover:underline">
          View all →
        </NavLink>
      </div>
      <ErrorBanner error={error} />
      <Table>
        <THead>
          <tr>
            <Th>Time</Th>
            <Th>Model</Th>
            <Th className="text-right">Cost</Th>
            <Th>Status</Th>
          </tr>
        </THead>
        <tbody>
          {loading ? (
            <LoadingRow colSpan={4} />
          ) : entries.length === 0 ? (
            <TableMessage colSpan={4}>No requests yet. Once the gateway serves traffic, it appears here.</TableMessage>
          ) : (
            entries.map((entry) => (
              <Tr key={entry.id}>
                <Td className="text-[var(--otari-muted)]">
                  <span title={new Date(entry.timestamp).toLocaleString()}>{formatRelative(entry.timestamp)}</span>
                </Td>
                <Td className="text-[var(--otari-ink)]">{entry.model}</Td>
                <Td className="text-right tabular-nums">{entry.cost === null ? "—" : formatUsd(entry.cost)}</Td>
                <Td>
                  <span
                    className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${
                      entry.status === "error"
                        ? "border-red-200 bg-red-50 text-red-700"
                        : "border-[var(--otari-line)] bg-[var(--otari-brand-tint)] text-[var(--otari-brand-dark)]"
                    }`}
                  >
                    {statusWord(entry.status)}
                  </span>
                </Td>
              </Tr>
            ))
          )}
        </tbody>
      </Table>
    </div>
  );
}
