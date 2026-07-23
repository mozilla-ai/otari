import { Button, Card } from "@heroui/react";
import { useEffect, useMemo, useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";

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
import { Sparkline } from "@/components/charts";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { DeltaHint, ErrorBanner, PageHeader, StatCard } from "@/components/ui";
import { deltaFraction, formatNumber, formatPct, formatRelative, formatUsd } from "@/lib/format";
import { budgetHealth, errorRateHealth, providerHealthStatus, toStatStatus } from "@/lib/overview";

const DAY_MS = 86_400_000;
const PERIOD_DAYS = 30;

// The operator's current local date, as a stable key. Used to hold the window
// bounds steady within a day (so query keys don't churn every render) while still
// letting them advance across midnight.
export function localDayKey(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

// Window bounds for the summary queries. Held stable within a local day so the
// query keys don't churn (and the vs-prev delta compares a fixed baseline), but
// advanced when the tab is refocused on a new day, so a tab left open overnight
// does not keep aggregating yesterday's "today" or a stale 30-day window.
// "Today" is LOCAL midnight rendered to an absolute instant, so the operator's
// wall-clock day is what the server aggregates, not UTC's.
function useWindows() {
  const [dayKey, setDayKey] = useState(localDayKey);
  useEffect(() => {
    const check = () => {
      if (document.visibilityState === "visible") {
        const next = localDayKey();
        setDayKey((prev) => (prev === next ? prev : next));
      }
    };
    document.addEventListener("visibilitychange", check);
    window.addEventListener("focus", check);
    return () => {
      document.removeEventListener("visibilitychange", check);
      window.removeEventListener("focus", check);
    };
  }, []);
  return useMemo(() => {
    const now = Date.now();
    const d = new Date(now);
    return {
      today: new Date(d.getFullYear(), d.getMonth(), d.getDate()).toISOString(),
      periodStart: new Date(now - PERIOD_DAYS * DAY_MS).toISOString(),
      prevStart: new Date(now - 2 * PERIOD_DAYS * DAY_MS).toISOString(),
    };
    // dayKey drives the refresh: same day -> same bounds; new day -> re-derive.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dayKey]);
}

// A status tile's short word (paired with the color so status never rides on hue
// alone), keyed off the derived Health.
const ERROR_WORDS = { ok: "Healthy", warn: "Elevated", alert: "High" } as const;
const BUDGET_WORDS = { ok: "On track", warn: "Near limit", alert: "Over budget" } as const;

// The authenticated index uses the provider list to make the empty gateway a
// useful getting-started page. The providers query is master-key gated, so this
// only runs once authenticated.
export function OverviewIndex() {
  const providers = useProviders();
  if (providers.isLoading) {
    return null;
  }
  return <OverviewPage needsSetup={providers.isSuccess && providers.data.providers.length === 0} />;
}

export function OverviewPage({ needsSetup = false }: { needsSetup?: boolean }) {
  const w = useWindows();

  const todayFilters = useMemo(() => ({ start_date: w.today }), [w]);
  const periodFilters = useMemo(() => ({ start_date: w.periodStart }), [w]);
  // Bounded previous window ([-60d, -30d)) so it does not overlap the current one.
  const prevFilters = useMemo(() => ({ start_date: w.prevStart, end_date: w.periodStart }), [w]);

  const today = useUsageSummary(todayFilters, "hour");
  const period = useUsageSummary(periodFilters, "day");
  const previous = useUsageSummary(prevFilters, "day");
  const health = useProviderHealth();
  const budgets = useBudgets();
  const keys = useKeys();
  const users = useUsers();
  const recent = useUsageLogs({}, 0, 5);

  const todayTotals = today.data?.totals;
  const periodTotals = period.data?.totals;
  const prevTotals = previous.data?.totals;

  // The 30-day daily series is already on the wire (used for tile sparklines).
  // A single point has no trend to draw, so sparklines only appear with 2+ days.
  const periodSeries = period.data?.series ?? [];
  const hasTrend = periodSeries.length > 1;

  const err = errorRateHealth(periodTotals);
  const errPrev = errorRateHealth(prevTotals);
  const errDelta = err.rate !== null && errPrev.rate !== null ? deltaFraction(err.rate, errPrev.rate) : null;

  const budget = budgetHealth(budgets.data ?? []);
  const providerHealth = providerHealthStatus(health.data);

  const activeKeys = (keys.data ?? []).filter((k) => k.is_active).length;
  const activeUsers = (users.data ?? []).filter((u) => !u.blocked).length;

  // Surface the first load error across the tile queries so a broken master key
  // or backend does not just leave a wall of "—". Recent activity is excluded: it
  // renders its own inline banner, so including it here would double-report.
  const loadError = today.error ?? period.error ?? health.error ?? budgets.error ?? keys.error ?? users.error;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader title="Overview" description="At-a-glance spend, traffic, and health across the gateway." />

      {needsSetup ? <GettingStartedPanel /> : null}

      <ErrorBanner error={loadError} />

      <SystemStatusStrip
        providerHealth={providerHealth}
        healthy={health.data?.healthy ?? 0}
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
        <StatCard label="Spend today" value={todayTotals ? formatUsd(todayTotals.cost) : "—"} />
        <StatCard
          label="Spend, last 30 days"
          value={periodTotals ? formatUsd(periodTotals.cost) : "—"}
          hint={periodTotals ? <DeltaHint fraction={deltaFraction(periodTotals.cost, prevTotals?.cost)} /> : null}
          chart={
            hasTrend ? (
              <Sparkline values={periodSeries.map((p) => p.cost)} ariaLabel="Spend trend over the last 30 days" />
            ) : undefined
          }
        />
        <StatCard
          label="Requests, last 30 days"
          value={periodTotals ? formatNumber(periodTotals.request_count) : "—"}
          hint={
            periodTotals ? (
              <DeltaHint fraction={deltaFraction(periodTotals.request_count, prevTotals?.request_count)} />
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
          statusLabel={err.status === "neutral" ? undefined : ERROR_WORDS[err.status]}
          hint={err.rate !== null ? <DeltaHint fraction={errDelta} /> : null}
        />
        <StatCard
          label="Budget health"
          value={budgets.data ? (budget.worst ? formatPct(budget.worst.pct) : "—") : "—"}
          status={budgets.data ? toStatStatus(budget.status) : undefined}
          statusLabel={budgets.data && budget.status !== "neutral" ? BUDGET_WORDS[budget.status] : undefined}
          hint={budgets.data ? (budget.worst ? `${budget.label} · worst: ${budget.worst.name}` : budget.label) : undefined}
          to="/budgets"
        />
        <StatCard label="Active keys" value={keys.data ? formatNumber(activeKeys) : "—"} to="/keys" />
        <StatCard label="Active users" value={users.data ? formatNumber(activeUsers) : "—"} to="/users" />
      </div>

      <RecentActivity entries={recent.data ?? []} loading={recent.isLoading} error={recent.error} />
    </div>
  );
}

function GettingStartedPanel() {
  const navigate = useNavigate();

  return (
    <Card>
      <Card.Content className="flex flex-col gap-3 p-6">
        <div>
          <h2 className="text-lg font-semibold text-[var(--otari-ink)]">Get started with Otari</h2>
          <p className="mt-1 text-sm text-[var(--otari-muted)]">
            Add a provider to begin serving models. Once it is configured, this page will show your gateway&rsquo;s
            traffic, spend, and health.
          </p>
        </div>
        <div>
          <Button variant="primary" onPress={() => navigate("/providers")}>
            Add your first provider
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

// A neutral, hue-free strip for a failed status source. Its details are also
// surfaced in the ErrorBanner, but this preserves context at the status area.
function NeutralStrip({ text }: { text: string }) {
  return (
    <div
      role="status"
      className="flex items-center gap-2 rounded-xl border border-[var(--otari-line)] bg-[var(--otari-bg)] px-4 py-3 text-sm text-[var(--otari-muted)]"
    >
      {text}
    </div>
  );
}

function SystemStatusStrip({
  providerHealth,
  healthy,
  total,
  budget,
  errStatus,
  errRate,
  ready,
  failed,
}: {
  providerHealth: "ok" | "warn" | "alert" | "neutral";
  healthy: number;
  total: number;
  budget: ReturnType<typeof budgetHealth>;
  errStatus: "ok" | "warn" | "alert" | "neutral";
  errRate: number | null;
  ready: boolean;
  failed: boolean;
}) {
  // A failed source deserves a visible status message; while loading, wait for
  // actionable information instead of reserving space for a transient banner.
  if (failed) {
    return <NeutralStrip text="Some status data could not be loaded." />;
  }
  if (!ready) {
    return null;
  }

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
    return null;
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
