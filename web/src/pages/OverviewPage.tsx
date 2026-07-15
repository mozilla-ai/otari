import { useUsageSummary } from "@/api/hooks";
import { ErrorBanner, PageHeader, StatCard } from "@/components/ui";
import { formatCost, formatNumber } from "@/lib/format";

export function OverviewPage() {
  const summary = useUsageSummary();
  const totals = summary.data?.totals;
  const errorRate = totals && totals.requests > 0 ? totals.errors / totals.requests : 0;

  const count = (value: number | undefined) => (value === undefined ? "…" : formatNumber(value));

  return (
    <div className="flex flex-col gap-6">
      <PageHeader title="Overview" description="Traffic across every logged request." />

      <ErrorBanner error={summary.error} />

      <div className="flex flex-wrap gap-4">
        <StatCard label="Requests" value={count(totals?.requests)} />
        <StatCard
          label="Tokens"
          value={count(totals?.total_tokens)}
          hint={totals ? `${formatNumber(totals.prompt_tokens)} in · ${formatNumber(totals.completion_tokens)} out` : undefined}
        />
        <StatCard label="Cost" value={totals ? formatCost(totals.cost) : "…"} />
        <StatCard
          label="Errors"
          value={count(totals?.errors)}
          hint={totals ? `${(errorRate * 100).toFixed(1)}% of requests` : undefined}
        />
      </div>
    </div>
  );
}
