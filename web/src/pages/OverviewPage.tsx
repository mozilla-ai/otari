import { useMemo } from "react";

import { useUsage } from "@/api/hooks";
import { ErrorBanner, PageHeader, StatCard } from "@/components/ui";
import { formatCost, formatNumber } from "@/lib/format";
import { summarizeUsage } from "@/lib/usage";

const USAGE_LIMIT = 500;

export function OverviewPage() {
  const usage = useUsage(USAGE_LIMIT);

  const entries = useMemo(() => usage.data ?? [], [usage.data]);
  const totals = useMemo(() => summarizeUsage(entries), [entries]);

  const dash = (value: number) => (usage.isLoading ? "…" : formatNumber(value));

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Overview"
        description={`Traffic across the last ${USAGE_LIMIT} logged requests.`}
      />

      <ErrorBanner error={usage.error} />

      <div className="flex flex-wrap gap-4">
        <StatCard label="Requests" value={dash(totals.requests)} />
        <StatCard
          label="Tokens"
          value={dash(totals.totalTokens)}
          hint={`${formatNumber(totals.promptTokens)} in · ${formatNumber(totals.completionTokens)} out`}
        />
        <StatCard label="Cost" value={usage.isLoading ? "…" : formatCost(totals.cost)} />
        <StatCard
          label="Errors"
          value={dash(totals.errors)}
          hint={`${(totals.errorRate * 100).toFixed(1)}% of requests`}
        />
      </div>
    </div>
  );
}
