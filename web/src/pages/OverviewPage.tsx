import { useMemo } from "react";

import { useKeys, useModels, useUsage, useUsers } from "@/api/hooks";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ErrorBanner, PageHeader, StatCard } from "@/components/ui";
import { formatCost, formatNumber } from "@/lib/format";
import { providersFromUsage, summarizeUsage, usageByModel } from "@/lib/usage";

const USAGE_LIMIT = 500;

export function OverviewPage() {
  const users = useUsers();
  const keys = useKeys();
  const models = useModels();
  const usage = useUsage(USAGE_LIMIT);

  const entries = useMemo(() => usage.data ?? [], [usage.data]);
  const totals = useMemo(() => summarizeUsage(entries), [entries]);
  const topModels = useMemo(() => usageByModel(entries).slice(0, 5), [entries]);

  const providerCount = useMemo(() => {
    const names = new Set(providersFromUsage(entries));
    for (const model of models.data?.data ?? []) {
      if (model.owned_by) {
        names.add(model.owned_by);
      }
    }
    return names.size;
  }, [entries, models.data]);

  const activeKeys = useMemo(() => (keys.data ?? []).filter((key) => key.is_active).length, [keys.data]);

  const dash = (value: number, isLoading: boolean) => (isLoading ? "…" : formatNumber(value));

  return (
    <div className="flex flex-col gap-6">
      <PageHeader title="Overview" description="A snapshot of your gateway: users, providers, models, and recent traffic." />

      <ErrorBanner error={users.error ?? keys.error ?? models.error ?? usage.error} />

      <div className="flex flex-wrap gap-4">
        <StatCard label="Users" value={dash(users.data?.length ?? 0, users.isLoading)} />
        <StatCard label="Providers" value={dash(providerCount, models.isLoading || usage.isLoading)} />
        <StatCard label="Models" value={dash(models.data?.data.length ?? 0, models.isLoading)} />
        <StatCard
          label="API keys"
          value={dash(keys.data?.length ?? 0, keys.isLoading)}
          hint={`${formatNumber(activeKeys)} active`}
        />
      </div>

      <div className="flex flex-wrap gap-4">
        <StatCard label="Requests" value={dash(totals.requests, usage.isLoading)} hint={`last ${USAGE_LIMIT} logged`} />
        <StatCard
          label="Tokens"
          value={dash(totals.totalTokens, usage.isLoading)}
          hint={`${formatNumber(totals.promptTokens)} in · ${formatNumber(totals.completionTokens)} out`}
        />
        <StatCard label="Cost" value={usage.isLoading ? "…" : formatCost(totals.cost)} />
        <StatCard
          label="Errors"
          value={dash(totals.errors, usage.isLoading)}
          hint={`${(totals.errorRate * 100).toFixed(1)}% of requests`}
        />
      </div>

      <div className="flex flex-col gap-3">
        <h2 className="text-base font-semibold text-[var(--otari-ink)]">Top models</h2>
        <Table>
          <THead>
            <Tr>
              <Th>Model</Th>
              <Th>Provider</Th>
              <Th className="text-right">Requests</Th>
              <Th className="text-right">Tokens</Th>
              <Th className="text-right">Cost</Th>
            </Tr>
          </THead>
          <tbody>
            {usage.isLoading ? (
              <LoadingRow colSpan={5} />
            ) : topModels.length > 0 ? (
              topModels.map((row) => (
                <Tr key={row.key}>
                  <Td className="font-medium break-all">{row.model}</Td>
                  <Td className="text-[var(--otari-muted)]">{row.provider}</Td>
                  <Td className="text-right">{formatNumber(row.requests)}</Td>
                  <Td className="text-right text-[var(--otari-muted)]">{formatNumber(row.totalTokens)}</Td>
                  <Td className="text-right">{formatCost(row.cost)}</Td>
                </Tr>
              ))
            ) : (
              <TableMessage colSpan={5}>No usage recorded yet.</TableMessage>
            )}
          </tbody>
        </Table>
      </div>
    </div>
  );
}
