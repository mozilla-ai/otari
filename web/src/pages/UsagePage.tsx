import { Card, Chip } from "@heroui/react";
import { useMemo } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { useUsage } from "@/api/hooks";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ErrorBanner, PageHeader, StatCard } from "@/components/ui";
import { formatCost, formatDateTime, formatNumber } from "@/lib/format";
import { summarizeUsage, usageByDay, usageByModel } from "@/lib/usage";

const USAGE_LIMIT = 500;

export function UsagePage() {
  const usage = useUsage(USAGE_LIMIT);
  const entries = useMemo(() => usage.data ?? [], [usage.data]);

  const totals = useMemo(() => summarizeUsage(entries), [entries]);
  const daily = useMemo(() => usageByDay(entries), [entries]);
  const byModel = useMemo(() => usageByModel(entries), [entries]);
  const recent = useMemo(() => entries.slice(0, 25), [entries]);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Usage"
        description={`Traffic and spend across the ${formatNumber(totals.requests)} most recent requests (max ${USAGE_LIMIT}).`}
      />

      <ErrorBanner error={usage.error} />

      <div className="flex flex-wrap gap-4">
        <StatCard label="Requests" value={formatNumber(totals.requests)} />
        <StatCard label="Total tokens" value={formatNumber(totals.totalTokens)} hint={`${formatNumber(totals.promptTokens)} in · ${formatNumber(totals.completionTokens)} out`} />
        <StatCard label="Cost" value={formatCost(totals.cost)} />
        <StatCard
          label="Error rate"
          value={`${(totals.errorRate * 100).toFixed(1)}%`}
          hint={`${formatNumber(totals.errors)} failed`}
        />
      </div>

      <Card>
        <Card.Content className="flex flex-col gap-3 p-5">
          <h2 className="text-base font-semibold text-[var(--otari-ink)]">Requests over time</h2>
          {usage.isLoading ? (
            <div className="h-64 animate-pulse rounded-lg bg-[var(--otari-bg)]" />
          ) : daily.length > 0 ? (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={daily} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                  <defs>
                    <linearGradient id="reqFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#4e8295" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#4e8295" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#dbe5e8" vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 12, fill: "#5b6b73" }} tickLine={false} axisLine={{ stroke: "#dbe5e8" }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 12, fill: "#5b6b73" }} tickLine={false} axisLine={false} width={36} />
                  <Tooltip
                    formatter={(value) => [formatNumber(Number(value)), "Requests"] as [string, string]}
                    contentStyle={{ borderRadius: 8, border: "1px solid #dbe5e8", fontSize: 12 }}
                  />
                  <Area type="monotone" dataKey="requests" stroke="#3c6678" strokeWidth={2} fill="url(#reqFill)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="py-10 text-center text-sm text-[var(--otari-muted)]">No usage recorded yet.</p>
          )}
        </Card.Content>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="flex flex-col gap-3">
          <h2 className="text-base font-semibold text-[var(--otari-ink)]">By model</h2>
          <Table>
            <THead>
              <Tr>
                <Th>Model</Th>
                <Th className="text-right">Requests</Th>
                <Th className="text-right">Tokens</Th>
                <Th className="text-right">Cost</Th>
              </Tr>
            </THead>
            <tbody>
              {usage.isLoading ? (
                <LoadingRow colSpan={4} />
              ) : byModel.length > 0 ? (
                byModel.map((row) => (
                  <Tr key={row.model}>
                    <Td className="font-medium break-all">{row.model}</Td>
                    <Td className="text-right">{formatNumber(row.requests)}</Td>
                    <Td className="text-right text-[var(--otari-muted)]">{formatNumber(row.totalTokens)}</Td>
                    <Td className="text-right">{formatCost(row.cost)}</Td>
                  </Tr>
                ))
              ) : (
                <TableMessage colSpan={4}>No usage recorded yet.</TableMessage>
              )}
            </tbody>
          </Table>
        </div>

        <div className="flex flex-col gap-3">
          <h2 className="text-base font-semibold text-[var(--otari-ink)]">Recent requests</h2>
          <Table>
            <THead>
              <Tr>
                <Th>Time</Th>
                <Th>Model</Th>
                <Th>Status</Th>
                <Th className="text-right">Cost</Th>
              </Tr>
            </THead>
            <tbody>
              {usage.isLoading ? (
                <LoadingRow colSpan={4} />
              ) : recent.length > 0 ? (
                recent.map((entry) => (
                  <Tr key={entry.id}>
                    <Td className="text-[var(--otari-muted)] whitespace-nowrap">{formatDateTime(entry.timestamp)}</Td>
                    <Td className="break-all">{entry.model}</Td>
                    <Td>
                      <Chip size="sm" color={entry.status === "success" ? "success" : "danger"}>
                        {entry.status}
                      </Chip>
                    </Td>
                    <Td className="text-right">{formatCost(entry.cost)}</Td>
                  </Tr>
                ))
              ) : (
                <TableMessage colSpan={4}>No usage recorded yet.</TableMessage>
              )}
            </tbody>
          </Table>
        </div>
      </div>
    </div>
  );
}
