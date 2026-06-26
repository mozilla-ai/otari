import { useMemo } from "react";

import { useUsage } from "@/api/hooks";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ErrorBanner, PageHeader } from "@/components/ui";
import { formatCost, formatNumber } from "@/lib/format";
import { usageByModel } from "@/lib/usage";

const USAGE_LIMIT = 500;

export function ModelsPage() {
  const usage = useUsage(USAGE_LIMIT);
  const rows = useMemo(() => usageByModel(usage.data ?? []), [usage.data]);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Model usage"
        description={`Usage rolled up per model across the last ${USAGE_LIMIT} logged requests.`}
      />

      <ErrorBanner error={usage.error} />

      <Table>
        <THead>
          <Tr>
            <Th>Model</Th>
            <Th>Provider</Th>
            <Th className="text-right">Requests</Th>
            <Th className="text-right">Prompt</Th>
            <Th className="text-right">Completion</Th>
            <Th className="text-right">Total tokens</Th>
            <Th className="text-right">Cost</Th>
          </Tr>
        </THead>
        <tbody>
          {usage.isLoading ? (
            <LoadingRow colSpan={7} />
          ) : rows.length > 0 ? (
            rows.map((row) => (
              <Tr key={row.model}>
                <Td className="font-medium break-all">{row.model}</Td>
                <Td className="text-[var(--otari-muted)]">{row.provider}</Td>
                <Td className="text-right">{formatNumber(row.requests)}</Td>
                <Td className="text-right text-[var(--otari-muted)]">{formatNumber(row.promptTokens)}</Td>
                <Td className="text-right text-[var(--otari-muted)]">{formatNumber(row.completionTokens)}</Td>
                <Td className="text-right">{formatNumber(row.totalTokens)}</Td>
                <Td className="text-right">{formatCost(row.cost)}</Td>
              </Tr>
            ))
          ) : (
            <TableMessage colSpan={7}>No usage recorded yet.</TableMessage>
          )}
        </tbody>
      </Table>
    </div>
  );
}
