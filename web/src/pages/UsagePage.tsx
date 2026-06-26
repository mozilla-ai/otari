import { Chip } from "@heroui/react";
import { useMemo } from "react";

import { useUsage } from "@/api/hooks";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ErrorBanner, PageHeader } from "@/components/ui";
import { formatCost, formatDateTime, formatNumber } from "@/lib/format";

const USAGE_LIMIT = 500;

export function UsagePage() {
  const usage = useUsage(USAGE_LIMIT);
  const entries = useMemo(() => usage.data ?? [], [usage.data]);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Usage"
        description={`Per-request log with token counts and cost (most recent ${USAGE_LIMIT}).`}
      />

      <ErrorBanner error={usage.error} />

      <Table>
        <THead>
          <Tr>
            <Th>Time</Th>
            <Th>Model</Th>
            <Th>Provider</Th>
            <Th>User</Th>
            <Th className="text-right">Prompt</Th>
            <Th className="text-right">Completion</Th>
            <Th className="text-right">Total</Th>
            <Th className="text-right">Cost</Th>
            <Th>Status</Th>
          </Tr>
        </THead>
        <tbody>
          {usage.isLoading ? (
            <LoadingRow colSpan={9} />
          ) : entries.length > 0 ? (
            entries.map((entry) => (
              <Tr key={entry.id}>
                <Td className="text-[var(--otari-muted)] whitespace-nowrap">{formatDateTime(entry.timestamp)}</Td>
                <Td className="break-all">{entry.model}</Td>
                <Td className="text-[var(--otari-muted)]">{entry.provider ?? "—"}</Td>
                <Td className="font-mono text-xs break-all">{entry.user_id ?? "—"}</Td>
                <Td className="text-right text-[var(--otari-muted)]">{formatNumber(entry.prompt_tokens)}</Td>
                <Td className="text-right text-[var(--otari-muted)]">{formatNumber(entry.completion_tokens)}</Td>
                <Td className="text-right">{formatNumber(entry.total_tokens)}</Td>
                <Td className="text-right">{formatCost(entry.cost)}</Td>
                <Td>
                  <Chip size="sm" color={entry.status === "success" ? "success" : "danger"}>
                    {entry.status}
                  </Chip>
                </Td>
              </Tr>
            ))
          ) : (
            <TableMessage colSpan={9}>No usage recorded yet.</TableMessage>
          )}
        </tbody>
      </Table>
    </div>
  );
}
