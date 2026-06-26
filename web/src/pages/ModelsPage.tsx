import { Chip } from "@heroui/react";
import { useMemo } from "react";

import { useModels, useUsage } from "@/api/hooks";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ErrorBanner, PageHeader } from "@/components/ui";
import { formatCost, formatNumber } from "@/lib/format";
import { providerFromModelKey } from "@/lib/pricing";
import { usageByModel } from "@/lib/usage";

const USAGE_LIMIT = 500;

interface ModelRow {
  key: string;
  model: string;
  provider: string;
  inputPrice: number | null;
  outputPrice: number | null;
  pricingSource: string;
  requests: number;
  totalTokens: number;
  cost: number;
}

export function ModelsPage() {
  const models = useModels();
  const usage = useUsage(USAGE_LIMIT);

  const rows = useMemo<ModelRow[]>(() => {
    const usageByKey = new Map(usageByModel(usage.data ?? []).map((row) => [row.key, row]));
    const seen = new Set<string>();
    const result: ModelRow[] = [];

    // Start from the full catalog so every configured/discovered model shows,
    // even with zero traffic (this is what the Overview "Models" tile counts).
    for (const model of models.data?.data ?? []) {
      const used = usageByKey.get(model.id);
      seen.add(model.id);
      result.push({
        key: model.id,
        model: model.id,
        provider: model.owned_by || providerFromModelKey(model.id),
        inputPrice: model.pricing?.input_price_per_million ?? null,
        outputPrice: model.pricing?.output_price_per_million ?? null,
        pricingSource: model.pricing_source,
        requests: used?.requests ?? 0,
        totalTokens: used?.totalTokens ?? 0,
        cost: used?.cost ?? 0,
      });
    }

    // Include anything seen in usage that the catalog doesn't list.
    for (const used of usageByKey.values()) {
      if (seen.has(used.key)) {
        continue;
      }
      result.push({
        key: used.key,
        model: used.model,
        provider: used.provider,
        inputPrice: null,
        outputPrice: null,
        pricingSource: "none",
        requests: used.requests,
        totalTokens: used.totalTokens,
        cost: used.cost,
      });
    }

    return result.sort((a, b) => b.requests - a.requests || a.model.localeCompare(b.model));
  }, [models.data, usage.data]);

  const isLoading = models.isLoading || usage.isLoading;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Models"
        description="Every configured and discovered model, with its price and recent usage."
      />

      <ErrorBanner error={models.error ?? usage.error} />

      <Table>
        <THead>
          <Tr>
            <Th>Model</Th>
            <Th>Provider</Th>
            <Th className="text-right">Input $ / 1M</Th>
            <Th className="text-right">Output $ / 1M</Th>
            <Th className="text-right">Requests</Th>
            <Th className="text-right">Tokens</Th>
            <Th className="text-right">Cost</Th>
          </Tr>
        </THead>
        <tbody>
          {isLoading ? (
            <LoadingRow colSpan={7} />
          ) : rows.length > 0 ? (
            rows.map((row) => (
              <Tr key={row.key}>
                <Td className="font-medium break-all">
                  <span className="inline-flex items-center gap-2">
                    {row.model}
                    {row.pricingSource === "default" ? (
                      <Chip size="sm" color="accent">
                        default
                      </Chip>
                    ) : null}
                  </span>
                </Td>
                <Td className="text-[var(--otari-muted)]">{row.provider}</Td>
                <Td className="text-right">{row.inputPrice == null ? "—" : formatCost(row.inputPrice)}</Td>
                <Td className="text-right">{row.outputPrice == null ? "—" : formatCost(row.outputPrice)}</Td>
                <Td className="text-right">{formatNumber(row.requests)}</Td>
                <Td className="text-right text-[var(--otari-muted)]">{formatNumber(row.totalTokens)}</Td>
                <Td className="text-right">{formatCost(row.cost)}</Td>
              </Tr>
            ))
          ) : (
            <TableMessage colSpan={7}>No models configured yet.</TableMessage>
          )}
        </tbody>
      </Table>
    </div>
  );
}
