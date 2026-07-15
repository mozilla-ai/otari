import { Button, Card, Chip } from "@heroui/react";
import { useMemo, useState } from "react";

import {
  useBackfillUsageCost,
  useDeletePricing,
  useModels,
  usePricing,
  useSetPricing,
  useSettings,
  useUnlistedModels,
  useUsageSummary,
} from "@/api/hooks";
import { Field } from "@/components/Field";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ConfirmButton, ErrorBanner, errorMessage, InfoBanner, PageHeader } from "@/components/ui";
import { formatCost, formatNumber } from "@/lib/format";
import { currentPricing, providerFromModelKey } from "@/lib/pricing";

// `owned_by` the gateway stamps on a configured alias (ALIAS_OWNED_BY in
// src/gateway/api/routes/models.py). Aliases are display names declared in
// config.yml, not models, so they cannot be priced here: pricing, budgets, and
// usage all key on the resolved target, and the API rejects a price posted
// against an alias name.
const ALIAS_OWNED_BY = "otari";

// Where a row's price comes from. "configured" is a price in the database and
// is the only kind that can be edited or cleared; "default" is the bundled
// genai-prices fallback; "alias" is inherited from the alias's target;
// "none" means the model is metered at no cost.
type PriceSource = "configured" | "default" | "alias" | "none";

interface ModelRow {
  key: string;
  model: string;
  provider: string;
  inputPrice: number | null;
  outputPrice: number | null;
  source: PriceSource;
  requests: number;
  totalTokens: number;
  cost: number;
}

function isValidPrice(value: string): boolean {
  const parsed = Number(value);
  return value.trim() !== "" && Number.isFinite(parsed) && parsed >= 0;
}

function MoneyInput({
  value,
  onChange,
  ariaLabel,
}: {
  value: string;
  onChange: (value: string) => void;
  ariaLabel: string;
}) {
  return (
    <input
      type="number"
      step="any"
      min="0"
      inputMode="decimal"
      aria-label={ariaLabel}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="w-28 rounded-md border border-[var(--otari-line)] bg-white px-2 py-1 text-right text-sm tabular-nums focus:border-[var(--otari-brand)] focus:outline-none"
    />
  );
}

function SourceChip({ source }: { source: PriceSource }) {
  if (source === "configured") {
    return (
      <Chip size="sm" color="default">
        configured
      </Chip>
    );
  }
  if (source === "default") {
    return (
      <Chip size="sm" color="accent">
        default
      </Chip>
    );
  }
  if (source === "alias") {
    return (
      <Chip size="sm" color="accent">
        alias
      </Chip>
    );
  }
  return <span className="text-xs text-[var(--otari-muted)]">not priced</span>;
}

// Offered only after pricing a model that already has traffic: those requests
// were metered at the old (or no) price, so their cost is stale until recomputed.
function BackfillPrompt({ modelKey, onDone }: { modelKey: string; onDone: () => void }) {
  const backfill = useBackfillUsageCost();

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-[var(--otari-line)] bg-[var(--otari-bg)] p-3">
      {backfill.data ? (
        <p className="text-sm text-[var(--otari-brand-dark)]">
          Backfilled {formatNumber(backfill.data.rows_updated)} request
          {backfill.data.rows_updated === 1 ? "" : "s"}, adding {formatCost(backfill.data.cost_added)} of spend across{" "}
          {formatNumber(backfill.data.users_updated)} user{backfill.data.users_updated === 1 ? "" : "s"}.
        </p>
      ) : (
        <p className="text-sm text-[var(--otari-muted)]">
          <code className="rounded bg-white px-1">{modelKey}</code> ran before this price. Recompute the cost of those
          past requests?
        </p>
      )}
      <ErrorBanner error={backfill.error} />
      <div className="flex items-center gap-2">
        {backfill.data ? null : (
          <Button size="sm" variant="primary" isDisabled={backfill.isPending} onPress={() => backfill.mutate(modelKey)}>
            {backfill.isPending ? "Backfilling…" : "Backfill past usage"}
          </Button>
        )}
        <Button size="sm" variant="ghost" onPress={onDone}>
          {backfill.data ? "Done" : "Dismiss"}
        </Button>
      </div>
    </div>
  );
}

function ModelTableRow({ row, onPriced }: { row: ModelRow; onPriced: (modelKey: string) => void }) {
  const setPricing = useSetPricing();
  const deletePricing = useDeletePricing();
  const [editing, setEditing] = useState(false);
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");

  const startEdit = () => {
    // Seed from whatever rate is in effect, so overriding a default price is a
    // tweak rather than re-typing it.
    setInput(row.inputPrice == null ? "" : String(row.inputPrice));
    setOutput(row.outputPrice == null ? "" : String(row.outputPrice));
    setEditing(true);
  };

  const save = () => {
    if (!isValidPrice(input) || !isValidPrice(output)) {
      return;
    }
    setPricing.mutate(
      {
        model_key: row.key,
        input_price_per_million: Number(input),
        output_price_per_million: Number(output),
      },
      {
        onSuccess: () => {
          setEditing(false);
          if (row.requests > 0) {
            onPriced(row.key);
          }
        },
      },
    );
  };

  return (
    <Tr>
      <Td className="font-medium break-all">{row.model}</Td>
      <Td className="text-[var(--otari-muted)]">{row.provider}</Td>
      <Td>
        <SourceChip source={row.source} />
      </Td>
      <Td className="text-right">
        {editing ? (
          <MoneyInput value={input} onChange={setInput} ariaLabel={`Input price for ${row.key}`} />
        ) : row.inputPrice == null ? (
          "—"
        ) : (
          formatCost(row.inputPrice)
        )}
      </Td>
      <Td className="text-right">
        {editing ? (
          <MoneyInput value={output} onChange={setOutput} ariaLabel={`Output price for ${row.key}`} />
        ) : row.outputPrice == null ? (
          "—"
        ) : (
          formatCost(row.outputPrice)
        )}
      </Td>
      <Td className="text-right">{formatNumber(row.requests)}</Td>
      <Td className="text-right text-[var(--otari-muted)]">{formatNumber(row.totalTokens)}</Td>
      <Td className="text-right">{formatCost(row.cost)}</Td>
      <Td className="text-right whitespace-nowrap">
        {row.source === "alias" ? (
          // An alias is priced through its target, and the API rejects a price
          // posted against the alias name, so there is nothing to offer here.
          <span className="text-xs text-[var(--otari-muted)]">priced by its target</span>
        ) : editing ? (
          <span className="inline-flex items-center gap-2">
            <Button
              size="sm"
              variant="primary"
              isDisabled={setPricing.isPending || !isValidPrice(input) || !isValidPrice(output)}
              onPress={save}
            >
              Save
            </Button>
            <Button size="sm" variant="ghost" isDisabled={setPricing.isPending} onPress={() => setEditing(false)}>
              Cancel
            </Button>
          </span>
        ) : (
          <span className="inline-flex items-center gap-2">
            <Button size="sm" variant="outline" onPress={startEdit}>
              {row.source === "configured" ? "Edit" : "Set price"}
            </Button>
            {row.source === "configured" ? (
              <ConfirmButton
                confirmLabel="Clear"
                isPending={deletePricing.isPending}
                onConfirm={() => deletePricing.mutate(row.key)}
              >
                Clear
              </ConfirmButton>
            ) : null}
            {setPricing.error || deletePricing.error ? (
              <span className="text-xs text-red-700">{errorMessage(setPricing.error ?? deletePricing.error)}</span>
            ) : null}
          </span>
        )}
      </Td>
    </Tr>
  );
}

// Prices a model the catalog does not list: an unconfigured provider, or a
// model discovery cannot see.
function AddModelForm({ onClose, onPriced }: { onClose: () => void; onPriced: (modelKey: string) => void }) {
  const setPricing = useSetPricing();
  const [modelKey, setModelKey] = useState("");
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");

  const canSubmit = modelKey.trim() !== "" && isValidPrice(input) && isValidPrice(output);

  const submit = () => {
    if (!canSubmit) {
      return;
    }
    const key = modelKey.trim();
    setPricing.mutate(
      {
        model_key: key,
        input_price_per_million: Number(input),
        output_price_per_million: Number(output),
      },
      {
        onSuccess: () => {
          onPriced(key);
          onClose();
        },
      },
    );
  };

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-[var(--otari-ink)]">Add a model</h2>
          <Button size="sm" variant="ghost" onPress={onClose}>
            Close
          </Button>
        </div>
        <ErrorBanner error={setPricing.error} />
        <div className="grid gap-4 sm:grid-cols-3">
          <Field
            label="Model key"
            value={modelKey}
            onChange={setModelKey}
            placeholder="provider:model"
            isRequired
            autoFocus
            description="e.g. openai:gpt-4o"
          />
          <div className="flex flex-col gap-1">
            <span className="text-sm font-medium text-[var(--otari-ink)]">Input $ / 1M</span>
            <MoneyInput value={input} onChange={setInput} ariaLabel="Input price per million" />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-sm font-medium text-[var(--otari-ink)]">Output $ / 1M</span>
            <MoneyInput value={output} onChange={setOutput} ariaLabel="Output price per million" />
          </div>
        </div>
        <div>
          <Button variant="primary" isDisabled={!canSubmit || setPricing.isPending} onPress={submit}>
            {setPricing.isPending ? "Saving…" : "Save price"}
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

function DefaultPricingBanner() {
  const settings = useSettings();
  if (!settings.data) {
    return null;
  }
  if (settings.data.default_pricing) {
    return (
      <InfoBanner tone="info">
        Default pricing is <strong>on</strong>: models without a configured price are metered using
        community-maintained rates (the bundled genai-prices dataset). Set a price to override the fallback.
      </InfoBanner>
    );
  }
  return (
    <InfoBanner tone="warning">
      Default pricing is <strong>off</strong>: only models with a configured price are metered.
      {settings.data.require_pricing
        ? " Requests for any other model are rejected (HTTP 402) because require_pricing is on."
        : " Other models are served without cost tracking."}
    </InfoBanner>
  );
}

export function ModelsPage() {
  const models = useModels();
  const usage = useUsageSummary();
  const pricing = usePricing();
  const [showForm, setShowForm] = useState(false);
  const [backfillFor, setBackfillFor] = useState<string | null>(null);

  // Models with traffic that the catalog does not describe: an alias's target
  // (withheld on purpose), or anything undiscoverable and unpriced. They are
  // still being billed, so their rate has to be fetched by name.
  const unlistedKeys = useMemo(() => {
    const catalogIds = new Set((models.data?.data ?? []).map((model) => model.id));
    const configuredKeys = new Set(currentPricing(pricing.data ?? []).map((row) => row.model_key));
    return (usage.data?.by_model ?? [])
      .map((row) => row.key)
      .filter((key) => !catalogIds.has(key) && !configuredKeys.has(key));
  }, [models.data, pricing.data, usage.data]);
  const unlisted = useUnlistedModels(unlistedKeys);

  const rows = useMemo<ModelRow[]>(() => {
    // Aggregated server-side over the whole log, so these counts do not drift
    // once the usage table outgrows a single page.
    const usageByKey = new Map((usage.data?.by_model ?? []).map((row) => [row.key, row]));
    // A configured price wins over whatever the catalog reports, and is the
    // only thing this page can edit or clear.
    const configured = new Map(currentPricing(pricing.data ?? []).map((row) => [row.model_key, row]));
    const result: ModelRow[] = [];
    const seen = new Set<string>();

    const add = (key: string, model: string, provider: string, catalogRow?: ModelRow) => {
      if (seen.has(key)) {
        return;
      }
      seen.add(key);
      const used = usageByKey.get(key);
      // An alias reports the price of whatever it resolves to. A pricing row
      // stored under the alias's own name is dead (nothing reads it), so it
      // must not be shown as this row's price.
      const isAlias = catalogRow?.source === "alias";
      const priced = isAlias ? undefined : configured.get(key);
      result.push({
        key,
        model,
        provider,
        inputPrice: priced ? priced.input_price_per_million : (catalogRow?.inputPrice ?? null),
        outputPrice: priced ? priced.output_price_per_million : (catalogRow?.outputPrice ?? null),
        source: priced ? "configured" : (catalogRow?.source ?? "none"),
        requests: used?.requests ?? 0,
        totalTokens: used?.total_tokens ?? 0,
        cost: used?.cost ?? 0,
      });
    };

    // The catalog first, so every configured/discovered model shows even with
    // no traffic.
    for (const model of models.data?.data ?? []) {
      // pricing_source says where an alias's price came from (its target's DB
      // row, or the default fallback), not that it is an alias, so identity
      // comes from owned_by.
      const isAlias = model.owned_by === ALIAS_OWNED_BY;
      const source: PriceSource = isAlias
        ? "alias"
        : model.pricing_source === "default"
          ? "default"
          : model.pricing
            ? "configured"
            : "none";
      add(model.id, model.id, model.owned_by || providerFromModelKey(model.id), {
        key: model.id,
        model: model.id,
        provider: model.owned_by,
        inputPrice: model.pricing?.input_price_per_million ?? null,
        outputPrice: model.pricing?.output_price_per_million ?? null,
        source,
        requests: 0,
        totalTokens: 0,
        cost: 0,
      });
    }
    // Then anything priced or used that the catalog does not list.
    for (const key of configured.keys()) {
      add(key, key, providerFromModelKey(key));
    }
    for (const used of usageByKey.values()) {
      // The gateway's effective rate for this model, if it could name one.
      // Without it the row would claim "not priced" while showing real spend.
      const priced = unlisted.get(used.key);
      add(
        used.key,
        used.model,
        used.provider ?? "—",
        priced
          ? {
              key: used.key,
              model: used.model,
              provider: used.provider ?? "—",
              inputPrice: priced.pricing?.input_price_per_million ?? null,
              outputPrice: priced.pricing?.output_price_per_million ?? null,
              source: priced.pricing_source === "default" ? "default" : priced.pricing ? "configured" : "none",
              requests: 0,
              totalTokens: 0,
              cost: 0,
            }
          : undefined,
      );
    }

    return result.sort((a, b) => b.requests - a.requests || a.model.localeCompare(b.model));
  }, [models.data, usage.data, pricing.data, unlisted]);

  const isLoading = models.isLoading || usage.isLoading || pricing.isLoading;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Models"
        description="Every configured and discovered model, its token price, and recent usage. Saving a price records it as effective now."
        action={
          <Button variant="primary" onPress={() => setShowForm((open) => !open)}>
            {showForm ? "Hide form" : "Add a model"}
          </Button>
        }
      />

      <DefaultPricingBanner />

      {showForm ? <AddModelForm onClose={() => setShowForm(false)} onPriced={setBackfillFor} /> : null}

      {backfillFor ? <BackfillPrompt modelKey={backfillFor} onDone={() => setBackfillFor(null)} /> : null}

      <ErrorBanner error={models.error ?? usage.error ?? pricing.error} />

      <Table>
        <THead>
          <Tr>
            <Th>Model</Th>
            <Th>Provider</Th>
            <Th>Price</Th>
            <Th className="text-right">Input $ / 1M</Th>
            <Th className="text-right">Output $ / 1M</Th>
            <Th className="text-right">Requests</Th>
            <Th className="text-right">Tokens</Th>
            <Th className="text-right">Cost</Th>
            <Th className="text-right">Actions</Th>
          </Tr>
        </THead>
        <tbody>
          {isLoading ? (
            <LoadingRow colSpan={9} />
          ) : rows.length > 0 ? (
            rows.map((row) => <ModelTableRow key={row.key} row={row} onPriced={setBackfillFor} />)
          ) : (
            <TableMessage colSpan={9}>No models yet. Use “Add a model” to price one.</TableMessage>
          )}
        </tbody>
      </Table>
    </div>
  );
}
