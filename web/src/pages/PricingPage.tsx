import { Button, Card, Chip } from "@heroui/react";
import { useMemo, useState } from "react";

import {
  useBackfillUsageCost,
  useDeletePricing,
  useModels,
  usePricing,
  useSetPricing,
  useSettings,
  useUsage,
} from "@/api/hooks";
import type { PricingResponse } from "@/api/types";
import { Field } from "@/components/Field";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ConfirmButton, ErrorBanner, errorMessage, InfoBanner, PageHeader } from "@/components/ui";
import { formatCost, formatDateTime, formatNumber } from "@/lib/format";
import { currentPricing, providerFromModelKey } from "@/lib/pricing";
import { usedModelKeys } from "@/lib/usage";

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

function isValidPrice(value: string): boolean {
  const parsed = Number(value);
  return value.trim() !== "" && Number.isFinite(parsed) && parsed >= 0;
}

function BackfillPanel({ modelKey, onDone }: { modelKey: string; onDone: () => void }) {
  const backfill = useBackfillUsageCost();

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-[var(--otari-line)] bg-[var(--otari-bg)] p-4">
      <p className="text-sm text-[var(--otari-ink)]">
        Price saved for <code className="rounded bg-white px-1">{modelKey}</code>.
      </p>
      {backfill.data ? (
        <p className="text-sm text-[var(--otari-brand-dark)]">
          Backfilled {formatNumber(backfill.data.rows_updated)} request
          {backfill.data.rows_updated === 1 ? "" : "s"}, adding {formatCost(backfill.data.cost_added)} of spend across{" "}
          {formatNumber(backfill.data.users_updated)} user{backfill.data.users_updated === 1 ? "" : "s"}.
        </p>
      ) : (
        <p className="text-sm text-[var(--otari-muted)]">
          If this model ran before it was priced, backfill recomputes the cost on those past requests and adds it to
          each user's spend.
        </p>
      )}
      <ErrorBanner error={backfill.error} />
      <div className="flex items-center gap-2">
        {backfill.data ? null : (
          <Button variant="primary" isDisabled={backfill.isPending} onPress={() => backfill.mutate(modelKey)}>
            {backfill.isPending ? "Backfilling…" : "Backfill past usage"}
          </Button>
        )}
        <Button variant="ghost" onPress={onDone}>
          Done
        </Button>
      </div>
    </div>
  );
}

interface PricePreset {
  modelKey: string;
  input: string;
  output: string;
}

function AddPricingForm({
  unpricedUsedModels,
  preset,
  onClose,
}: {
  unpricedUsedModels: string[];
  preset?: PricePreset;
  onClose: () => void;
}) {
  const setPricing = useSetPricing();
  const [modelKey, setModelKey] = useState(preset?.modelKey ?? "");
  const [input, setInput] = useState(preset?.input ?? "");
  const [output, setOutput] = useState(preset?.output ?? "");
  const [savedModelKey, setSavedModelKey] = useState<string | null>(null);

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
      { onSuccess: () => setSavedModelKey(key) },
    );
  };

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-[var(--otari-ink)]">Set pricing</h2>
          <Button size="sm" variant="ghost" onPress={onClose}>
            Close
          </Button>
        </div>

        {savedModelKey ? (
          <BackfillPanel modelKey={savedModelKey} onDone={onClose} />
        ) : (
          <>
            <ErrorBanner error={setPricing.error} />
            {unpricedUsedModels.length > 0 ? (
              <label className="flex flex-col gap-1">
                <span className="text-sm font-medium text-[var(--otari-ink)]">Used model without a price</span>
                <select
                  aria-label="Used model without a price"
                  value=""
                  onChange={(event) => {
                    if (event.target.value) {
                      setModelKey(event.target.value);
                    }
                  }}
                  className="w-full rounded-md border border-[var(--otari-line)] bg-white px-2 py-2 text-sm focus:border-[var(--otari-brand)] focus:outline-none"
                >
                  <option value="">Pick a model that has been used…</option>
                  {unpricedUsedModels.map((key) => (
                    <option key={key} value={key}>
                      {key}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
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
                {setPricing.isPending ? "Saving…" : "Save pricing"}
              </Button>
            </div>
          </>
        )}
      </Card.Content>
    </Card>
  );
}

function PricingRow({ row }: { row: PricingResponse }) {
  const setPricing = useSetPricing();
  const deletePricing = useDeletePricing();
  const [editing, setEditing] = useState(false);
  const [input, setInput] = useState(String(row.input_price_per_million));
  const [output, setOutput] = useState(String(row.output_price_per_million));

  const startEdit = () => {
    setInput(String(row.input_price_per_million));
    setOutput(String(row.output_price_per_million));
    setEditing(true);
  };

  const save = () => {
    if (!isValidPrice(input) || !isValidPrice(output)) {
      return;
    }
    setPricing.mutate(
      {
        model_key: row.model_key,
        input_price_per_million: Number(input),
        output_price_per_million: Number(output),
      },
      { onSuccess: () => setEditing(false) },
    );
  };

  return (
    <Tr>
      <Td className="font-medium break-all">{row.model_key}</Td>
      <Td className="text-[var(--otari-muted)]">{providerFromModelKey(row.model_key)}</Td>
      <Td>
        <Chip size="sm" color="default">
          configured
        </Chip>
      </Td>
      <Td className="text-right">
        {editing ? (
          <MoneyInput value={input} onChange={setInput} ariaLabel={`Input price for ${row.model_key}`} />
        ) : (
          formatCost(row.input_price_per_million)
        )}
      </Td>
      <Td className="text-right">
        {editing ? (
          <MoneyInput value={output} onChange={setOutput} ariaLabel={`Output price for ${row.model_key}`} />
        ) : (
          formatCost(row.output_price_per_million)
        )}
      </Td>
      <Td className="text-[var(--otari-muted)] whitespace-nowrap">{formatDateTime(row.effective_at)}</Td>
      <Td className="text-right whitespace-nowrap">
        {editing ? (
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
              Edit
            </Button>
            <ConfirmButton
              confirmLabel="Delete"
              isPending={deletePricing.isPending}
              onConfirm={() => deletePricing.mutate(row.model_key)}
            >
              Delete
            </ConfirmButton>
            {setPricing.error || deletePricing.error ? (
              <span className="text-xs text-red-700">{errorMessage(setPricing.error ?? deletePricing.error)}</span>
            ) : null}
          </span>
        )}
      </Td>
    </Tr>
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
        Default pricing is <strong>on</strong>: models without an explicit price below are metered using
        community-maintained rates (the bundled genai-prices dataset). Set a price to override the fallback.
      </InfoBanner>
    );
  }
  return (
    <InfoBanner tone="warning">
      Default pricing is <strong>off</strong>: only the models priced below are metered.
      {settings.data.require_pricing
        ? " Requests for any other model are rejected (HTTP 402) because require_pricing is on."
        : " Other models are served without cost tracking."}
    </InfoBanner>
  );
}

interface DefaultRow {
  modelKey: string;
  provider: string;
  inputPrice: number;
  outputPrice: number;
}

function DefaultPricingRow({ row, onOverride }: { row: DefaultRow; onOverride: (preset: PricePreset) => void }) {
  return (
    <Tr>
      <Td className="font-medium break-all">{row.modelKey}</Td>
      <Td className="text-[var(--otari-muted)]">{providerFromModelKey(row.modelKey)}</Td>
      <Td>
        <Chip size="sm" color="accent">
          default
        </Chip>
      </Td>
      <Td className="text-right">{formatCost(row.inputPrice)}</Td>
      <Td className="text-right">{formatCost(row.outputPrice)}</Td>
      <Td className="text-[var(--otari-muted)]">genai-prices</Td>
      <Td className="text-right">
        <Button
          size="sm"
          variant="outline"
          onPress={() =>
            onOverride({
              modelKey: row.modelKey,
              input: String(row.inputPrice),
              output: String(row.outputPrice),
            })
          }
        >
          Override
        </Button>
      </Td>
    </Tr>
  );
}

export function PricingPage() {
  const pricing = usePricing();
  const usage = useUsage();
  const models = useModels();
  const [showForm, setShowForm] = useState(false);
  const [preset, setPreset] = useState<PricePreset | undefined>(undefined);

  const rows = useMemo(() => currentPricing(pricing.data ?? []), [pricing.data]);

  // Models that have been used but have no DB price yet.
  const unpricedUsedModels = useMemo(() => {
    const priced = new Set(rows.map((row) => row.model_key));
    return usedModelKeys(usage.data ?? []).filter((key) => !priced.has(key));
  }, [rows, usage.data]);

  // Of those, the ones the gateway is metering via the genai-prices fallback,
  // shown read-only so you can see the rate and override it if you want.
  const defaultRows = useMemo<DefaultRow[]>(() => {
    const byId = new Map((models.data?.data ?? []).map((model) => [model.id, model]));
    const result: DefaultRow[] = [];
    for (const key of unpricedUsedModels) {
      const model = byId.get(key);
      if (model?.pricing && model.pricing_source === "default") {
        result.push({
          modelKey: key,
          provider: model.owned_by,
          inputPrice: model.pricing.input_price_per_million,
          outputPrice: model.pricing.output_price_per_million,
        });
      }
    }
    return result;
  }, [unpricedUsedModels, models.data]);

  const openForm = (next?: PricePreset) => {
    setPreset(next);
    setShowForm(true);
  };

  const isEmpty = rows.length === 0 && defaultRows.length === 0;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Pricing"
        description="View and edit per-model token prices. Saving an edit records a new price effective now."
        action={
          <Button
            variant="primary"
            onPress={() => (showForm ? setShowForm(false) : openForm(undefined))}
          >
            {showForm ? "Hide form" : "Set pricing"}
          </Button>
        }
      />

      <DefaultPricingBanner />

      {showForm ? (
        <AddPricingForm
          key={preset?.modelKey ?? "new"}
          unpricedUsedModels={unpricedUsedModels}
          preset={preset}
          onClose={() => setShowForm(false)}
        />
      ) : null}

      <ErrorBanner error={pricing.error ?? models.error} />

      <Table>
        <THead>
          <Tr>
            <Th>Model key</Th>
            <Th>Provider</Th>
            <Th>Source</Th>
            <Th className="text-right">Input $ / 1M</Th>
            <Th className="text-right">Output $ / 1M</Th>
            <Th>Effective</Th>
            <Th className="text-right">Actions</Th>
          </Tr>
        </THead>
        <tbody>
          {pricing.isLoading ? (
            <LoadingRow colSpan={7} />
          ) : isEmpty ? (
            <TableMessage colSpan={7}>No pricing yet. Use “Set pricing” to add a model.</TableMessage>
          ) : (
            <>
              {rows.map((row) => (
                <PricingRow key={row.model_key} row={row} />
              ))}
              {defaultRows.map((row) => (
                <DefaultPricingRow key={row.modelKey} row={row} onOverride={openForm} />
              ))}
            </>
          )}
        </tbody>
      </Table>
    </div>
  );
}
