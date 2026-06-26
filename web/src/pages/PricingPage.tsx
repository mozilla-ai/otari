import { Button, Card } from "@heroui/react";
import { useMemo, useState } from "react";

import { useDeletePricing, usePricing, useSetPricing } from "@/api/hooks";
import type { PricingResponse } from "@/api/types";
import { Field } from "@/components/Field";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ConfirmButton, ErrorBanner, PageHeader } from "@/components/ui";
import { formatCost, formatDateTime } from "@/lib/format";
import { currentPricing, providerFromModelKey } from "@/lib/pricing";

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

function AddPricingForm({ onClose }: { onClose: () => void }) {
  const setPricing = useSetPricing();
  const [modelKey, setModelKey] = useState("");
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");

  const canSubmit = modelKey.trim() !== "" && isValidPrice(input) && isValidPrice(output);

  const submit = () => {
    if (!canSubmit) {
      return;
    }
    setPricing.mutate(
      {
        model_key: modelKey.trim(),
        input_price_per_million: Number(input),
        output_price_per_million: Number(output),
      },
      {
        onSuccess: () => {
          setModelKey("");
          setInput("");
          setOutput("");
          onClose();
        },
      },
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
            {setPricing.isPending ? "Saving…" : "Save pricing"}
          </Button>
        </div>
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
          </span>
        )}
      </Td>
    </Tr>
  );
}

export function PricingPage() {
  const pricing = usePricing();
  const setPricing = useSetPricing();
  const deletePricing = useDeletePricing();
  const [showForm, setShowForm] = useState(false);

  const rows = useMemo(() => currentPricing(pricing.data ?? []), [pricing.data]);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Pricing"
        description="View and edit per-model token prices. Saving an edit records a new price effective now."
        action={
          <Button variant="primary" onPress={() => setShowForm((value) => !value)}>
            {showForm ? "Hide form" : "Set pricing"}
          </Button>
        }
      />

      {showForm ? <AddPricingForm onClose={() => setShowForm(false)} /> : null}

      <ErrorBanner error={pricing.error ?? setPricing.error ?? deletePricing.error} />

      <Table>
        <THead>
          <Tr>
            <Th>Model key</Th>
            <Th>Provider</Th>
            <Th className="text-right">Input $ / 1M</Th>
            <Th className="text-right">Output $ / 1M</Th>
            <Th>Effective</Th>
            <Th className="text-right">Actions</Th>
          </Tr>
        </THead>
        <tbody>
          {pricing.isLoading ? (
            <LoadingRow colSpan={6} />
          ) : rows.length > 0 ? (
            rows.map((row) => <PricingRow key={row.model_key} row={row} />)
          ) : (
            <TableMessage colSpan={6}>No pricing set yet. Use “Set pricing” to add a model.</TableMessage>
          )}
        </tbody>
      </Table>
    </div>
  );
}
