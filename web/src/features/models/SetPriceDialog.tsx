import { AlertDialog, Button, Input, Label, TextField } from "@heroui/react";
import { useEffect, useState } from "react";

import { Field } from "@/shared/ui/Field";
import { ErrorBanner, InfoBanner } from "@/shared/ui/ui";

// Per-1M rates entered by an operator to reprice imported usage rows. Input and
// output are required; the cache rates are optional (blank folds those tokens
// into the fresh-input charge, matching how unset cache pricing behaves).
export interface ManualRates {
  input_price_per_million: number;
  output_price_per_million: number;
  cache_read_price_per_million?: number;
  cache_write_price_per_million?: number;
}

interface RateFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  isRequired?: boolean;
  autoFocus?: boolean;
}

function RateField({ label, value, onChange, isRequired, autoFocus }: RateFieldProps) {
  return (
    <TextField
      value={value}
      onChange={onChange}
      isRequired={isRequired}
      className="flex flex-col gap-1"
    >
      <Label className="text-sm font-medium text-[var(--otari-ink)]">{label}</Label>
      <Input inputMode="decimal" placeholder="0.00" autoFocus={autoFocus} />
    </TextField>
  );
}

function parseRate(value: string): number | null {
  const trimmed = value.trim();
  if (trimmed === "") return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : Number.NaN;
}

// A pricing row is only ever read back under a `prefix:model` selector, so a key
// with no provider or instance prefix would store a price nothing bills against
// (see normalize_pricing_key in services/provider_kwargs.py). Accept the legacy
// slash form too; the backend collapses it onto the colon form.
export function isValidModelKey(value: string): boolean {
  return /^[^\s:/]+[:/][^\s]+$/.test(value.trim());
}

export interface SetPriceDialogProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  /** How many rows the price will be applied to, for the dialog copy. */
  targetCount?: number;
  isPending: boolean;
  error: unknown;
  onSubmit: (rates: ManualRates, modelKey: string) => void;
  /** Dialog heading; defaults to "Set price". */
  title?: string;
  /** Body copy explaining what the rates apply to; a sensible usage default is used when omitted. */
  description?: (count: number) => string;
  /**
   * Also collect the model key the rates apply to, for pricing a model that is
   * not in the catalog (a provider without model discovery). The trimmed key
   * is passed to `onSubmit`; without this the second argument is an empty string.
   */
  collectModelKey?: boolean;
  /**
   * Seeds the model key each time the dialog opens (a selector taken from a
   * search box, a logged request, or a provider prefix). Only read with
   * `collectModelKey`.
   */
  initialModelKey?: string;
}

const defaultDescription = (count: number): string =>
  `Recompute cost for ${count.toLocaleString()} imported ${
    count === 1 ? "row" : "rows"
  } from each row's own token counts at these per-1M rates. Enforced gateway rows are never affected.`;

export function SetPriceDialog({
  isOpen,
  onOpenChange,
  targetCount = 0,
  isPending,
  error,
  onSubmit,
  title = "Set price",
  description = defaultDescription,
  collectModelKey = false,
  initialModelKey = "",
}: SetPriceDialogProps) {
  const [modelKey, setModelKey] = useState(initialModelKey);
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [cacheRead, setCacheRead] = useState("");
  const [cacheWrite, setCacheWrite] = useState("");

  // The dialog stays mounted across close/reopen, so clear the rate fields each time
  // it opens: reopening for a different selection must not inherit the last rates
  // (a real footgun when the values set money).
  useEffect(() => {
    if (isOpen) {
      setModelKey(initialModelKey);
      setInput("");
      setOutput("");
      setCacheRead("");
      setCacheWrite("");
    }
  }, [isOpen, initialModelKey]);

  const inputRate = parseRate(input);
  const outputRate = parseRate(output);
  const cacheReadRate = parseRate(cacheRead);
  const cacheWriteRate = parseRate(cacheWrite);

  const keyInvalid = collectModelKey && !isValidModelKey(modelKey);

  const invalid =
    keyInvalid ||
    inputRate === null ||
    Number.isNaN(inputRate) ||
    outputRate === null ||
    Number.isNaN(outputRate) ||
    Number.isNaN(cacheReadRate ?? 0) ||
    Number.isNaN(cacheWriteRate ?? 0);

  const submit = () => {
    if (invalid || inputRate === null || outputRate === null) return;
    onSubmit(
      {
        input_price_per_million: inputRate,
        output_price_per_million: outputRate,
        ...(cacheReadRate !== null && !Number.isNaN(cacheReadRate)
          ? { cache_read_price_per_million: cacheReadRate }
          : {}),
        ...(cacheWriteRate !== null && !Number.isNaN(cacheWriteRate)
          ? { cache_write_price_per_million: cacheWriteRate }
          : {}),
      },
      modelKey.trim(),
    );
  };

  return (
    <AlertDialog isOpen={isOpen} onOpenChange={onOpenChange}>
      {isOpen ? (
        <AlertDialog.Backdrop>
          <AlertDialog.Container placement="center" size="lg">
            <AlertDialog.Dialog>
              <AlertDialog.Header>
                <AlertDialog.Heading>{title}</AlertDialog.Heading>
              </AlertDialog.Header>
              <AlertDialog.Body className="flex flex-col gap-4">
                <p className="text-sm text-[var(--otari-muted)]">{description(targetCount)}</p>
                {collectModelKey ? (
                  <Field
                    label="Model key"
                    value={modelKey}
                    onChange={setModelKey}
                    placeholder="provider:model"
                    isRequired
                    autoFocus
                    description={
                      modelKey.trim() !== "" && keyInvalid
                        ? "Include the provider or instance prefix, as in ollama:llama3.2."
                        : "The selector callers send as model, prefix included (for example vllm:mistral-small)."
                    }
                  />
                ) : null}
                <div className="grid gap-3 sm:grid-cols-2">
                  <RateField
                    label="Input $ / 1M"
                    value={input}
                    onChange={setInput}
                    isRequired
                    autoFocus={!collectModelKey}
                  />
                  <RateField label="Output $ / 1M" value={output} onChange={setOutput} isRequired />
                  <RateField label="Cache read $ / 1M" value={cacheRead} onChange={setCacheRead} />
                  <RateField label="Cache write $ / 1M" value={cacheWrite} onChange={setCacheWrite} />
                </div>
                <InfoBanner tone="info">
                  Leave a cache rate blank to bill those tokens at the input rate.
                </InfoBanner>
                <ErrorBanner error={error} />
              </AlertDialog.Body>
              <AlertDialog.Footer>
                <Button variant="ghost" isDisabled={isPending} onPress={() => onOpenChange(false)}>
                  Cancel
                </Button>
                <Button variant="primary" isDisabled={invalid} isPending={isPending} onPress={submit}>
                  Set price
                </Button>
              </AlertDialog.Footer>
            </AlertDialog.Dialog>
          </AlertDialog.Container>
        </AlertDialog.Backdrop>
      ) : null}
    </AlertDialog>
  );
}
