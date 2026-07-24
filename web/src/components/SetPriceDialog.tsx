import { AlertDialog, Button, Input, Label, TextField } from "@heroui/react";
import { useState } from "react";

import { ErrorBanner, InfoBanner } from "@/components/ui";

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

export interface SetPriceDialogProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  /** How many rows the price will be applied to, for the dialog copy. */
  targetCount: number;
  isPending: boolean;
  error: unknown;
  onSubmit: (rates: ManualRates) => void;
}

export function SetPriceDialog({ isOpen, onOpenChange, targetCount, isPending, error, onSubmit }: SetPriceDialogProps) {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [cacheRead, setCacheRead] = useState("");
  const [cacheWrite, setCacheWrite] = useState("");

  const inputRate = parseRate(input);
  const outputRate = parseRate(output);
  const cacheReadRate = parseRate(cacheRead);
  const cacheWriteRate = parseRate(cacheWrite);

  const invalid =
    inputRate === null ||
    Number.isNaN(inputRate) ||
    outputRate === null ||
    Number.isNaN(outputRate) ||
    Number.isNaN(cacheReadRate ?? 0) ||
    Number.isNaN(cacheWriteRate ?? 0);

  const submit = () => {
    if (invalid || inputRate === null || outputRate === null) return;
    onSubmit({
      input_price_per_million: inputRate,
      output_price_per_million: outputRate,
      ...(cacheReadRate !== null && !Number.isNaN(cacheReadRate)
        ? { cache_read_price_per_million: cacheReadRate }
        : {}),
      ...(cacheWriteRate !== null && !Number.isNaN(cacheWriteRate)
        ? { cache_write_price_per_million: cacheWriteRate }
        : {}),
    });
  };

  return (
    <AlertDialog isOpen={isOpen} onOpenChange={onOpenChange}>
      {isOpen ? (
        <AlertDialog.Backdrop>
          <AlertDialog.Container placement="center" size="lg">
            <AlertDialog.Dialog>
              <AlertDialog.Header>
                <AlertDialog.Heading>Set price</AlertDialog.Heading>
              </AlertDialog.Header>
              <AlertDialog.Body className="flex flex-col gap-4">
                <p className="text-sm text-[var(--otari-muted)]">
                  Recompute cost for {targetCount.toLocaleString()} imported {targetCount === 1 ? "row" : "rows"} from
                  each row&rsquo;s own token counts at these per-1M rates. Enforced gateway rows are never affected.
                </p>
                <div className="grid gap-3 sm:grid-cols-2">
                  <RateField label="Input $ / 1M" value={input} onChange={setInput} isRequired autoFocus />
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
