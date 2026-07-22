import { Button, Card, ComboBox, Input, Label, ListBox, ListBoxItem } from "@heroui/react";
import { useEffect, useId, useState } from "react";
import type { ReactNode } from "react";

import { ApiError } from "@/api/client";

export function StatCard({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <Card className="flex-1 min-w-[180px]">
      <Card.Content className="flex flex-col gap-1 p-5">
        <span className="text-xs font-medium uppercase tracking-wide text-[var(--otari-muted)]">{label}</span>
        <span className="text-2xl font-semibold text-[var(--otari-ink)]">{value}</span>
        {hint ? <span className="text-xs text-[var(--otari-muted)]">{hint}</span> : null}
      </Card.Content>
    </Card>
  );
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Something went wrong.";
}

export function ErrorBanner({ error }: { error: unknown }) {
  if (!error) {
    return null;
  }
  return (
    <div
      role="alert"
      className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
    >
      {errorMessage(error)}
    </div>
  );
}

export function InfoBanner({ tone = "info", children }: { tone?: "info" | "warning"; children: ReactNode }) {
  const styles =
    tone === "warning"
      ? "border-amber-200 bg-amber-50 text-amber-800"
      : "border-[var(--otari-brand)] bg-[var(--otari-brand-tint)] text-[var(--otari-brand-dark)]";
  return <div className={`rounded-lg border px-4 py-3 text-sm ${styles}`}>{children}</div>;
}

export function PageHeader({ title, description, action }: { title: string; description?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col gap-3">
      <div>
        <h1 className="text-xl font-semibold text-[var(--otari-ink)]">{title}</h1>
        {description ? <p className="mt-1 text-sm text-[var(--otari-muted)]">{description}</p> : null}
      </div>
      {/* The primary action sits on its own left-aligned row under the heading,
          so it stays near the sidebar the operator just came from rather than
          across the page at the top right. Wrapped so the button keeps its
          natural size instead of stretching in this flex column. */}
      {action ? <div className="flex flex-wrap gap-2">{action}</div> : null}
    </div>
  );
}

// A destructive button that requires a second click to confirm, avoiding a
// modal dependency for revoke/delete actions.
export function ConfirmButton({
  children,
  confirmLabel,
  onConfirm,
  isPending,
}: {
  children: ReactNode;
  confirmLabel: string;
  onConfirm: () => void;
  isPending?: boolean;
}) {
  const [armed, setArmed] = useState(false);

  if (armed) {
    return (
      <span className="inline-flex items-center gap-1">
        <Button size="sm" variant="danger" isDisabled={isPending} onPress={onConfirm}>
          {confirmLabel}
        </Button>
        <Button size="sm" variant="ghost" isDisabled={isPending} onPress={() => setArmed(false)}>
          Cancel
        </Button>
      </span>
    );
  }

  return (
    <Button size="sm" variant="danger-soft" onPress={() => setArmed(true)}>
      {children}
    </Button>
  );
}

const FILTER_SELECT_CLASS =
  "rounded-lg border border-[var(--otari-line)] bg-[var(--otari-bg)] px-3 py-2 text-sm text-[var(--otari-ink)] focus:border-[var(--otari-brand)] focus:outline-none";

// Token-styled native select for page filter bars. Pass `label` (+ `id`) for a
// visible label, or `ariaLabel` alone for a compact control. Prefer `options`
// for static lists; use `children` when options are grouped or conditional.
export function FilterSelect({
  id,
  label,
  ariaLabel,
  value,
  onChange,
  options,
  children,
  disabled,
}: {
  id?: string;
  label?: string;
  ariaLabel?: string;
  value: string;
  onChange: (value: string) => void;
  options?: { value: string; label: string }[];
  children?: ReactNode;
  disabled?: boolean;
}) {
  const fallbackId = useId();
  const selectId = id ?? (label ? fallbackId : undefined);
  const select = (
    <select
      id={selectId}
      aria-label={label ? undefined : ariaLabel}
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
      className={FILTER_SELECT_CLASS}
    >
      {options
        ? options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))
        : children}
    </select>
  );

  if (label) {
    return (
      <div className="flex flex-col gap-1">
        <label htmlFor={selectId} className="text-xs font-medium text-[var(--otari-muted)]">
          {label}
        </label>
        {select}
      </div>
    );
  }
  return select;
}

// A type-to-filter combobox for page filter bars where the option list is large
// (users, models): a native <select> with thousands of <option>s is unusable, so
// this filters the list as you type and commits on selection. An empty input
// means "no filter" (the caller's cleared value). Options-only by intent: you can
// only filter to a value that exists, matching how the backend filters. If a
// selected value is not in `options` (e.g. seeded from a URL), it still shows.
export function FilterComboBox({
  label,
  value,
  onChange,
  options,
  placeholder,
  maxVisible = 50,
  allowsCustom = false,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  placeholder?: string;
  maxVisible?: number;
  // When true the typed text is itself a valid filter value (committed live, like
  // a plain text box) and the options are suggestions. Use for a filter whose full
  // value space is not enumerable (e.g. any model name the log might hold). When
  // false the control is options-only: typing just narrows the suggestion list and
  // a value is committed only by selecting one.
  allowsCustom?: boolean;
}) {
  const labelFor = (v: string): string => options.find((o) => o.value === v)?.label ?? v;
  const [text, setText] = useState(() => labelFor(value));

  // Re-sync the input when the committed value changes from outside (a "Clear
  // filters" press, or a drill-down seeding the value). Typing does not change
  // `value` until a selection commits, so this does not fight the user mid-type.
  useEffect(() => {
    setText(labelFor(value));
    // labelFor closes over options; value is the external source of truth here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  const query = text.trim().toLowerCase();
  const visible = options
    .filter((o) => !query || o.value.toLowerCase().includes(query) || o.label.toLowerCase().includes(query))
    .slice(0, maxVisible);

  return (
    <ComboBox.Root
      allowsEmptyCollection
      allowsCustomValue={allowsCustom}
      menuTrigger="focus"
      inputValue={text}
      onInputChange={(next) => {
        setText(next);
        if (allowsCustom) {
          // Free-text mode: the input is the filter (committed live, like a plain
          // text box), with the options offered as suggestions.
          onChange(next.trim());
        } else if (next.trim() === "") {
          // Options-only: deleting the text clears the filter; partial text just
          // narrows the dropdown (committing it would filter to a missing value).
          onChange("");
        }
      }}
      onSelectionChange={(key) => {
        if (key != null) {
          onChange(String(key));
        }
      }}
      className="flex flex-col gap-1"
    >
      <Label className="text-xs font-medium text-[var(--otari-muted)]">{label}</Label>
      <ComboBox.InputGroup>
        <Input placeholder={placeholder} autoComplete="off" onFocus={(event) => event.currentTarget.select()} />
        <ComboBox.Trigger />
      </ComboBox.InputGroup>
      <ComboBox.Popover>
        <ListBox items={visible} className="max-h-72 overflow-auto">
          {(option: { value: string; label: string }) => (
            <ListBoxItem id={option.value} textValue={option.label}>
              {option.label}
            </ListBoxItem>
          )}
        </ListBox>
      </ComboBox.Popover>
    </ComboBox.Root>
  );
}
