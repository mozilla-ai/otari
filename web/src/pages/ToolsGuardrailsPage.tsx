import { Button, Card } from "@heroui/react";
import { useEffect, useRef, useState } from "react";

import { useTestService, useToolSettings, useUpdateToolSettings } from "@/api/hooks";
import type { ToolServiceName, ToolSettingField, UpdateToolSettingsRequest } from "@/api/types";
import { ErrorBanner, FilterSelect, PageHeader, errorMessage } from "@/components/ui";

// One settable field maps onto one key of the update request; cast at this one
// boundary (the keys come from the backend's field list).
function oneField(key: string, value: boolean | number | string | null): UpdateToolSettingsRequest {
  return { [key]: value } as UpdateToolSettingsRequest;
}

// The services, in display order, with the fields each owns (url first). Fields
// not listed for a service are still rendered under it via the fallback, but
// this fixes the order and lets us give each service a short blurb.
const SERVICES: { key: ToolServiceName; label: string; blurb: string; order: string[] }[] = [
  {
    key: "web_search",
    label: "Web search",
    blurb: "Backend for otari_web_search tools (a SearXNG instance or a search adapter).",
    order: [
      "web_search_url",
      "web_search_engines",
      "web_search_max_results",
      "web_search_extract",
      "web_search_purpose_hint",
    ],
  },
  {
    key: "sandbox",
    label: "Code execution",
    blurb: "Backend for otari_code_execution tools (the sandbox that runs generated code).",
    order: ["sandbox_url", "sandbox_purpose_hint"],
  },
  {
    key: "guardrails",
    label: "Guardrails",
    blurb: "Default input-guardrails service used when a request does not pass its own guardrail URL.",
    order: ["guardrails_url"],
  },
];

// A short-lived confirmation toast, styled to match the app's ConnectionStatus
// toast (bottom-right), so a save gives visible feedback without a page-level banner.
function useSaveToast(): [string | null, (message: string) => void] {
  const [message, setMessage] = useState<string | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const show = (next: string) => {
    setMessage(next);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setMessage(null), 2500);
  };
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return [message, show];
}

function SaveToast({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed right-4 bottom-4 z-50 flex items-center gap-2 rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm font-medium text-green-700 shadow-lg"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden className="h-5 w-5">
        <path d="M20 6 9 17l-5-5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      {message}
    </div>
  );
}

const INPUT_CLASS =
  "rounded-md border border-[var(--otari-line)] bg-[var(--otari-surface)] px-2 py-1 text-sm focus:border-[var(--otari-brand)] focus:outline-none disabled:opacity-50";

// Every field renders as one grid row with three fixed-width tracks:
// label | input (16rem) | actions (10rem). Because the input and action tracks
// have the same width on every row, the boxes and the Save buttons line up in
// columns down a card regardless of which rows also carry a Test button, an
// extra help line, or a narrower numeric input. Below `sm` the grid collapses
// to a single column and the pieces stack.
const ROW_CLASS = "grid gap-x-4 gap-y-1.5 py-4 sm:grid-cols-[minmax(0,1fr)_16rem_10rem] sm:items-start";
const INPUT_CELL = `w-full sm:col-start-2 ${INPUT_CLASS}`;
const ACTIONS_CELL = "flex items-center gap-2 sm:col-start-3";
const MESSAGE_CELL = "flex flex-col gap-1 sm:col-span-2 sm:col-start-2";

function SaveError({ message }: { message?: string }) {
  if (!message) return null;
  return <span className="break-words text-xs text-red-700">{message}</span>;
}

function FieldLabel({ field, help }: { field: ToolSettingField; help?: string }) {
  return (
    <div className="min-w-0 sm:col-start-1">
      <code className="text-sm font-medium text-[var(--otari-ink)]">{field.key}</code>
      {field.description ? <p className="mt-1 text-sm text-[var(--otari-muted)]">{field.description}</p> : null}
      {help ? <p className="mt-1 text-xs text-[var(--otari-muted)]">{help}</p> : null}
    </div>
  );
}

// A URL row: a shared draft that Test probes and Save commits. Test runs against
// the typed (possibly unsaved) value, so an operator can verify before saving.
function UrlRow({
  field,
  onSave,
  saveError,
  disabled,
}: {
  field: ToolSettingField;
  onSave: (value: string | null) => void;
  saveError?: string;
  disabled: boolean;
}) {
  const committed = typeof field.value === "string" ? field.value : "";
  const [draft, setDraft] = useState(committed);
  const test = useTestService();

  useEffect(() => {
    setDraft(committed);
  }, [committed]);

  const changed = draft.trim() !== committed;
  const trimmed = draft.trim();

  return (
    <div className={ROW_CLASS}>
      <FieldLabel field={field} help="Leave blank and Save to fall back to the configured default." />
      <input
        type="text"
        inputMode="url"
        aria-label={field.key}
        value={draft}
        disabled={disabled}
        placeholder="unset"
        onChange={(event) => {
          setDraft(event.target.value);
          // Drop any prior reachability result so a result for the old URL
          // never sits beside a newly-typed, untested one.
          test.reset();
        }}
        className={INPUT_CELL}
      />
      <div className={ACTIONS_CELL}>
        <Button
          size="sm"
          variant="primary"
          aria-label={`Save ${field.key}`}
          isDisabled={disabled || !changed}
          onPress={() => onSave(trimmed === "" ? null : trimmed)}
        >
          Save
        </Button>
        <Button
          size="sm"
          variant="outline"
          aria-label={`Test ${field.service}`}
          isDisabled={trimmed === "" || test.isPending}
          onPress={() => test.mutate({ service: field.service, url: trimmed })}
        >
          {test.isPending ? "Testing…" : "Test"}
        </Button>
      </div>
      <div className={MESSAGE_CELL}>
        {/* aria-live so the reachability outcome is announced, not just shown. */}
        <span role="status" aria-live="polite" className="block break-words text-xs">
          {test.isPending ? null : test.error ? (
            <span className="text-red-700">{errorMessage(test.error)}</span>
          ) : test.data ? (
            <span className={test.data.ok ? "font-medium text-green-700" : "text-red-700"}>{test.data.reason}</span>
          ) : null}
        </span>
        <SaveError message={saveError} />
      </div>
    </div>
  );
}

// A nullable free-text row (engines, purpose hints). Empty clears to the default.
function TextRow({
  field,
  onSave,
  saveError,
  disabled,
}: {
  field: ToolSettingField;
  onSave: (value: string | null) => void;
  saveError?: string;
  disabled: boolean;
}) {
  const committed = typeof field.value === "string" ? field.value : "";
  const [draft, setDraft] = useState(committed);

  useEffect(() => {
    setDraft(committed);
  }, [committed]);

  const changed = draft !== committed;

  return (
    <div className={ROW_CLASS}>
      <FieldLabel field={field} />
      <input
        type="text"
        aria-label={field.key}
        value={draft}
        disabled={disabled}
        placeholder="default"
        onChange={(event) => setDraft(event.target.value)}
        className={INPUT_CELL}
      />
      <div className={ACTIONS_CELL}>
        <Button
          size="sm"
          variant="primary"
          aria-label={`Save ${field.key}`}
          isDisabled={disabled || !changed}
          onPress={() => onSave(draft.trim() === "" ? null : draft.trim())}
        >
          Save
        </Button>
      </div>
      {saveError ? (
        <div className={MESSAGE_CELL}>
          <SaveError message={saveError} />
        </div>
      ) : null}
    </div>
  );
}

// A nullable integer row (max_results). Empty clears to the backend default.
function NumberRow({
  field,
  onSave,
  saveError,
  disabled,
}: {
  field: ToolSettingField;
  onSave: (value: number | null) => void;
  saveError?: string;
  disabled: boolean;
}) {
  const committed = typeof field.value === "number" ? String(field.value) : "";
  const [draft, setDraft] = useState(committed);

  useEffect(() => {
    setDraft(committed);
  }, [committed]);

  const trimmed = draft.trim();
  const parsed = Number(trimmed);
  const valid = trimmed === "" || (Number.isInteger(parsed) && parsed >= 1);
  const changed = valid && trimmed !== committed;

  return (
    <div className={ROW_CLASS}>
      <FieldLabel field={field} help="Leave blank to use the backend default." />
      <input
        type="number"
        min="1"
        step="1"
        inputMode="numeric"
        aria-label={field.key}
        value={draft}
        disabled={disabled}
        placeholder="default"
        onChange={(event) => setDraft(event.target.value)}
        // Narrower than the text inputs but right-aligned in the same column, so
        // its right edge (and the Save button beside it) still lines up with them.
        className={`w-full text-right tabular-nums sm:col-start-2 sm:w-28 sm:justify-self-end ${INPUT_CLASS}`}
      />
      <div className={ACTIONS_CELL}>
        <Button
          size="sm"
          variant="primary"
          aria-label={`Save ${field.key}`}
          isDisabled={disabled || !changed}
          onPress={() => onSave(trimmed === "" ? null : parsed)}
        >
          Save
        </Button>
      </div>
      {saveError ? (
        <div className={MESSAGE_CELL}>
          <SaveError message={saveError} />
        </div>
      ) : null}
    </div>
  );
}

// A nullable boolean (web_search_extract) has three meaningful states: default
// (backend decides, currently on), on, or off. A tri-state select is honest
// about "default" in a way a two-state toggle cannot be.
function BoolRow({
  field,
  onSave,
  saveError,
  disabled,
}: {
  field: ToolSettingField;
  onSave: (value: boolean | null) => void;
  saveError?: string;
  disabled: boolean;
}) {
  // A tri-state select applies on change (like the toggles on the Settings page),
  // so a discrete choice needs no separate Save; text/number/url rows keep an
  // explicit Save because they have intermediate, typed-but-unsaved states.
  const current = field.value === true ? "on" : field.value === false ? "off" : "default";
  return (
    <div className={ROW_CLASS}>
      <FieldLabel field={field} />
      <div className="sm:col-start-2 sm:justify-self-start">
        <FilterSelect
          ariaLabel={field.key}
          value={current}
          onChange={(next) => onSave(next === "default" ? null : next === "on")}
          options={[
            { value: "default", label: "Default" },
            { value: "on", label: "On" },
            { value: "off", label: "Off" },
          ]}
          disabled={disabled}
        />
      </div>
      {saveError ? (
        <div className={MESSAGE_CELL}>
          <SaveError message={saveError} />
        </div>
      ) : null}
    </div>
  );
}

function ServiceRow({
  field,
  onSave,
  saveError,
  disabled,
}: {
  field: ToolSettingField;
  onSave: (value: boolean | number | string | null) => void;
  saveError?: string;
  disabled: boolean;
}) {
  if (field.type === "url") {
    return <UrlRow field={field} onSave={onSave} saveError={saveError} disabled={disabled} />;
  }
  if (field.type === "int") {
    return <NumberRow field={field} onSave={onSave} saveError={saveError} disabled={disabled} />;
  }
  if (field.type === "bool") {
    return <BoolRow field={field} onSave={onSave} saveError={saveError} disabled={disabled} />;
  }
  return <TextRow field={field} onSave={onSave} saveError={saveError} disabled={disabled} />;
}

export function ToolsGuardrailsPage() {
  const query = useToolSettings();
  const update = useUpdateToolSettings();
  const [toast, showToast] = useSaveToast();
  const [errors, setErrors] = useState<Record<string, string>>({});

  const data = query.data;
  const disabled = !data || update.isPending;

  const byKey = new Map((data?.fields ?? []).map((field) => [field.key, field]));

  const save = (field: ToolSettingField, value: boolean | number | string | null) => {
    setErrors((prev) => {
      const { [field.key]: _removed, ...rest } = prev;
      return rest;
    });
    update.mutate(oneField(field.key, value), {
      onSuccess: () => showToast(`${field.key} saved`),
      onError: (error) => setErrors((prev) => ({ ...prev, [field.key]: errorMessage(error) })),
    });
  };

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Tools & Guardrails"
        description="Configure the built-in tool and guardrail service endpoints without a restart. Changes apply immediately and persist. URLs are validated for shape (http/https) and can be tested for reachability before saving; the network-safety gates for these services live on the Settings page."
      />

      <ErrorBanner error={query.error} />

      {SERVICES.map((service) => {
        const ordered = service.order
          .map((key) => byKey.get(key))
          .filter((f): f is ToolSettingField => f !== undefined);
        // Any field the backend reports for this service that isn't in `order`
        // (e.g. a newly added key) still renders, after the ordered ones, so a
        // backend addition is visible without a frontend change.
        const extra = (data?.fields ?? []).filter(
          (f) => f.service === service.key && !service.order.includes(f.key),
        );
        const fields = [...ordered, ...extra];
        if (fields.length === 0) return null;
        return (
          <section key={service.key} className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-[var(--otari-ink)]">{service.label}</h2>
            <p className="text-sm text-[var(--otari-muted)]">{service.blurb}</p>
            <Card>
              <Card.Content className="flex flex-col divide-y divide-[var(--otari-line)] px-5 py-1">
                {fields.map((field) => (
                  <ServiceRow
                    key={field.key}
                    field={field}
                    onSave={(value) => save(field, value)}
                    saveError={errors[field.key]}
                    disabled={disabled}
                  />
                ))}
              </Card.Content>
            </Card>
          </section>
        );
      })}

      <SaveToast message={toast} />
    </div>
  );
}
