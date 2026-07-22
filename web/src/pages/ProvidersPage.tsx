import { Button, Card, Chip, ComboBox, Description, Input, Label, ListBox, ListBoxItem, Spinner, TextField } from "@heroui/react";
import { type ReactNode, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import {
  useCreateStoredProvider,
  useDeleteStoredProvider,
  useProviderCatalog,
  useProviderHealth,
  useProviders,
  useRecheckProviderHealth,
  useSettings,
  useStoredProviders,
  useTestProviderCredentials,
  useTestStoredProvider,
  useUpdateSettings,
  useUpdateStoredProvider,
} from "@/api/hooks";
import type {
  CreateStoredProviderRequest,
  ProviderHealth,
  ProviderInfo,
  StoredProvider,
  TestProviderResult,
  UpdateStoredProviderRequest,
} from "@/api/types";
import { Field } from "@/components/Field";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ConfirmButton, ErrorBanner, errorMessage, PageHeader } from "@/components/ui";
import { formatRelative } from "@/lib/format";

// A masked, never-prefilled secret input. Native password masking protects
// Firefox users; self-hosted deployments should use HTTPS to avoid browser warnings.
function SecretField({
  value,
  onChange,
  label,
  placeholder,
  description,
}: {
  value: string;
  onChange: (next: string) => void;
  label: string;
  placeholder?: string;
  description?: string;
}) {
  return (
    <TextField value={value} onChange={onChange} className="flex max-w-md flex-col gap-1">
      <Label className="text-sm font-medium text-[var(--otari-ink)]">{label}</Label>
      <Input
        type="password"
        placeholder={placeholder ?? "sk-…"}
        autoComplete="off"
        autoCorrect="off"
        autoCapitalize="off"
        spellCheck={false}
        data-1p-ignore
        data-lpignore="true"
      />
      {description ? <Description className="text-xs text-[var(--otari-muted)]">{description}</Description> : null}
    </TextField>
  );
}

// A searchable provider picker over the known-provider catalog. Selection sets
// an id (provider id, or a provider_type) while the input shows the display
// name. `extra` prepends synthetic options like "OpenAI-compatible".
function ProviderComboBox({
  label,
  value,
  onChange,
  description,
  placeholder,
  extra = [],
  includeCatalog = true,
}: {
  label: string;
  value: string;
  onChange: (id: string) => void;
  description?: ReactNode;
  placeholder?: string;
  extra?: { id: string; name: string }[];
  // When false, offer only `extra` (e.g. the two API dialects), not the full
  // provider catalog.
  includeCatalog?: boolean;
}) {
  const catalog = useProviderCatalog();
  const options = useMemo(
    () => (includeCatalog ? [...extra, ...(catalog.data ?? []).map((p) => ({ id: p.id, name: p.name }))] : extra),
    [catalog.data, extra, includeCatalog],
  );

  // Seed the input with the selected option's display name. The field owns its
  // text after mount (updated on typing and on selection); syncing it back from
  // `value` on every render would wipe out what the user is typing, since the
  // options array is recreated each render.
  const [text, setText] = useState(() => options.find((o) => o.id === value)?.name ?? "");

  // When the input merely shows the current selection, treat the query as empty
  // so opening the dropdown reveals every option, not just the selected one.
  const selectedName = options.find((o) => o.id === value)?.name ?? "";
  const query = text.trim() === selectedName.trim() ? "" : text.trim().toLowerCase();
  const visible = options
    .filter((o) => !query || o.name.toLowerCase().includes(query) || o.id.toLowerCase().includes(query))
    .slice(0, 50);

  return (
    <ComboBox.Root
      allowsEmptyCollection
      // Open the full list on focus/click and filter as you type: this is a
      // pick-from-a-list control, not a free-text field, and it is not
      // autofocused, so the list does not spring open when the form appears.
      menuTrigger="focus"
      inputValue={text}
      onInputChange={setText}
      onSelectionChange={(key) => {
        if (key != null) {
          onChange(String(key));
          setText(options.find((o) => o.id === String(key))?.name ?? "");
        } else {
          // Selection cleared: clear the parent value too, so the submitted
          // data cannot keep a stale provider after the field is emptied.
          onChange("");
          setText("");
        }
      }}
      className="flex max-w-md flex-col gap-1"
    >
      <Label className="text-sm font-medium text-[var(--otari-ink)]">{label}</Label>
      <ComboBox.InputGroup>
        {/* Not a credential field: keep browser password managers from offering to fill it.
            Select the text on focus so typing replaces the current selection instead of
            appending to it (otherwise "OpenAI-compatible" + typing filters to nothing). */}
        <Input
          placeholder={placeholder ?? "Search providers…"}
          autoComplete="off"
          data-1p-ignore
          data-lpignore="true"
          onFocus={(event) => event.currentTarget.select()}
        />
        <ComboBox.Trigger />
      </ComboBox.InputGroup>
      <ComboBox.Popover>
        <ListBox items={visible} className="max-h-72 overflow-auto">
          {(option: { id: string; name: string }) => (
            <ListBoxItem id={option.id} textValue={option.name}>
              {option.name}
            </ListBoxItem>
          )}
        </ListBox>
      </ComboBox.Popover>
      {description ? <span className="text-xs text-[var(--otari-muted)]">{description}</span> : null}
    </ComboBox.Root>
  );
}

// A "Test connection" button + inline result, testing the form's credentials
// before they are saved. `getPayload` returns null when the minimum fields for a
// test are not filled in yet, which disables the button.
function ConnectionTest({ getPayload }: { getPayload: () => CreateStoredProviderRequest | null }) {
  const test = useTestProviderCredentials();
  const payload = getPayload();

  return (
    <div className="flex flex-col gap-1.5">
      <Button
        variant="outline"
        isDisabled={payload === null || test.isPending}
        onPress={() => {
          if (payload) test.mutate(payload);
        }}
      >
        {test.isPending ? "Testing…" : "Test connection"}
      </Button>
      {/* aria-live so the connection outcome is announced to assistive tech. */}
      <span role="status" aria-live="polite">
        {test.isPending ? null : test.error ? (
          <span className="text-xs text-red-700">{errorMessage(test.error)}</span>
        ) : test.data ? (
          test.data.ok ? (
            <span className="text-xs font-medium text-green-700">
              Connected. {test.data.model_count} model{test.data.model_count === 1 ? "" : "s"} available.
            </span>
          ) : (
            <span className="block max-w-md break-words text-xs text-red-700">
              {test.data.error ?? "Connection failed."}
            </span>
          )
        ) : null}
      </span>
    </div>
  );
}

// Add a hosted provider whose endpoint is built into the SDK: pick it, paste a
// key. Name and api_base are only exposed under Advanced.
function KnownProviderForm({ onClose }: { onClose: () => void }) {
  const catalog = useProviderCatalog();
  const create = useCreateStoredProvider();
  const [providerId, setProviderId] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [apiBase, setApiBase] = useState("");
  const [name, setName] = useState("");

  const selected = catalog.data?.find((p) => p.id === providerId);
  const envKeyPresent = selected?.env_key_present ?? false;
  // The key is only mandatory when the provider needs one and its env var is not
  // already set on the server; any-llm falls back to that env var otherwise.
  const needsKey = (selected?.requires_api_key ?? true) && !envKeyPresent;
  const renamed = name.trim() !== "" && name.trim() !== providerId;
  const nameHasDelimiter = /[:/]/.test(name);
  // Require the key when the chosen provider says it needs one; keyless local
  // backends (Ollama, llama.cpp) can submit without it.
  const canSubmit =
    providerId !== "" && !nameHasDelimiter && (!needsKey || apiKey.trim() !== "") && !create.isPending;

  const submit = () => {
    if (!canSubmit) return;
    create.mutate(
      {
        instance: renamed ? name.trim() : providerId,
        // A renamed instance is no longer named after its provider, so record the
        // provider it is so routing still resolves.
        provider_type: renamed ? providerId : null,
        api_base: apiBase.trim() || null,
        api_key: apiKey.trim() || null,
      },
      { onSuccess: onClose },
    );
  };

  return (
    <div className="flex flex-col gap-4">
      <ErrorBanner error={create.error} />
      <ProviderComboBox
        label="Provider"
        value={providerId}
        onChange={(id) => {
          setProviderId(id);
          setName("");
          // Prefill the (editable) API base with the provider's built-in default
          // so Advanced shows what will be used; empty when the SDK has no explicit one.
          setApiBase(catalog.data?.find((p) => p.id === id)?.default_api_base ?? "");
        }}
        description="Its endpoint is built in."
      />
      <SecretField
        value={apiKey}
        onChange={setApiKey}
        label={selected && !needsKey ? "API key (optional)" : "API key"}
        description={
          selected
            ? needsKey
              ? `${selected.name}'s endpoint is built in — just add your key.`
              : envKeyPresent
                ? `${selected.env_key} is set on the server, so a key is optional here. Paste one to override it.`
                : `${selected.name} needs no API key.`
            : "Stored encrypted. Requires OTARI_SECRET_KEY on the server."
        }
      />
      <button
        type="button"
        className="self-start text-xs font-medium text-[var(--otari-brand-dark)]"
        onClick={() => setShowAdvanced((v) => !v)}
      >
        {showAdvanced ? "Hide advanced" : "Advanced (API base, rename)"}
      </button>
      {showAdvanced ? (
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="API base"
            value={apiBase}
            onChange={setApiBase}
            placeholder={selected?.default_api_base ?? "https://…/v1"}
            description="Only if you route through a proxy. Blank uses the built-in default."
          />
          <Field
            label="Name"
            value={name}
            onChange={setName}
            placeholder={providerId || "instance name"}
            description={
              nameHasDelimiter ? (
                <span className="text-red-700">A name cannot contain “:” or “/”.</span>
              ) : (
                "Rename to run two instances of the same provider."
              )
            }
          />
        </div>
      ) : null}
      <div className="flex flex-wrap items-start gap-2">
        <Button variant="primary" isDisabled={!canSubmit} onPress={submit}>
          {create.isPending ? "Adding…" : "Add provider"}
        </Button>
        <Button variant="ghost" onPress={onClose}>
          Cancel
        </Button>
        <ConnectionTest
          getPayload={() =>
            providerId === ""
              ? null
              : {
                  instance: renamed ? name.trim() : providerId,
                  provider_type: renamed ? providerId : null,
                  api_base: apiBase.trim() || null,
                  api_key: apiKey.trim() || null,
                }
          }
        />
      </div>
    </div>
  );
}

// Add a self-hosted or OpenAI-compatible endpoint: name it anything, say what
// API it speaks, and give the base URL (and a key if it needs one).
function CustomProviderForm({ onClose }: { onClose: () => void }) {
  const create = useCreateStoredProvider();
  const [name, setName] = useState("");
  const [providerType, setProviderType] = useState("openai-compatible");
  const [apiBase, setApiBase] = useState("");
  const [apiKey, setApiKey] = useState("");

  const nameHasDelimiter = /[:/]/.test(name);
  const canSubmit = name.trim() !== "" && !nameHasDelimiter && apiBase.trim() !== "" && !create.isPending;

  const submit = () => {
    if (!canSubmit) return;
    create.mutate(
      {
        instance: name.trim(),
        provider_type: providerType || "openai-compatible",
        api_base: apiBase.trim(),
        api_key: apiKey.trim() || null,
      },
      { onSuccess: onClose },
    );
  };

  return (
    <div className="flex flex-col gap-4">
      <ErrorBanner error={create.error} />
      <div className="grid gap-4 sm:grid-cols-2">
        <Field
          label="Name"
          value={name}
          onChange={setName}
          placeholder="my-local-llm"
          isRequired
          autoFocus
          description={
            nameHasDelimiter ? (
              <span className="text-red-700">A name cannot contain “:” or “/”.</span>
            ) : (
              "Call it whatever you want."
            )
          }
        />
        <ProviderComboBox
          label="Compatible with"
          value={providerType}
          onChange={setProviderType}
          includeCatalog={false}
          description="The API this endpoint speaks."
          extra={[
            { id: "openai-compatible", name: "OpenAI" },
            { id: "anthropic-compatible", name: "Anthropic" },
          ]}
        />
      </div>
      <Field
        label="API base"
        value={apiBase}
        onChange={setApiBase}
        placeholder="http://localhost:8000/v1"
        isRequired
        description="The endpoint URL of your server."
      />
      <SecretField
        value={apiKey}
        onChange={setApiKey}
        label="API key (optional)"
        description="Many local backends need none. Stored encrypted."
      />
      <div className="flex flex-wrap items-start gap-2">
        <Button variant="primary" isDisabled={!canSubmit} onPress={submit}>
          {create.isPending ? "Adding…" : "Add provider"}
        </Button>
        <Button variant="ghost" onPress={onClose}>
          Cancel
        </Button>
        <ConnectionTest
          getPayload={() =>
            name.trim() === "" || apiBase.trim() === ""
              ? null
              : {
                  instance: name.trim(),
                  provider_type: providerType || "openai-compatible",
                  api_base: apiBase.trim(),
                  api_key: apiKey.trim() || null,
                }
          }
        />
      </div>
    </div>
  );
}

type ProviderTab = "known" | "custom";

function AddProviderForm({ onClose }: { onClose: () => void }) {
  const [tab, setTab] = useState<ProviderTab>("known");

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1 rounded-lg bg-[var(--otari-bg)] p-1">
            {(
              [
                ["known", "Known provider"],
                ["custom", "Custom endpoint"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                aria-pressed={tab === id}
                onClick={() => setTab(id)}
                className={
                  tab === id
                    ? "rounded-md bg-white px-3 py-1.5 text-sm font-medium text-[var(--otari-ink)] shadow-sm"
                    : "rounded-md px-3 py-1.5 text-sm text-[var(--otari-muted)] hover:text-[var(--otari-ink)]"
                }
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        {tab === "known" ? <KnownProviderForm onClose={onClose} /> : <CustomProviderForm onClose={onClose} />}
      </Card.Content>
    </Card>
  );
}

function EditProviderForm({ provider, onClose }: { provider: StoredProvider; onClose: () => void }) {
  const update = useUpdateStoredProvider();
  const [providerType, setProviderType] = useState(provider.provider_type ?? "");
  const [apiBase, setApiBase] = useState(provider.api_base ?? "");
  const [replacingKey, setReplacingKey] = useState(false);
  const [apiKey, setApiKey] = useState("");

  const submit = () => {
    if (update.isPending) return;
    const body: UpdateStoredProviderRequest = {
      provider_type: providerType.trim() || null,
      api_base: apiBase.trim() || null,
      // Guard against clobbering a concurrent edit; a 412 tells the operator to reload.
      expected_updated_at: provider.updated_at,
    };
    if (replacingKey && apiKey.trim()) {
      body.api_key = apiKey.trim();
    }
    update.mutate({ instance: provider.instance, body }, { onSuccess: onClose });
  };

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-[var(--otari-ink)]">
          Edit <code>{provider.instance}</code>
        </div>
        <ErrorBanner error={update.error} />
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Provider type" value={providerType} onChange={setProviderType} placeholder="openai" />
          <Field label="API base" value={apiBase} onChange={setApiBase} placeholder="https://api.openai.com/v1" />
        </div>
        <div className="flex flex-col gap-2">
          {replacingKey ? (
            <>
              <SecretField
                value={apiKey}
                onChange={setApiKey}
                label="New API key"
                description="Stored encrypted. The old key is replaced when you save."
              />
              <button
                type="button"
                className="self-start text-xs font-medium text-[var(--otari-brand-dark)]"
                onClick={() => {
                  setReplacingKey(false);
                  setApiKey("");
                }}
              >
                Keep the current key
              </button>
            </>
          ) : (
            <div className="flex items-center gap-3">
              <span className="text-sm text-[var(--otari-muted)]">
                API key: <code>{provider.last4 ? `••••${provider.last4}` : "none set"}</code>
              </span>
              <Button size="sm" variant="outline" onPress={() => setReplacingKey(true)}>
                Replace key
              </Button>
            </div>
          )}
        </div>
        <div className="flex gap-2">
          <Button variant="primary" isDisabled={update.isPending} onPress={submit}>
            {update.isPending ? "Saving…" : "Save changes"}
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

interface ProviderRow {
  instance: string;
  source: "config" | "stored";
  stored: StoredProvider | undefined;
  meta: ProviderInfo | undefined;
}

function buildRows(meta: ProviderInfo[] | undefined, stored: StoredProvider[] | undefined): ProviderRow[] {
  const storedByInstance = new Map((stored ?? []).map((p) => [p.instance, p]));
  const metaByInstance = new Map((meta ?? []).map((p) => [p.instance, p]));
  const instances = new Set<string>([...storedByInstance.keys(), ...metaByInstance.keys()]);
  return [...instances]
    .sort()
    .map((instance) => {
      const s = storedByInstance.get(instance);
      return { instance, source: s ? "stored" : "config", stored: s, meta: metaByInstance.get(instance) } as const;
    });
}

type TestState = { status: "pending" } | ({ status: "done" } & TestProviderResult);

function TestOutcome({ state }: { state: TestState | undefined }) {
  if (!state) return null;
  if (state.status === "pending") {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-[var(--otari-muted)]">
        <Spinner size="sm" /> Testing…
      </span>
    );
  }
  if (state.ok) {
    return (
      <span className="text-xs font-medium text-green-700">
        Connected. {state.model_count} model{state.model_count === 1 ? "" : "s"} available.
      </span>
    );
  }
  return (
    <span className="block max-w-xs break-words text-xs text-red-700">
      {state.error ?? "Connection failed."}
    </span>
  );
}

// A provider's reachability, from the shared model-discovery health path. Config
// providers (no per-row Test button) get a status here too, not just stored ones.
// Semantic status surface: raw Tailwind palette classes, matching TestOutcome and
// ErrorBanner rather than the --otari-* chrome.
function HealthPill({ health }: { health: ProviderHealth | undefined }) {
  if (!health) {
    return <span className="text-xs text-[var(--otari-muted)]">—</span>;
  }
  const styles = health.ok
    ? "border-green-200 bg-green-50 text-green-700"
    : "border-red-200 bg-red-50 text-red-700";
  // The last-checked time lives in the top summary banner; the row just shows the
  // status. The error (and time) stay available on hover as the pill's tooltip.
  const checked = health.checked_at ? `Last checked ${formatRelative(health.checked_at)}` : "Not checked yet";
  const title = health.ok ? checked : `${health.error ?? "Unreachable"} · ${checked}`;
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${styles}`}
    >
      <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${health.ok ? "bg-green-500" : "bg-red-500"}`} />
      {health.ok ? "Reachable" : "Unreachable"}
    </span>
  );
}

// A one-line "N of M providers reachable" summary with a live re-check, above the
// table. The healthy/total counts come precomputed from the gateway, and the
// same counts feed the overview page's summary tile (issue #302).
function HealthSummary({ healthy, total, checkedAt }: { healthy: number; total: number; checkedAt: string | null }) {
  const allHealthy = healthy === total;
  const recheck = useRecheckProviderHealth();
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-[var(--otari-line)] bg-[var(--otari-surface)] px-4 py-2.5 text-sm">
      <span aria-hidden className={`h-2 w-2 rounded-full ${allHealthy ? "bg-green-500" : "bg-red-500"}`} />
      <span className="font-medium text-[var(--otari-ink)]">
        {healthy} of {total} provider{total === 1 ? "" : "s"} reachable
      </span>
      {checkedAt ? (
        <span className="text-[var(--otari-muted)]">Last checked {formatRelative(checkedAt)}</span>
      ) : null}
      <Button
        size="sm"
        variant="ghost"
        className="ml-auto"
        isDisabled={recheck.isPending}
        onPress={() => recheck.mutate()}
      >
        {recheck.isPending ? "Re-checking…" : "Re-check all"}
      </Button>
    </div>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[var(--otari-brand-tint)] text-xs font-semibold text-[var(--otari-brand-dark)]">
        {n}
      </span>
      <div className="text-sm">
        <div className="font-medium text-[var(--otari-ink)]">{title}</div>
        <div className="text-[var(--otari-muted)]">{children}</div>
      </div>
    </li>
  );
}

// Shown on first run (no provider configured yet). It disappears the moment a
// provider exists, so it is a nudge to the first key, not a permanent banner.
function OnboardingPanel({
  onAddProvider,
  needsPricing,
  onEnablePricing,
  enabling,
}: {
  onAddProvider: () => void;
  needsPricing: boolean;
  onEnablePricing: () => void;
  enabling: boolean;
}) {
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-6">
        <div>
          <h2 className="text-lg font-semibold text-[var(--otari-ink)]">Welcome to Otari</h2>
          <p className="mt-1 text-sm text-[var(--otari-muted)]">
            You are signed in. Add a provider to start serving models: three quick steps.
          </p>
        </div>
        <ol className="flex flex-col gap-3">
          <Step n={1} title="Add a provider">
            Enter a provider name (like <code>openai</code>) and its API key. Keys are encrypted at rest.
          </Step>
          <Step n={2} title="Test the connection">
            Use <strong>Test</strong> on the provider row to confirm the key works and see how many models it serves.
          </Step>
          <Step n={3} title="Send your first request">
            Point your app at <code>/v1</code> on this gateway with the API key printed in the server logs
            (<code>gw-…</code>). See the{" "}
            <a href="/welcome" className="font-medium text-[var(--otari-brand-dark)]">
              quickstart
            </a>
            .
          </Step>
        </ol>
        {needsPricing ? (
          <p className="text-sm text-[var(--otari-muted)]">
            Tip: <code>require_pricing</code> is on, so requests are rejected until pricing is set.{" "}
            <button
              type="button"
              className="font-medium text-[var(--otari-brand-dark)] disabled:opacity-50"
              disabled={enabling}
              onClick={onEnablePricing}
            >
              Enable default pricing
            </button>{" "}
            to meter new models with public rates.
          </p>
        ) : null}
        <div>
          <Button variant="primary" onPress={onAddProvider}>
            Add your first provider
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

export function ProvidersPage() {
  const meta = useProviders();
  const stored = useStoredProviders();
  const settings = useSettings();
  const health = useProviderHealth();
  const deleteProvider = useDeleteStoredProvider();
  const testProvider = useTestStoredProvider();
  const updateSettings = useUpdateSettings();

  const [addOpen, setAddOpen] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [tests, setTests] = useState<Record<string, TestState>>({});

  const rows = buildRows(meta.data?.providers, stored.data);
  const healthByInstance = new Map((health.data?.providers ?? []).map((item) => [item.instance, item]));
  const loading = meta.isLoading || stored.isLoading;
  const editingProvider = stored.data?.find((p) => p.instance === editing) ?? null;
  const needsPricing = settings.data?.require_pricing === true && settings.data.default_pricing === false;
  const showOnboarding = !loading && rows.length === 0 && !addOpen;

  const runTest = (instance: string) => {
    setTests((prev) => ({ ...prev, [instance]: { status: "pending" } }));
    testProvider.mutate(instance, {
      onSuccess: (result) => setTests((prev) => ({ ...prev, [instance]: { status: "done", ...result } })),
      onError: (error) =>
        setTests((prev) => ({
          ...prev,
          [instance]: { status: "done", ok: false, model_count: 0, error: errorMessage(error) },
        })),
    });
  };

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Providers"
        description="Add provider API keys here to serve models without editing config.yml. Keys are encrypted at rest."
        action={
          // The first-run card supplies its own focused call to action. The form
          // card has its own Close, so the header action is redundant while open.
          addOpen || showOnboarding ? null : (
            <Button
              variant="primary"
              onPress={() => {
                setEditing(null);
                setAddOpen(true);
              }}
            >
              Add provider
            </Button>
          )
        }
      />

      <ErrorBanner
        error={
          meta.error ?? stored.error ?? settings.error ?? health.error ?? updateSettings.error ?? deleteProvider.error
        }
      />

      {showOnboarding ? (
        <OnboardingPanel
          onAddProvider={() => {
            setEditing(null);
            setAddOpen(true);
          }}
          needsPricing={needsPricing}
          onEnablePricing={() => updateSettings.mutate({ default_pricing: true })}
          enabling={updateSettings.isPending}
        />
      ) : null}
      {/* The gateway-wide "requests are rejected until pricing is set" alarm now
          lives in the app shell (PricingWarning), so it shows on every page, not
          only here. The first-run onboarding tip above stays as onboarding guidance. */}

      {addOpen ? <AddProviderForm onClose={() => setAddOpen(false)} /> : null}
      {editingProvider ? <EditProviderForm provider={editingProvider} onClose={() => setEditing(null)} /> : null}

      {!loading && rows.length > 0 && health.data && health.data.total > 0 ? (
        <HealthSummary healthy={health.data.healthy} total={health.data.total} checkedAt={health.data.checked_at} />
      ) : null}

      <Table>
        <THead>
          <tr>
            <Th>Provider</Th>
            <Th>Type</Th>
            <Th>Source</Th>
            <Th>API key</Th>
            <Th>Status</Th>
            <Th className="text-right">Actions</Th>
          </tr>
        </THead>
        <tbody>
          {loading ? (
            <LoadingRow colSpan={6} />
          ) : rows.length === 0 ? (
            <TableMessage colSpan={6}>No providers yet. Add your first provider to start serving models.</TableMessage>
          ) : (
            rows.map((row) => (
              <Tr key={row.instance}>
                <Td className="font-medium">
                  <Link
                    to={`/models?provider=${encodeURIComponent(row.instance)}`}
                    className="text-[var(--otari-ink)] hover:text-[var(--otari-brand-dark)] hover:underline"
                  >
                    {row.instance}
                  </Link>
                </Td>
                <Td className="text-[var(--otari-muted)]">
                  {row.meta?.provider_type ?? row.stored?.provider_type ?? row.instance}
                </Td>
                <Td>
                  {row.source === "stored" ? (
                    <Chip size="sm" color="accent">
                      stored
                    </Chip>
                  ) : (
                    <Chip size="sm" color="default">
                      config
                    </Chip>
                  )}
                </Td>
                <Td className="text-[var(--otari-muted)]">
                  {row.source === "stored" ? (
                    row.stored && !row.stored.decryptable ? (
                      <span
                        className="text-amber-700"
                        title="This key can't be decrypted with the current OTARI_SECRET_KEY. Replace the key, or restore the original OTARI_SECRET_KEY."
                      >
                        ⚠ key unreadable
                      </span>
                    ) : (
                      <code>{row.stored?.last4 ? `••••${row.stored.last4}` : "none set"}</code>
                    )
                  ) : row.meta?.env_key ? (
                    <span>
                      via <code>{row.meta.env_key}</code>
                    </span>
                  ) : (
                    "config.yml"
                  )}
                </Td>
                <Td>
                  <HealthPill health={healthByInstance.get(row.instance)} />
                </Td>
                <Td>
                  {row.source === "stored" ? (
                    <div className="flex flex-col items-end gap-1.5">
                      <div className="flex items-center gap-1.5">
                        <Button
                          size="sm"
                          variant="outline"
                          // A row whose key can't be decrypted can't be tested; Edit/Delete still recover it.
                          isDisabled={
                            tests[row.instance]?.status === "pending" || row.stored?.decryptable === false
                          }
                          onPress={() => runTest(row.instance)}
                        >
                          Test
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onPress={() => {
                            setAddOpen(false);
                            setEditing(row.instance);
                          }}
                        >
                          Edit
                        </Button>
                        <ConfirmButton
                          confirmLabel="Delete"
                          isPending={deleteProvider.isPending}
                          onConfirm={() => deleteProvider.mutate(row.instance)}
                        >
                          Delete
                        </ConfirmButton>
                      </div>
                      <TestOutcome state={tests[row.instance]} />
                    </div>
                  ) : (
                    <span className="block text-right text-xs text-[var(--otari-muted)]">managed in config.yml</span>
                  )}
                </Td>
              </Tr>
            ))
          )}
        </tbody>
      </Table>
    </div>
  );
}
