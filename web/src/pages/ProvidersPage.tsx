import {
  Button,
  Card,
  Chip,
  ComboBox,
  Description,
  Input,
  Label,
  ListBox,
  ListBoxItem,
  Spinner,
  TextArea,
  TextField,
} from "@heroui/react";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "@tanstack/react-router";

import {
  useCreateStoredProvider,
  useDeleteStoredProvider,
  useProviderCatalog,
  useProviderDetail,
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
} from "@/client";
import { Field } from "@/components/Field";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { ConfirmButton, ErrorBanner, errorMessage, InfoBanner, PageHeader } from "@/components/ui";
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

// client_args is whatever the provider's SDK client constructor takes (timeouts,
// custom headers), so it has no fixed schema and the form edits it as JSON. Blank
// means "none": the API reads an explicit null as "clear it".
type ClientArgsParse =
  | { ok: true; value: Record<string, unknown> | null }
  | { ok: false; error: string };

function parseClientArgs(text: string): ClientArgsParse {
  const raw = text.trim();
  if (raw === "") return { ok: true, value: null };
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { ok: false, error: "Not valid JSON." };
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    return { ok: false, error: 'Must be a JSON object, like {"timeout": 1800}.' };
  }
  return { ok: true, value: parsed as Record<string, unknown> };
}

// Render stored client_args back into the textarea, leaving it blank when there
// are none so an untouched form submits null rather than an empty object.
function formatClientArgs(args: Record<string, unknown> | null | undefined): string {
  return args && Object.keys(args).length > 0 ? JSON.stringify(args, null, 2) : "";
}

// The client_args editor. Options are passed straight to the provider client, so
// a bad value is rejected here rather than sent (issue #517).
function ClientArgsField({
  value,
  onChange,
  error,
}: {
  value: string;
  onChange: (next: string) => void;
  error: string | null;
}) {
  return (
    <TextField value={value} onChange={onChange} isInvalid={error !== null} className="flex max-w-md flex-col gap-1">
      <Label className="text-sm font-medium text-[var(--otari-ink)]">Client options (JSON)</Label>
      <TextArea rows={3} placeholder={'{"timeout": 1800}'} spellCheck={false} className="font-mono text-xs" />
      <Description className={error ? "text-xs text-red-700" : "text-xs text-[var(--otari-muted)]"}>
        {error ??
          // Unlike the API key, these are stored and returned unencrypted, so say
          // so before someone puts a token in a custom header here.
          "Passed to the provider's client, e.g. a request timeout in seconds or custom headers. Stored in plain text, so keep secrets out."}
      </Description>
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
          ) : test.data.discovery_unsupported ? (
            // No /v1/models on this backend: the test cannot confirm the key, but
            // it is not evidence the key is wrong either (issue #447). The error is
            // kept because this is the form where the operator just typed api_base,
            // and a wrong one 404s exactly like an absent listing endpoint.
            <span className="block max-w-md break-words text-xs text-amber-800">
              This provider does not list models, so the key could not be verified here. Save it and use the provider;
              declare its model ids under <code>models:</code> to have them show up in the catalogue. If you did not
              expect this, check the provider's reply below.
              {test.data.error ? (
                <span className="mt-0.5 block text-[var(--otari-muted)]">{test.data.error}</span>
              ) : null}
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
  const create = useCreateStoredProvider();
  const [providerId, setProviderId] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [apiBase, setApiBase] = useState("");
  const [name, setName] = useState("");
  const [clientArgsText, setClientArgsText] = useState("");
  const clientArgs = parseClientArgs(clientArgsText);

  // Autofill hints are fetched lazily for just the selected provider, so the
  // picker itself never imports every provider SDK (issue #365).
  const detail = useProviderDetail(providerId);
  const selected = detail.data?.id === providerId ? detail.data : undefined;
  // Prefill the (editable) API base with the provider's built-in default once its
  // detail loads, so Advanced shows what will be used. Keyed on the selected
  // provider so it fires once per selection and does not clobber later edits.
  useEffect(() => {
    if (selected) setApiBase(selected.default_api_base ?? "");
  }, [selected]);
  const envKeyPresent = selected?.env_key_present ?? false;
  // The key is only mandatory when the provider needs one and its env var is not
  // already set on the server; any-llm falls back to that env var otherwise.
  const needsKey = (selected?.requires_api_key ?? true) && !envKeyPresent;
  const renamed = name.trim() !== "" && name.trim() !== providerId;
  const nameHasDelimiter = /[:/]/.test(name);
  // Require the key when the chosen provider says it needs one; keyless local
  // backends (Ollama, llama.cpp) can submit without it.
  const canSubmit =
    providerId !== "" &&
    !nameHasDelimiter &&
    (!needsKey || apiKey.trim() !== "") &&
    clientArgs.ok &&
    !create.isPending;
  // Hold the section open while something inside it is what's blocking submit,
  // so collapsing it can't leave a disabled button with its reason off screen.
  // A hide requested meanwhile is remembered and applies once the field is fixed.
  const advancedOpen = showAdvanced || !clientArgs.ok || nameHasDelimiter;

  const submit = () => {
    // The clientArgs.ok half is already covered by canSubmit; it is repeated to
    // narrow the union so `.value` is reachable.
    if (!canSubmit || !clientArgs.ok) return;
    create.mutate(
      {
        instance: renamed ? name.trim() : providerId,
        // A renamed instance is no longer named after its provider, so record the
        // provider it is so routing still resolves.
        provider_type: renamed ? providerId : null,
        api_base: apiBase.trim() || null,
        api_key: apiKey.trim() || null,
        client_args: clientArgs.value,
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
          // Clear the API base; the effect above refills it from the provider's
          // built-in default once this provider's detail loads.
          setApiBase("");
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
        {advancedOpen ? "Hide advanced" : "Advanced (API base, rename, client options)"}
      </button>
      {advancedOpen ? (
        <div className="flex flex-col gap-4">
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
          <ClientArgsField
            value={clientArgsText}
            onChange={setClientArgsText}
            error={clientArgs.ok ? null : clientArgs.error}
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
            providerId === "" || !clientArgs.ok
              ? null
              : {
                  instance: renamed ? name.trim() : providerId,
                  provider_type: renamed ? providerId : null,
                  api_base: apiBase.trim() || null,
                  api_key: apiKey.trim() || null,
                  client_args: clientArgs.value,
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
  const [clientArgsText, setClientArgsText] = useState("");
  const clientArgs = parseClientArgs(clientArgsText);

  const nameHasDelimiter = /[:/]/.test(name);
  const canSubmit =
    name.trim() !== "" && !nameHasDelimiter && apiBase.trim() !== "" && clientArgs.ok && !create.isPending;

  const submit = () => {
    if (!canSubmit || !clientArgs.ok) return;
    create.mutate(
      {
        instance: name.trim(),
        provider_type: providerType || "openai-compatible",
        api_base: apiBase.trim(),
        api_key: apiKey.trim() || null,
        client_args: clientArgs.value,
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
      <ClientArgsField
        value={clientArgsText}
        onChange={setClientArgsText}
        error={clientArgs.ok ? null : clientArgs.error}
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
            name.trim() === "" || apiBase.trim() === "" || !clientArgs.ok
              ? null
              : {
                  instance: name.trim(),
                  provider_type: providerType || "openai-compatible",
                  api_base: apiBase.trim(),
                  api_key: apiKey.trim() || null,
                  client_args: clientArgs.value,
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

function EditProviderForm({
  provider,
  onClose,
  onSaved,
}: {
  provider: StoredProvider;
  onClose: () => void;
  // Called with the saved instance when a save succeeds, so the page can retire
  // anything that described the credentials as they were. Distinct from onClose,
  // which also fires on cancel, where nothing was written and an existing verdict
  // still holds.
  onSaved: (instance: string) => void;
}) {
  const update = useUpdateStoredProvider();
  const [providerType, setProviderType] = useState(provider.provider_type ?? "");
  const [apiBase, setApiBase] = useState(provider.api_base ?? "");
  const [replacingKey, setReplacingKey] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [clientArgsText, setClientArgsText] = useState(() => formatClientArgs(provider.client_args));
  const clientArgs = parseClientArgs(clientArgsText);

  const submit = () => {
    if (update.isPending || !clientArgs.ok) return;
    const body: UpdateStoredProviderRequest = {
      provider_type: providerType.trim() || null,
      api_base: apiBase.trim() || null,
      // Sent on every save, so emptying the field clears the stored options.
      client_args: clientArgs.value,
      // Guard against clobbering a concurrent edit; a 412 tells the operator to reload.
      expected_updated_at: provider.updated_at,
    };
    if (replacingKey && apiKey.trim()) {
      body.api_key = apiKey.trim();
    }
    update.mutate(
      { instance: provider.instance, body },
      {
        onSuccess: () => {
          onSaved(provider.instance);
          onClose();
        },
      },
    );
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
        <ClientArgsField
          value={clientArgsText}
          onChange={setClientArgsText}
          error={clientArgs.ok ? null : clientArgs.error}
        />
        <div className="flex gap-2">
          <Button variant="primary" isDisabled={update.isPending || !clientArgs.ok} onPress={submit}>
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
  // A backend with no model-listing endpoint cannot be verified this way, but the
  // key is not therefore wrong: say so instead of reporting a failed connection.
  // The provider error stays visible underneath, because a 404 is also what a
  // wrong api_base returns, and that is a misconfiguration to fix, not to reassure
  // away.
  if (state.discovery_unsupported) {
    return (
      <span className="block max-w-xs break-words text-xs text-amber-800">
        Could not list models, so the key could not be verified. It may still work for requests.
        {state.error ? <span className="mt-0.5 block text-[var(--otari-muted)]">{state.error}</span> : null}
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
// ErrorBanner rather than the --otari-* chrome. A provider that answers no model
// listing (its backend never implemented /v1/models) is not unreachable: only
// discovery is broken, and it may still serve requests, so it gets the amber
// warning state rather than the red one (issue #447).
function HealthPill({ health }: { health: ProviderHealth | undefined }) {
  if (!health) {
    return <span className="text-xs text-[var(--otari-muted)]">—</span>;
  }
  const degraded = !health.ok && health.discovery_unsupported;
  const styles = health.ok
    ? "border-green-200 bg-green-50 text-green-700"
    : degraded
      ? "border-amber-200 bg-amber-50 text-amber-800"
      : "border-red-200 bg-red-50 text-red-700";
  const dot = health.ok ? "bg-green-500" : degraded ? "bg-amber-500" : "bg-red-500";
  // The last-checked time lives in the top summary banner; the row just shows the
  // status. The error (and time) stay available on hover as the pill's tooltip.
  const checked = health.checked_at ? `Last checked ${formatRelative(health.checked_at)}` : "Not checked yet";
  const reason = degraded
    ? `${health.error ?? "This provider does not list models."} Requests to it may still work.`
    : (health.error ?? "Unreachable");
  const title = health.ok ? checked : `${reason} · ${checked}`;
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${styles}`}
    >
      <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {health.ok ? "Reachable" : degraded ? "No model discovery" : "Unreachable"}
    </span>
  );
}

// A one-line "N of M providers reachable" summary with a live re-check, above the
// table. The healthy/degraded/total counts come precomputed from the gateway, and
// the same counts feed the overview page's summary tile (issue #302). `degraded`
// providers are not reachable-by-discovery but are not failures either, so they
// are called out separately and keep the dot amber rather than red.
function HealthSummary({
  healthy,
  degraded,
  total,
  checkedAt,
}: {
  healthy: number;
  degraded: number;
  total: number;
  checkedAt: string | null;
}) {
  const allHealthy = healthy === total;
  const dot = allHealthy ? "bg-green-500" : healthy + degraded === total ? "bg-amber-500" : "bg-red-500";
  const recheck = useRecheckProviderHealth();
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-[var(--otari-line)] bg-[var(--otari-surface)] px-4 py-2.5 text-sm">
      <span aria-hidden className={`h-2 w-2 rounded-full ${dot}`} />
      <span className="font-medium text-[var(--otari-ink)]">
        {healthy} of {total} provider{total === 1 ? "" : "s"} reachable
      </span>
      {degraded > 0 ? <span className="text-amber-800">{degraded} without model discovery</span> : null}
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
  secretKeyConfigured,
}: {
  onAddProvider: () => void;
  needsPricing: boolean;
  onEnablePricing: () => void;
  enabling: boolean;
  secretKeyConfigured: boolean;
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
            {/* /welcome is served by the gateway itself, not by a client route, so this
                stays a plain path anchor: a router Link would resolve to /#/welcome, which
                the catch-all route sends back to the overview. It leaves the SPA, so open a
                new tab and the operator keeps the dashboard (as the guide's links do). */}
            <a
              href="/welcome"
              target="_blank"
              rel="noreferrer"
              className="font-medium text-[var(--otari-brand-dark)]"
            >
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
          <Button variant="primary" isDisabled={!secretKeyConfigured} onPress={onAddProvider}>
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
  // Gate adding providers on the server having OTARI_SECRET_KEY. Fail closed:
  // enabled only while settings are still loading (so the button does not
  // flicker to disabled on first paint) or once a value has actually loaded
  // that is not `false`. A settings *error* leaves us unable to confirm the
  // key, so we disable rather than let the operator hit a submit-time failure.
  // Older gateways omit the field; a present-but-missing value reads as
  // configured (they never gated on it).
  const secretKeyConfigured = settings.data
    ? settings.data.secret_key_configured !== false
    : !settings.isError;
  const showOnboarding = !loading && rows.length === 0 && !addOpen;

  // Which test run each row is currently showing. A row's result is only worth
  // recording while it is still the answer to the newest thing the operator asked
  // for, and neither the pending marker nor the Test button's disabled state can
  // establish that on its own: the button is per-row and keys off the marker
  // (below), so clearing the marker re-enables it in the same instant, and a
  // retest can start while the previous request is still in flight. A monotonic
  // id per instance is the thing that actually settles it. In a ref because a
  // late callback must read the current value, not the one its render closed over.
  const testRuns = useRef<Record<string, number>>({});
  const startTestRun = (instance: string) => {
    const run = (testRuns.current[instance] ?? 0) + 1;
    testRuns.current[instance] = run;
    return run;
  };

  // A test verdict describes the credentials as they were when it ran, so drop
  // it once the provider changes underneath it: otherwise the row keeps showing
  // a failure from the previous configuration, contradicting the status pill
  // right above it (issue #464). Bumping the run id retires any request still in
  // flight, so a slow test cannot write the old verdict back afterwards.
  const clearTest = (instance: string) => {
    startTestRun(instance);
    setTests((prev) => {
      // Own-property check: `in` also reports inherited keys, so an instance
      // named after one (`toString`) would read as having a verdict it does not.
      if (!Object.hasOwn(prev, instance)) return prev;
      const next = { ...prev };
      delete next[instance];
      return next;
    });
  };

  // Record a result only if its run is still the current one for that row, which
  // rules out both a verdict retired by an edit or delete and one superseded by a
  // later test on the same row.
  const settleTest = (instance: string, run: number, state: TestState) => {
    if (testRuns.current[instance] !== run) return;
    setTests((prev) => ({ ...prev, [instance]: state }));
  };

  // Resolve each row's test from its own promise. One useMutation observer serves
  // every row, and TanStack Query detaches it from the previous mutation as soon
  // as the next `mutate` lands, discarding that call's onSuccess/onError: testing
  // a second provider while the first was still in flight left the first row
  // spinning on "Testing…", with its Test button disabled, for the life of the
  // page. The promise from mutateAsync settles regardless of that detach.
  const runTest = async (instance: string) => {
    const run = startTestRun(instance);
    setTests((prev) => ({ ...prev, [instance]: { status: "pending" } }));
    try {
      const result = await testProvider.mutateAsync(instance);
      settleTest(instance, run, { status: "done", ...result });
    } catch (error) {
      settleTest(instance, run, {
        status: "done",
        ok: false,
        model_count: 0,
        error: errorMessage(error),
        discovery_unsupported: false,
      });
    }
  };

  const columns: DataTableColumn<ProviderRow>[] = [
    {
      id: "provider",
      header: "Provider",
      isRowHeader: true,
      cell: (row) => (
        <Link
          to="/models"
          search={{ provider: row.instance }}
          className="font-medium text-[var(--otari-ink)] hover:text-[var(--otari-brand-dark)] hover:underline"
        >
          {row.instance}
        </Link>
      ),
    },
    {
      id: "type",
      header: "Type",
      cell: (row) => (
        <span className="text-[var(--otari-muted)]">
          {row.meta?.provider_type ?? row.stored?.provider_type ?? row.instance}
        </span>
      ),
    },
    {
      id: "source",
      header: "Source",
      cell: (row) => (
        <Chip size="sm" color={row.source === "stored" ? "accent" : "default"}>
          {row.source === "stored" ? "stored" : "config"}
        </Chip>
      ),
    },
    {
      id: "api_key",
      header: "API key",
      cell: (row) => (
        <span className="text-[var(--otari-muted)]">
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
        </span>
      ),
    },
    { id: "status", header: "Status", cell: (row) => <HealthPill health={healthByInstance.get(row.instance)} /> },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) =>
        row.source === "stored" ? (
          <div className="flex flex-col items-end gap-1.5">
            <div className="flex items-center gap-1.5">
              <Button
                size="sm"
                variant="outline"
                // A row whose key can't be decrypted can't be tested; Edit/Delete still recover it.
                isDisabled={tests[row.instance]?.status === "pending" || row.stored?.decryptable === false}
                onPress={() => void runTest(row.instance)}
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
                // Clear the verdict too: a provider re-added under the same name
                // is a different provider, and would otherwise inherit it.
                onConfirm={() => deleteProvider.mutate(row.instance, { onSuccess: () => clearTest(row.instance) })}
              >
                Delete
              </ConfirmButton>
            </div>
            <TestOutcome state={tests[row.instance]} />
          </div>
        ) : (
          <span className="block text-right text-xs text-[var(--otari-muted)]">managed in config.yml</span>
        ),
    },
  ];

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
              isDisabled={!secretKeyConfigured}
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

      {!secretKeyConfigured ? (
        <InfoBanner tone="warning">
          <code>OTARI_SECRET_KEY</code> is not set, so provider keys can't be encrypted at rest and adding providers
          from the dashboard is disabled. Set it on the server and restart to add providers here. Providers defined in{" "}
          <code>config.yml</code> keep working without it.
        </InfoBanner>
      ) : null}

      {showOnboarding ? (
        <OnboardingPanel
          onAddProvider={() => {
            setEditing(null);
            setAddOpen(true);
          }}
          needsPricing={needsPricing}
          onEnablePricing={() => updateSettings.mutate({ default_pricing: true })}
          enabling={updateSettings.isPending}
          secretKeyConfigured={secretKeyConfigured}
        />
      ) : null}
      {/* The gateway-wide "requests are rejected until pricing is set" alarm now
          lives in the app shell (PricingWarning), so it shows on every page, not
          only here. The first-run onboarding tip above stays as onboarding guidance. */}

      {/* Also gate the form itself on the flag, not just the buttons that open it:
          if it was opened while settings were still loading and the key then turns
          out to be unavailable, retract it so its submit can never reach the create
          mutation. The banner above explains why. */}
      {addOpen && secretKeyConfigured ? <AddProviderForm onClose={() => setAddOpen(false)} /> : null}
      {editingProvider ? (
        <EditProviderForm
          // Remount when the operator switches rows: the fields are seeded from
          // the provider once, so without this, editing a second provider would
          // open with the first one's values (and save them onto it).
          key={editingProvider.instance}
          provider={editingProvider}
          onClose={() => setEditing(null)}
          onSaved={clearTest}
        />
      ) : null}

      {!loading && rows.length > 0 && health.data && health.data.total > 0 ? (
        <HealthSummary
          healthy={health.data.healthy}
          degraded={health.data.degraded}
          total={health.data.total}
          checkedAt={health.data.checked_at ?? null}
        />
      ) : null}

      {/* Suppress the table (and its own empty message) while the onboarding
          panel owns the empty state, so a fresh gateway shows one call to action,
          not a panel stacked over a redundant "no rows" table. */}
      {showOnboarding ? null : (
        <DataTable
          ariaLabel="Providers"
          columns={columns}
          rows={rows}
          getRowKey={(row) => row.instance}
          isLoading={loading}
          emptyContent="No providers yet. Add your first provider to start serving models."
        />
      )}
    </div>
  );
}
