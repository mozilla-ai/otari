import { Button, Card } from "@heroui/react";
import { useEffect, useState } from "react";

import {
  useCreateSearchTool,
  useDeleteSearchTool,
  useSearchProviders,
  useSearchTools,
  useUpdateSearchTool,
} from "@/shared/api/hooks";
import type { ConfigSearchTool, SearchProviderInfo, StoredSearchTool } from "@/client";
import { ConfirmButton, ErrorBanner, FilterSelect, errorMessage } from "@/shared/components/ui";

// Search tools are what POST /v1/search dispatches against. They used to be
// declarable only in a config file, so a deployment configured entirely through
// the dashboard could not use that endpoint at all. This card is the route in:
// stored tools are editable here, config-file tools are shown read-only so the
// operator can see every tool a caller could name.

const INPUT_CLASS =
  "rounded-md border border-[var(--otari-line)] bg-[var(--otari-surface)] px-2 py-1 text-sm focus:border-[var(--otari-brand)] focus:outline-none disabled:opacity-50";

function Badge({ tone, children }: { tone: "muted" | "warn"; children: string }) {
  const className =
    tone === "warn"
      ? "border-amber-200 bg-amber-50 text-amber-700"
      : "border-[var(--otari-line)] bg-[var(--otari-surface)] text-[var(--otari-muted)]";
  return <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${className}`}>{children}</span>;
}

// The endpoint a tool with no api_base of its own will actually call, so a blank
// box reads as "inherits X" rather than as "unconfigured".
function inheritedBase(providers: SearchProviderInfo[], provider: string): string | null {
  return providers.find((entry) => entry.id === provider)?.default_api_base ?? null;
}

function StoredToolRow({
  tool,
  providers,
  onSaved,
}: {
  tool: StoredSearchTool;
  providers: SearchProviderInfo[];
  onSaved: (message: string) => void;
}) {
  const update = useUpdateSearchTool();
  const remove = useDeleteSearchTool();
  const [apiBase, setApiBase] = useState(tool.api_base ?? "");
  // Blank means "keep the stored key". The field is write-only, so it never
  // shows what is stored, only the last four of it in the badge above.
  const [apiKey, setApiKey] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    setApiBase(tool.api_base ?? "");
  }, [tool.api_base]);

  const inherited = inheritedBase(providers, tool.provider);
  const changed = apiBase.trim() !== (tool.api_base ?? "") || apiKey !== "";
  const busy = update.isPending || remove.isPending;

  const save = () => {
    setError("");
    update.mutate(
      {
        name: tool.name,
        body: {
          api_base: apiBase.trim() === "" ? null : apiBase.trim(),
          // Omitted entirely when blank, so saving a URL never clears the key.
          ...(apiKey === "" ? {} : { api_key: apiKey }),
          expected_updated_at: tool.updated_at,
        },
      },
      {
        onSuccess: () => {
          setApiKey("");
          onSaved(`${tool.name} saved`);
        },
        onError: (err) => setError(errorMessage(err)),
      },
    );
  };

  return (
    <div className="flex flex-col gap-2 py-4">
      <div className="flex flex-wrap items-center gap-2">
        <code className="text-sm font-medium text-[var(--otari-ink)]">{tool.name}</code>
        <Badge tone="muted">{tool.provider}</Badge>
        {tool.last4 ? <Badge tone="muted">{`key ····${tool.last4}`}</Badge> : null}
        {tool.decryptable ? null : <Badge tone="warn">Key unreadable: check OTARI_SECRET_KEY</Badge>}
        {tool.shadows_config ? <Badge tone="warn">Overrides the config-file tool of this name</Badge> : null}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <input
          type="text"
          inputMode="url"
          aria-label={`Backend URL for ${tool.name}`}
          value={apiBase}
          disabled={busy}
          placeholder={inherited ? `inherits ${inherited}` : "backend URL"}
          onChange={(event) => setApiBase(event.target.value)}
          className={`w-full sm:w-72 ${INPUT_CLASS}`}
        />
        <input
          type="password"
          autoComplete="new-password"
          aria-label={`New API key for ${tool.name}`}
          value={apiKey}
          disabled={busy}
          placeholder={tool.last4 ? "replace key" : "add key (optional)"}
          onChange={(event) => setApiKey(event.target.value)}
          className={`w-full sm:w-52 ${INPUT_CLASS}`}
        />
        <Button
          size="sm"
          variant="primary"
          aria-label={`Save ${tool.name}`}
          isDisabled={busy || !changed}
          onPress={save}
        >
          {update.isPending ? "Saving…" : "Save"}
        </Button>
        <ConfirmButton
          confirmLabel="Remove"
          isPending={busy}
          onConfirm={() => {
            setError("");
            remove.mutate(tool.name, {
              onSuccess: () => onSaved(`${tool.name} removed`),
              onError: (err) => setError(errorMessage(err)),
            });
          }}
        >
          Remove
        </ConfirmButton>
      </div>
      {error ? <span className="break-words text-xs text-red-700">{error}</span> : null}
    </div>
  );
}

function ConfigToolRow({ tool }: { tool: ConfigSearchTool }) {
  return (
    <div className="flex flex-wrap items-center gap-2 py-4">
      <code className="text-sm font-medium text-[var(--otari-ink)]">{tool.name}</code>
      <Badge tone="muted">{tool.provider}</Badge>
      <Badge tone="muted">Config file</Badge>
      {tool.has_api_key ? <Badge tone="muted">key set</Badge> : null}
      {tool.shadowed ? <Badge tone="warn">Overridden by the stored tool of this name</Badge> : null}
      <span className="text-xs text-[var(--otari-muted)]">
        {tool.api_base ?? "no api_base declared"} · editable only where the config file is defined
      </span>
    </div>
  );
}

function AddToolForm({ providers, onSaved }: { providers: SearchProviderInfo[]; onSaved: (message: string) => void }) {
  const create = useCreateSearchTool();
  const [name, setName] = useState("");
  const [provider, setProvider] = useState(providers[0]?.id ?? "");
  const [apiBase, setApiBase] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [error, setError] = useState("");

  const selected = providers.find((entry) => entry.id === provider);
  const inherited = selected?.default_api_base ?? null;
  // Required only when the provider has no endpoint of its own and nothing else
  // supplies one: a searxng tool on a deployment with a web-search URL set
  // inherits that URL, so the box may be left blank there.
  const baseRequired = Boolean(selected?.requires_api_base) && inherited === null;
  const keyRequired = Boolean(selected?.requires_api_key);
  const ready =
    name.trim() !== "" && (!baseRequired || apiBase.trim() !== "") && (!keyRequired || apiKey !== "");

  const submit = () => {
    setError("");
    const created = name.trim();
    create.mutate(
      {
        name: created,
        provider,
        api_base: apiBase.trim() === "" ? null : apiBase.trim(),
        api_key: apiKey === "" ? null : apiKey,
      },
      {
        onSuccess: () => {
          setName("");
          setApiBase("");
          setApiKey("");
          onSaved(`${created} added`);
        },
        onError: (err) => setError(errorMessage(err)),
      },
    );
  };

  return (
    <div className="flex flex-col gap-2 py-4">
      <span className="text-sm font-medium text-[var(--otari-ink)]">Add a search tool</span>
      <div className="flex flex-wrap items-end gap-2">
        <input
          type="text"
          aria-label="Search tool name"
          value={name}
          placeholder="name, e.g. local"
          disabled={create.isPending}
          onChange={(event) => setName(event.target.value)}
          className={`w-full sm:w-44 ${INPUT_CLASS}`}
        />
        <FilterSelect
          ariaLabel="Search provider"
          value={provider}
          onChange={setProvider}
          options={providers.map((entry) => ({ value: entry.id, label: entry.id }))}
          disabled={create.isPending}
        />
        <input
          type="text"
          inputMode="url"
          aria-label="Search backend URL"
          value={apiBase}
          disabled={create.isPending}
          placeholder={inherited ? `inherits ${inherited}` : baseRequired ? "backend URL (required)" : "backend URL"}
          onChange={(event) => setApiBase(event.target.value)}
          className={`w-full sm:w-72 ${INPUT_CLASS}`}
        />
        <input
          type="password"
          autoComplete="new-password"
          aria-label="Search API key"
          value={apiKey}
          disabled={create.isPending}
          placeholder={keyRequired ? "API key (required)" : "API key (optional)"}
          onChange={(event) => setApiKey(event.target.value)}
          className={`w-full sm:w-52 ${INPUT_CLASS}`}
        />
        <Button size="sm" variant="primary" isDisabled={!ready || create.isPending} onPress={submit}>
          {create.isPending ? "Adding…" : "Add"}
        </Button>
      </div>
      <span className="text-xs text-[var(--otari-muted)]">
        Callers name the tool in <code className="font-mono">search_tool_name</code>, or in the{" "}
        <code className="font-mono">POST /v1/search/{"{tool}"}</code> path. Storing an API key needs{" "}
        <code className="font-mono">OTARI_SECRET_KEY</code> set on the gateway.
      </span>
      {error ? <span className="break-words text-xs text-red-700">{error}</span> : null}
    </div>
  );
}

export function SearchToolsCard({ onSaved }: { onSaved: (message: string) => void }) {
  const tools = useSearchTools();
  const providers = useSearchProviders();
  const known = providers.data ?? [];
  const stored = tools.data?.stored ?? [];
  const fromConfig = tools.data?.config ?? [];

  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold text-[var(--otari-ink)]">Search tools</h2>
      <p className="text-sm text-[var(--otari-muted)]">
        The tools <code className="font-mono">POST /v1/search</code> dispatches against. A searxng tool that declares no
        backend URL uses the web-search URL above, so one entry here exposes the same backend on the direct endpoint.
      </p>
      <ErrorBanner error={tools.error ?? providers.error} />
      <Card>
        <Card.Content className="flex flex-col divide-y divide-[var(--otari-line)] px-5 py-1">
          {stored.map((tool) => (
            <StoredToolRow key={tool.name} tool={tool} providers={known} onSaved={onSaved} />
          ))}
          {fromConfig.map((tool) => (
            <ConfigToolRow key={tool.name} tool={tool} />
          ))}
          {stored.length === 0 && fromConfig.length === 0 && !tools.isLoading ? (
            <p className="py-4 text-sm text-[var(--otari-muted)]">
              No search tools configured, so <code className="font-mono">POST /v1/search</code> refuses every request.
            </p>
          ) : null}
          {known.length > 0 ? <AddToolForm providers={known} onSaved={onSaved} /> : null}
        </Card.Content>
      </Card>
    </section>
  );
}
