import { Button } from "@heroui/react"
import { useEffect, useState } from "react"

import type {
  ConfigSearchTool,
  SearchProviderInfo,
  StoredSearchTool,
} from "@/client"
import {
  useCreateSearchTool,
  useDeleteSearchTool,
  useSearchProviders,
  useSearchTools,
  useUpdateSearchTool,
} from "@/shared/api/tools"
import { ConfirmButton } from "@/shared/components/actions/ConfirmButton"
import { ErrorBanner } from "@/shared/components/feedback/ErrorBanner"
import { errorMessage } from "@/shared/components/feedback/errorMessage"
import { INPUT_CLASS } from "@/shared/components/forms/inputClass"
import { Dot } from "@/shared/components/indicators/Dot"
import { SettingsGroup } from "@/shared/components/layout/SettingsGroup"
import { DisclosureRow } from "@/shared/components/navigation/DisclosureRow"
import { FilterSelect } from "@/shared/components/navigation/FilterSelect"
import { useAutosave } from "@/shared/hooks/useAutosave"

// Search tools are what POST /v1/search dispatches against. They used to be
// declarable only in a config file, so a deployment configured entirely through
// the dashboard could not use that endpoint at all. This is the route in:
// stored tools are editable here, config-file tools are shown read-only so the
// operator can see every tool a caller could name.
//
// A drill-in rather than a fifth group of rows, because it is a list of things
// rather than a set of settings, and it is empty on most deployments.

/** The lanes every line in the panel shares, so the columns read down. */
const NAME_LANE = "w-full shrink-0 font-mono text-xs md:w-[7.5rem]"
const PROVIDER_LANE = "w-full shrink-0 text-xs text-subtle md:w-[5.5rem]"

// The endpoint a tool with no api_base of its own will actually call, so a blank
// box reads as "inherits X" rather than as "unconfigured".
function inheritedBase(
  providers: SearchProviderInfo[],
  provider: string,
): string | null {
  return (
    providers.find((entry) => entry.id === provider)?.default_api_base ?? null
  )
}

function toolCount(count: number): string {
  return `${count} ${count === 1 ? "tool" : "tools"}`
}

function StoredToolLine({
  tool,
  providers,
}: {
  tool: StoredSearchTool
  providers: SearchProviderInfo[]
}) {
  const update = useUpdateSearchTool()
  const remove = useDeleteSearchTool()
  const save = useAutosave()
  const [apiBase, setApiBase] = useState(tool.api_base ?? "")
  // Blank means "keep the stored key". The field is write-only, so it never
  // shows what is stored, only the last four of it beside the row.
  const [apiKey, setApiKey] = useState("")
  // Re-synced from the server's answer, so a change made elsewhere lands in the
  // box rather than leaving a stale draft over it.
  useEffect(() => setApiBase(tool.api_base ?? ""), [tool.api_base])

  const inherited = inheritedBase(providers, tool.provider)
  const changed = apiBase.trim() !== (tool.api_base ?? "") || apiKey !== ""
  const busy = update.isPending || remove.isPending

  const commit = () =>
    void save.run(async () => {
      await update.mutateAsync({
        name: tool.name,
        body: {
          api_base: apiBase.trim() === "" ? null : apiBase.trim(),
          // Omitted entirely when blank, so saving a URL never clears the key.
          ...(apiKey === "" ? {} : { api_key: apiKey }),
          expected_updated_at: tool.updated_at,
        },
      })
      setApiKey("")
    })

  return (
    <div className="flex flex-col gap-2 px-4 py-3">
      <div className="flex flex-col gap-2 md:flex-row md:flex-wrap md:items-center">
        <code className={NAME_LANE}>{tool.name}</code>
        <span className={PROVIDER_LANE}>{tool.provider}</span>
        <input
          type="text"
          inputMode="url"
          aria-label={`Backend URL for ${tool.name}`}
          value={apiBase}
          disabled={busy}
          placeholder={inherited ? `inherits ${inherited}` : "backend URL"}
          onChange={(event) => setApiBase(event.target.value)}
          className={`field-machine w-full md:w-[15rem] ${INPUT_CLASS}`}
        />
        <input
          type="password"
          autoComplete="new-password"
          aria-label={`New API key for ${tool.name}`}
          value={apiKey}
          disabled={busy}
          placeholder={tool.last4 ? `replace key ····${tool.last4}` : "add key"}
          onChange={(event) => setApiKey(event.target.value)}
          className={`field-machine w-full md:w-[10rem] ${INPUT_CLASS}`}
        />
        <Button
          size="sm"
          aria-label={`Save ${tool.name}`}
          isDisabled={busy || !changed}
          onPress={commit}
        >
          Save
        </Button>
        <ConfirmButton
          confirmLabel="Remove permanently"
          isPending={busy}
          onConfirm={() => void save.run(() => remove.mutateAsync(tool.name))}
        >
          Remove
        </ConfirmButton>
      </div>
      {tool.decryptable ? null : (
        <p className="text-xs text-warning">
          Key unreadable: check OTARI_SECRET_KEY
        </p>
      )}
      {tool.shadows_config ? (
        <p className="text-xs text-warning">
          Overrides the config-file tool of this name
        </p>
      ) : null}
      {save.error ? (
        <p role="alert" className="break-words text-xs text-danger">
          {save.error}
        </p>
      ) : null}
    </div>
  )
}

// A config-file tool is a fact, not a control: it is editable only where the
// file is defined, so it reads as a line of values.
function ConfigToolLine({ tool }: { tool: ConfigSearchTool }) {
  return (
    <div className="flex flex-col gap-2 px-4 py-3 md:flex-row md:flex-wrap md:items-center">
      <code className={NAME_LANE}>{tool.name}</code>
      <span className={PROVIDER_LANE}>{tool.provider}</span>
      <span className="min-w-0 flex-1 text-xs text-subtle">
        {tool.api_base ?? "no api_base declared"} · config file, editable where
        it is defined
      </span>
      <span className="flex shrink-0 items-center gap-2.5 text-mono-overline text-subtle">
        <Dot className="bg-text-subtle" />
        {tool.has_api_key ? "Key set" : "No key"}
      </span>
      {tool.shadowed ? (
        <p className="text-xs text-warning">
          Overridden by the stored tool of this name
        </p>
      ) : null}
    </div>
  )
}

function AddToolForm({ providers }: { providers: SearchProviderInfo[] }) {
  const create = useCreateSearchTool()
  const [name, setName] = useState("")
  const [provider, setProvider] = useState(providers[0]?.id ?? "")
  const [apiBase, setApiBase] = useState("")
  const [apiKey, setApiKey] = useState("")
  const [error, setError] = useState("")

  const selected = providers.find((entry) => entry.id === provider)
  const inherited = selected?.default_api_base ?? null
  // Required only when the provider has no endpoint of its own and nothing else
  // supplies one: a searxng tool on a deployment with a web-search URL set
  // inherits that URL, so the box may be left blank there.
  const baseRequired =
    Boolean(selected?.requires_api_base) && inherited === null
  const keyRequired = Boolean(selected?.requires_api_key)
  const ready =
    name.trim() !== "" &&
    (!baseRequired || apiBase.trim() !== "") &&
    (!keyRequired || apiKey !== "")

  const submit = () => {
    setError("")
    create.mutate(
      {
        name: name.trim(),
        provider,
        api_base: apiBase.trim() === "" ? null : apiBase.trim(),
        api_key: apiKey === "" ? null : apiKey,
      },
      {
        onSuccess: () => {
          setName("")
          setApiBase("")
          setApiKey("")
        },
        onError: (cause) => setError(errorMessage(cause)),
      },
    )
  }

  return (
    <div className="flex flex-col gap-2 px-4 py-3">
      <div className="flex flex-col gap-2 md:flex-row md:flex-wrap md:items-center">
        <input
          type="text"
          aria-label="Search tool name"
          value={name}
          placeholder="local"
          disabled={create.isPending}
          onChange={(event) => setName(event.target.value)}
          className={`field-machine w-full md:w-[7.5rem] ${INPUT_CLASS}`}
        />
        <FilterSelect
          ariaLabel="Search provider"
          value={provider}
          onChange={setProvider}
          options={providers.map((entry) => ({
            value: entry.id,
            label: entry.id,
          }))}
          disabled={create.isPending}
        />
        <input
          type="text"
          inputMode="url"
          aria-label="Search backend URL"
          value={apiBase}
          disabled={create.isPending}
          placeholder={
            inherited
              ? `inherits ${inherited}`
              : baseRequired
                ? "backend URL (required)"
                : "backend URL"
          }
          onChange={(event) => setApiBase(event.target.value)}
          className={`field-machine w-full md:w-[15rem] ${INPUT_CLASS}`}
        />
        <input
          type="password"
          autoComplete="new-password"
          aria-label="Search API key"
          value={apiKey}
          disabled={create.isPending}
          placeholder={
            keyRequired ? "API key (required)" : "API key (optional)"
          }
          onChange={(event) => setApiKey(event.target.value)}
          className={`field-machine w-full md:w-[10rem] ${INPUT_CLASS}`}
        />
        <Button
          size="sm"
          variant="primary"
          isDisabled={!ready || create.isPending}
          onPress={submit}
        >
          {create.isPending ? "Adding…" : "Add"}
        </Button>
      </div>
      <p className="text-xs text-subtle">
        Storing an API key needs{" "}
        <code className="font-mono">OTARI_SECRET_KEY</code> set on the gateway.
      </p>
      {error ? (
        <p role="alert" className="break-words text-xs text-danger">
          {error}
        </p>
      ) : null}
    </div>
  )
}

/**
 * The named tools behind `POST /v1/search`, as a row that drills in.
 *
 * A searxng tool that declares no backend URL of its own inherits the
 * deployment's web-search URL, which is why this sits directly under the
 * backend settings: one entry here exposes the same backend on the direct
 * endpoint.
 */
export function SearchToolsCard({ docsHref }: { docsHref: string }) {
  const tools = useSearchTools()
  const providers = useSearchProviders()
  const [isOpen, setIsOpen] = useState(false)

  const known = providers.data ?? []
  const stored = tools.data?.stored ?? []
  const fromConfig = tools.data?.config ?? []
  const count = stored.length + fromConfig.length

  return (
    <SettingsGroup
      bounded
      title="Search tools"
      description="Named tools behind the direct endpoint, POST /v1/search. A searxng tool with no URL of its own reuses the backend above."
      docsHref={docsHref}
    >
      <DisclosureRow
        label="Configure search tools"
        help={
          count === 0
            ? "None configured, so POST /v1/search refuses every request."
            : "Callers name one in search_tool_name, or in the /v1/search/{tool} path."
        }
        isOpen={isOpen}
        onToggle={() => setIsOpen((open) => !open)}
        trailing={
          <span className="text-caption text-subtle tabular-nums">
            {toolCount(count)}
          </span>
        }
      >
        <div className="flex flex-col divide-y divide-border-subtle">
          <ErrorBanner error={tools.error ?? providers.error} />
          {stored.map((tool) => (
            <StoredToolLine key={tool.name} tool={tool} providers={known} />
          ))}
          {fromConfig.map((tool) => (
            <ConfigToolLine key={tool.name} tool={tool} />
          ))}
          {known.length > 0 ? <AddToolForm providers={known} /> : null}
        </div>
      </DisclosureRow>
    </SettingsGroup>
  )
}
