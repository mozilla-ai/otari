import { useState } from "react"
import type {
  ConfigSearchTool,
  SearchProviderInfo,
  StoredSearchTool,
  UpdateSearchToolRequest,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { INPUT_CLASS } from "@/design-system/forms/inputClass"
import { SecretField } from "@/design-system/forms/SecretField"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { DisclosureRow } from "@/design-system/navigation/DisclosureRow"
import { usePolicyWriter } from "@/features/tools/usePolicyWriter"
import {
  useCreateSearchTool,
  useDeleteSearchTool,
  useSearchProviders,
  useSearchTools,
  useUpdateSearchTool,
} from "@/shared/api/tools"
import { commitOnEnter, useAutosave } from "@/shared/hooks/useAutosave"

// Search tools are what POST /api/v1/search dispatches against. They used to be
// declarable only in a config file, so a deployment configured entirely through
// the dashboard could not use that endpoint at all. This is the route in:
// stored tools are editable here, config-file tools are shown read-only so the
// operator can see every tool a caller could name.
//
// A drill-in rather than a fifth group of rows, because it is a list of things
// rather than a set of settings, and it is empty on most deployments.

/** The lanes every line in the panel shares, so the columns read down. */
const NAME_LANE = "w-full shrink-0 font-mono text-xs md:w-[7.5rem]"
const PROVIDER_LANE = "w-full shrink-0 text-caption text-subtle md:w-[5.5rem]"

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
  const urlSave = useAutosave()
  const keySave = useAutosave()
  const [apiBase, setApiBase] = useState(tool.api_base ?? "")
  const [syncedBase, setSyncedBase] = useState(tool.api_base ?? "")
  // Blank means "keep the stored key". The field is write-only, so it never
  // shows what is stored, only the last four of it in its own placeholder.
  const [apiKey, setApiKey] = useState("")
  const [isDeleteOpen, setDeleteOpen] = useState(false)

  const committedBase = tool.api_base ?? ""
  // Re-synced from the server's answer, in render rather than an effect, which
  // is the idiom the other autosaving rows use. After one of this row's own
  // writes the two already agree, so it moves nothing.
  if (committedBase !== syncedBase) {
    setSyncedBase(committedBase)
    setApiBase(committedBase)
  }

  // The row is one stored tool, and every write carries `expected_updated_at`.
  // Under a Save button that was one write per click; under autosave, leaving
  // the URL and then the key fires two, and the second would still carry the
  // `updated_at` from before the first. The writer takes the fresh one from the
  // previous write's own response, which is the same job it does for the
  // workspace policies.
  const write = usePolicyWriter({
    server: tool,
    resetKey: tool.name,
    // `api_key` is deliberately absent: omitted means "keep the stored key",
    // so only the commit that changes it puts it on the wire.
    toBody: (stored: StoredSearchTool): UpdateSearchToolRequest => ({
      api_base: stored.api_base,
      expected_updated_at: stored.updated_at,
    }),
    put: (body: UpdateSearchToolRequest) =>
      update.mutateAsync({ name: tool.name, body }),
  })

  const inherited = inheritedBase(providers, tool.provider)
  const busy = remove.isPending

  return (
    <div className="flex flex-col gap-2 px-4 py-3">
      <div className="flex flex-col gap-2 md:flex-row md:flex-wrap md:items-center">
        <code className={NAME_LANE}>{tool.name}</code>
        <span className={PROVIDER_LANE}>{tool.provider}</span>
        <input
          type="text"
          inputMode="url"
          aria-label={`Backend URL for ${tool.name}`}
          aria-invalid={urlSave.error ? true : undefined}
          value={apiBase}
          disabled={busy || urlSave.isSaving}
          placeholder={inherited ? `inherits ${inherited}` : "backend URL"}
          onChange={(event) => setApiBase(event.target.value)}
          onKeyDown={commitOnEnter}
          onBlur={() => {
            const next = apiBase.trim()
            if (next === committedBase) return
            void urlSave.run(() =>
              write({ api_base: next === "" ? null : next }),
            )
          }}
          className={`otari-machine-field w-full md:w-[15rem] ${INPUT_CLASS}`}
        />
        <input
          type="password"
          autoComplete="new-password"
          aria-label={`New API key for ${tool.name}`}
          aria-invalid={keySave.error ? true : undefined}
          value={apiKey}
          disabled={busy || keySave.isSaving}
          placeholder={tool.last4 ? `replace key ····${tool.last4}` : "add key"}
          onChange={(event) => setApiKey(event.target.value)}
          onKeyDown={commitOnEnter}
          // Blank is not a value here, it is "leave the stored key alone", so
          // an empty blur writes nothing and a focus-and-leave costs nothing.
          onBlur={() => {
            if (apiKey === "") return
            void keySave.run(async () => {
              await write({ api_key: apiKey })
              setApiKey("")
            })
          }}
          className={`otari-machine-field w-full md:w-[10rem] ${INPUT_CLASS}`}
        />
        <Button
          variant="ghost"
          // Named per row, as this row's fields are: the card is a list of
          // tools, so a bare "Remove" is the same name on every one of them.
          aria-label={`Remove ${tool.name}`}
          isDisabled={busy}
          onPress={() => setDeleteOpen(true)}
        >
          Remove
        </Button>
      </div>
      {tool.decryptable ? null : (
        <p className="text-caption text-warning">
          Key unreadable: check OTARI_SECRET_KEY
        </p>
      )}
      {tool.shadows_config ? (
        <p className="text-caption text-warning">
          Overrides the config-file tool of this name
        </p>
      ) : null}
      {urlSave.error || keySave.error ? (
        <p role="alert" className="break-words text-caption text-danger">
          {urlSave.error || keySave.error}
        </p>
      ) : null}

      <ConfirmDialog
        isOpen={isDeleteOpen}
        // Cleared on the way out: a refusal otherwise sits on the mutation
        // and greets the next open as if it had just happened.
        onOpenChange={(open) => {
          setDeleteOpen(open)
          if (!open) remove.reset()
        }}
        heading="Remove search tool"
        body={`${tool.name} and the key stored with it are removed. A request that still names it as a tool is refused, so update the callers that use it.`}
        confirmLabel="Remove permanently"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          remove.mutate(tool.name, { onSuccess: () => setDeleteOpen(false) })
        }}
      />
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
      <span className="min-w-0 flex-1 text-caption text-subtle">
        {tool.api_base ?? "no api_base declared"} · config file, editable where
        it is defined
      </span>
      <span className="shrink-0 text-mono-overline text-subtle">
        {tool.has_api_key ? "Key set" : "No key"}
      </span>
      {tool.shadowed ? (
        <p className="text-caption text-warning">
          Overridden by the stored tool of this name
        </p>
      ) : null}
    </div>
  )
}

function AddToolDialog({
  isOpen,
  onClose,
  onCreated,
  providers,
}: {
  isOpen: boolean
  onClose: () => void
  /**
   * A tool landed. Separate from `onClose` because the created row is inside a
   * drill-in that is collapsed by default: closing on a cancel should leave it
   * as it was, and closing on a create should open it, or the only thing that
   * changes on screen is the trailing count.
   */
  onCreated: () => void
  providers: SearchProviderInfo[]
}) {
  const create = useCreateSearchTool()
  const [name, setName] = useState("")
  const [provider, setProvider] = useState(providers[0]?.id ?? "")
  const [apiBase, setApiBase] = useState("")
  const [apiKey, setApiKey] = useState("")

  const selected = providers.find((entry) => entry.id === provider)
  const inherited = selected?.default_api_base ?? null
  // Required only when the provider has no endpoint of its own and nothing else
  // supplies one: a searxng tool on a deployment with a web-search URL set
  // inherits that URL, so the box may be left blank there.
  const baseRequired =
    Boolean(selected?.requires_api_base) && inherited === null
  const keyRequired = Boolean(selected?.requires_api_key)
  // One snapshot rather than a hand-listed predicate: the provider is a field
  // like the others, and a guard that forgot it discarded a changed provider
  // with no question asked.
  const { isDirty } = useDirtySnapshot({ name, provider, apiBase, apiKey })

  const ready =
    name.trim() !== "" &&
    (!baseRequired || apiBase.trim() !== "") &&
    (!keyRequired || apiKey !== "")

  const submit = () => {
    create.mutate(
      {
        name: name.trim(),
        provider,
        api_base: apiBase.trim() === "" ? null : apiBase.trim(),
        api_key: apiKey === "" ? null : apiKey,
      },
      { onSuccess: onCreated },
    )
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title="New search tool"
      submitLabel="Add search tool"
      onSubmit={submit}
      isPending={create.isPending}
      isSubmitDisabled={!ready}
      isDirty={isDirty}
      error={create.error}
    >
      <Field
        label="Name"
        value={name}
        onChange={setName}
        isRequired
        autoFocus
        placeholder="local"
        description="What a caller names in search_tool_name, or in the /api/v1/search/{tool} path."
      />
      <Select
        label="Provider"
        value={provider}
        onChange={setProvider}
        options={providers.map((entry) => ({
          value: entry.id,
          label: entry.id,
        }))}
        reserveMessage={false}
      />
      <Field
        label="Backend URL"
        value={apiBase}
        onChange={setApiBase}
        isRequired={baseRequired}
        placeholder={inherited ?? "https://…"}
        description={
          inherited
            ? `Leave blank to inherit ${inherited}.`
            : baseRequired
              ? "This provider has no endpoint of its own, so it needs one here."
              : "Optional for this provider."
        }
      />
      <SecretField
        label="API key"
        value={apiKey}
        onChange={setApiKey}
        isRequired={keyRequired}
        description={
          keyRequired
            ? "This provider needs one. Storing it needs OTARI_SECRET_KEY set on the gateway."
            : "Optional for this provider. Storing one needs OTARI_SECRET_KEY set on the gateway."
        }
      />
    </FormDialog>
  )
}

/**
 * The named tools behind `POST /api/v1/search`, as a row that drills in.
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
  const [adding, setAdding] = useState(false)
  const [openCount, setOpenCount] = useState(0)

  const known = providers.data ?? []
  const stored = tools.data?.stored ?? []
  const fromConfig = tools.data?.config ?? []
  const count = stored.length + fromConfig.length
  // Nothing is known until the read answers, and "0 tools · refuses every
  // request" is a claim, not a placeholder. A failed read is not an empty
  // deployment either: `isLoading` goes false with no data behind it.
  const failed = Boolean(tools.error)
  const answered = !tools.isLoading && !failed

  return (
    <>
      {/* Outside the group, not inside it: `FormDialog` renders its trigger
          slot as a real element, and a group's rows are a `divide-y` container
          where one more child changes which row is last. */}
      {/* Keyed on the open count, so each open remounts a blank form. Clearing
          the draft on close instead would blank the fields while the dialog is
          still animating away. */}
      <AddToolDialog
        key={openCount}
        isOpen={adding}
        onClose={() => setAdding(false)}
        onCreated={() => {
          setAdding(false)
          // The new row lives in the drill-in, which is collapsed by default,
          // so without this the only thing that changes on screen is the count.
          setIsOpen(true)
        }}
        providers={known}
      />
      <SettingsGroup
        bounded
        title="Search tools"
        description="Named tools behind the direct endpoint, POST /api/v1/search. A searxng tool with no URL of its own reuses the backend above."
        docsHref={docsHref}
        action={
          known.length > 0 ? (
            <Button
              variant="primary"
              onPress={() => {
                setOpenCount((count) => count + 1)
                setAdding(true)
              }}
            >
              Add search tool
            </Button>
          ) : null
        }
      >
        {tools.error || providers.error ? (
          // Outside the disclosure: a read that failed is the thing the operator
          // most needs to see, and the row is collapsed by default.
          <div className="px-4 py-3">
            <ErrorBanner error={tools.error ?? providers.error} />
          </div>
        ) : null}
        <DisclosureRow
          label="Configure search tools"
          help={
            failed
              ? "Could not read the tools this deployment serves."
              : !answered
                ? "Reading the tools this deployment serves."
                : count === 0
                  ? "None configured, so POST /api/v1/search refuses every request."
                  : "Callers name one in search_tool_name, or in the /api/v1/search/{tool} path."
          }
          isOpen={isOpen}
          onToggle={() => setIsOpen((open) => !open)}
          trailing={
            <span className="text-caption text-subtle tabular-nums">
              {answered ? toolCount(count) : ""}
            </span>
          }
        >
          <div className="flex flex-col divide-y divide-border-subtle">
            {stored.map((tool) => (
              <StoredToolLine key={tool.name} tool={tool} providers={known} />
            ))}
            {fromConfig.map((tool) => (
              <ConfigToolLine key={tool.name} tool={tool} />
            ))}
          </div>
        </DisclosureRow>
      </SettingsGroup>
    </>
  )
}
