import { useState } from "react"
import { FiChevronDown } from "react-icons/fi"

import type {
  CreatedSearchTool,
  CreateSearchToolRequest,
  SearchProviderInfo,
  UpdateSearchToolRequest,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { parsePerCallPrice, toPerCall } from "@/features/toolPricing/toolRates"
import { useSetPricing } from "@/shared/api/pricing"
import {
  useCreateSearchTool,
  useTestSearchTool,
  useTestUnsavedSearchTool,
  useUpdateSearchTool,
} from "@/shared/api/tools"
import { formatUnitRate } from "@/shared/helpers/format"

import { ProviderOptionFields } from "./ProviderOptionFields"
import {
  buildOptions,
  nameProblem,
  perCallText,
  sameOptions,
  servesTheModel,
  splitOptions,
  storedToolRow,
  suggestName,
  TEST_INPUT,
  testOutcome,
  type WebToolKind,
  type WebToolRow,
} from "./webTools"

// Adds a search or fetch tool, or configures a stored one, with its price.
//
// The provider comes first and nothing else shows until it is picked, because
// everything after it is the provider's: whether a key or an endpoint is
// required, and which options exist. Then the required fields and the price,
// and everything optional under "More settings".
//
// The price is written after the tool, as its own `POST /pricing`: no route
// takes the two together. A price that fails leaves the tool in place, and the
// result step says so, with Edit on the row to retry.

const NOUN: Readonly<Record<WebToolKind, string>> = {
  search: "search tool",
  fetch: "fetch tool",
}

const REDIRECT_WARNING =
  "does not report where a redirect led, so a workspace's allowed and blocked domains are not checked past one. Keep builtin_fetch where they must hold."

const PRICE_HELP: Readonly<Record<WebToolKind, string>> = {
  search:
    "Blank: the model's searches on it are billed at Web search's rate, and its calls on the search API are unpriced, which require_pricing refuses. A call the provider reports a cost for is billed at that cost instead.",
  fetch:
    "Blank bills it at Web fetch's rate. A call the provider reports a cost for is billed at that cost instead.",
}

/** What a tool added with no price of its own is billed at, for the result step. */
const UNPRICED_RESULT: Readonly<Record<WebToolKind, string>> = {
  search:
    "It has no price of its own: the model's searches on it are billed at Web search's rate, and its calls on the search API go unpriced, which require_pricing refuses. Set one with Edit on its row.",
  fetch:
    "It has no price of its own, so its calls are billed at Web fetch's rate. Set one with Edit on its row.",
}

function capitalized(text: string) {
  return text.charAt(0).toUpperCase() + text.slice(1)
}

interface Draft {
  provider: string
  name: string
  apiKey: string
  apiBase: string
  timeout: string
  fetchTool: string
  options: Record<string, string>
  extraJson: string
  price: string
}

function initialDraft(
  editing: WebToolRow | undefined,
  provider: SearchProviderInfo | undefined,
  currentPrice: number | undefined,
): Draft {
  const split = splitOptions(provider, editing?.options ?? {})
  return {
    provider: editing?.provider ?? "",
    name: editing?.name ?? "",
    apiKey: "",
    apiBase: editing?.apiBase ?? "",
    timeout: editing?.timeout === undefined ? "" : String(editing.timeout),
    fetchTool: editing?.fetchTool ?? "",
    options: split.typed,
    extraJson: split.extraJson,
    price: currentPrice === undefined ? "" : perCallText(currentPrice),
  }
}

export function WebToolDialog({
  kind,
  editing,
  providers,
  takenNames,
  fetchTools,
  fetchProviders,
  fetchDefault,
  currentPrice,
  onClose,
  onAddAnother,
}: {
  kind: WebToolKind
  /** The stored tool being configured; absent to add one. */
  editing?: WebToolRow
  /** The catalog for this kind. */
  providers: readonly SearchProviderInfo[]
  /** Every search and fetch tool's name, since names are unique across both. */
  takenNames: readonly string[]
  /** The fetch tools a search tool may be enriched by. */
  fetchTools: readonly WebToolRow[]
  fetchProviders: readonly SearchProviderInfo[]
  /** The fetch tool that enriches a search tool that names none. */
  fetchDefault: string
  /** The tool's own per-million rate in force, `undefined` when it has none. */
  currentPrice: number | undefined
  onClose: () => void
  onAddAnother: () => void
}) {
  const isEdit = editing !== undefined
  const noun = NOUN[kind]
  const create = useCreateSearchTool()
  const update = useUpdateSearchTool()
  const setPricing = useSetPricing()
  const testUnsaved = useTestUnsavedSearchTool()
  const testSaved = useTestSearchTool()
  const test = isEdit ? testSaved : testUnsaved

  const [draft, setDraft] = useState<Draft>(() =>
    initialDraft(
      editing,
      providers.find((entry) => entry.id === editing?.provider),
      currentPrice,
    ),
  )
  // The price as the dialog opened with it, so an edit writes the price only
  // when its field was changed, and never rewrites one it only displayed.
  const [openedPrice] = useState(draft.price)
  // The stored tool as last saved: an edit whose price fails after its update
  // went through compares against this, with the update's own updated_at.
  const [saved, setSaved] = useState(editing)
  const [isMoreOpen, setIsMoreOpen] = useState(false)
  const [testInput, setTestInput] = useState(
    TEST_INPUT[kind].query ?? TEST_INPUT[kind].url ?? "",
  )
  const [created, setCreated] = useState<{
    tool: CreatedSearchTool
    priceError: unknown
    hasPrice: boolean
  }>()
  const { isDirty } = useDirtySnapshot(draft)

  const provider = providers.find((entry) => entry.id === draft.provider)
  const others = takenNames.filter((name) => name !== editing?.name)
  const suggested = suggestName(kind, draft.provider, others)
  const name = isEdit ? draft.name : draft.name.trim() || suggested
  const namedProblem = isEdit ? "" : nameProblem(draft.name.trim(), others)
  const keyRequired = !isEdit && Boolean(provider?.requires_api_key)
  const baseRequired =
    Boolean(provider?.requires_api_base) && !provider?.default_api_base
  const built = buildOptions(provider, draft.options, draft.extraJson)
  const price = parsePerCallPrice(draft.price)
  const isPriceInvalid = draft.price.trim() !== "" && price === undefined
  const timeout = draft.timeout.trim() === "" ? null : Number(draft.timeout)
  const isTimeoutInvalid =
    timeout !== null && !(Number.isFinite(timeout) && timeout > 0)
  const isReady =
    provider !== undefined &&
    name !== "" &&
    namedProblem === "" &&
    (!keyRequired || draft.apiKey !== "") &&
    (!baseRequired || draft.apiBase.trim() !== "") &&
    built.ok &&
    !isPriceInvalid &&
    !isTimeoutInvalid
  const enrichedBy = draft.fetchTool || fetchDefault
  const enrichingProvider = fetchProviders.find(
    (entry) =>
      entry.id ===
      fetchTools.find((tool) => tool.name === enrichedBy)?.provider,
  )

  const set = (patch: Partial<Draft>) => {
    setDraft({ ...draft, ...patch })
    // An unsaved test describes the settings it ran on, not the changed ones.
    if (!isEdit) testUnsaved.reset()
  }

  const request = (): CreateSearchToolRequest => ({
    name,
    kind,
    provider: draft.provider,
    api_key: draft.apiKey === "" ? null : draft.apiKey,
    api_base: draft.apiBase.trim() === "" ? null : draft.apiBase.trim(),
    timeout,
    fetch_tool: kind === "search" && draft.fetchTool ? draft.fetchTool : null,
    options:
      built.ok && Object.keys(built.options).length > 0 ? built.options : null,
  })

  // Only what changed, so an edit never trips over a rule its untouched fields
  // break (the API checks names and options only where a write sets them).
  const changes = (): UpdateSearchToolRequest => {
    const body: UpdateSearchToolRequest = {
      expected_updated_at: saved?.stored?.updated_at ?? null,
    }
    const next = request()
    if (draft.apiKey !== "") body.api_key = draft.apiKey
    if ((next.api_base ?? "") !== (saved?.apiBase ?? "")) {
      body.api_base = next.api_base
    }
    if ((next.timeout ?? undefined) !== saved?.timeout) {
      body.timeout = next.timeout
    }
    if (kind === "search" && (next.fetch_tool ?? "") !== saved?.fetchTool) {
      body.fetch_tool = next.fetch_tool
    }
    if (!sameOptions(next.options ?? {}, saved?.options ?? {})) {
      body.options = next.options ?? {}
    }
    return body
  }

  const priceRequest = (toolName: string, perMillion: number) =>
    setPricing.mutateAsync({
      kind,
      model_key: toolName,
      input_price_per_million: perMillion,
      // A call is one unit; there is no output side of it to price.
      output_price_per_million: 0,
      unit: "requests",
    })

  const submit = async () => {
    if (!isReady) return
    if (editing) {
      try {
        const body = changes()
        if (Object.keys(body).length > 1) {
          const updated = await update.mutateAsync({ name: editing.name, body })
          setSaved(storedToolRow(kind, updated))
          setDraft((current) => ({ ...current, apiKey: "" }))
        }
        if (price !== undefined && draft.price.trim() !== openedPrice) {
          await priceRequest(editing.name, price)
        }
        onClose()
      } catch {
        // Shown in the dialog through the mutations' errors.
      }
      return
    }
    let tool: CreatedSearchTool
    try {
      tool = await create.mutateAsync(request())
    } catch {
      return
    }
    let priceError: unknown = null
    if (price !== undefined) {
      try {
        await priceRequest(tool.name, price)
      } catch (error) {
        priceError = error
      }
    }
    setCreated({ tool, priceError, hasPrice: price !== undefined })
  }

  const runTest = () => {
    const input =
      kind === "search"
        ? { query: testInput.trim(), url: null }
        : { query: null, url: testInput.trim() }
    if (editing) testSaved.mutate({ name: editing.name, body: input })
    else testUnsaved.mutate({ ...request(), ...input })
  }

  if (created) {
    const { tool, priceError, hasPrice } = created
    return (
      <FormDialog
        isOpen
        onOpenChange={(open) => {
          if (!open) onClose()
        }}
        title={`${capitalized(noun)} added`}
        submitLabel="Done"
        onSubmit={onClose}
        isPending={false}
        hideCancel
        footerStart={<Button onPress={onAddAnother}>Add another</Button>}
      >
        <div className="flex flex-col gap-1">
          <code className="font-mono text-body">{tool.name}</code>
          <p className="text-caption">
            {[
              tool.provider,
              tool.api_base ?? "provider endpoint",
              hasPrice && priceError === null && price !== undefined
                ? `${formatUnitRate(toPerCall(price))} a call`
                : "no price of its own",
            ].join(" · ")}
          </p>
        </div>
        {tool.notice ? <InfoBanner>{tool.notice}</InfoBanner> : null}
        {priceError ? (
          <InfoBanner tone="warning">
            Its price was not saved: {errorMessage(priceError)} Set it with Edit
            on its row.
          </InfoBanner>
        ) : !hasPrice ? (
          <p className="text-caption">{UNPRICED_RESULT[kind]}</p>
        ) : null}
      </FormDialog>
    )
  }

  const keyDescription = isEdit
    ? "Leave blank to keep the stored key."
    : keyRequired
      ? "Encrypted at rest and never shown again. Storing it needs OTARI_SECRET_KEY on the gateway."
      : "Optional for this provider. Encrypted at rest; storing it needs OTARI_SECRET_KEY on the gateway."
  const endpoint = (
    <Field
      label="Endpoint"
      value={draft.apiBase}
      onChange={(apiBase) => set({ apiBase })}
      isRequired={baseRequired}
      placeholder={provider?.default_api_base ?? "https://…"}
      description={
        provider?.default_api_base
          ? `Leave blank to use ${provider.default_api_base}.`
          : baseRequired
            ? "This provider has no endpoint of its own, so it needs one here."
            : "Optional for this provider."
      }
    />
  )

  return (
    <FormDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      size="lg"
      title={editing ? `Configure ${editing.name}` : `Add ${noun}`}
      description={
        kind === "search"
          ? "What the model's otari_web_search and POST /api/v1/search run on, for every workspace."
          : "What the model's otari_web_fetch, and search hits with no text, are read with, for every workspace."
      }
      submitLabel={isEdit ? "Save" : `Add ${noun}`}
      onSubmit={() => void submit()}
      isPending={create.isPending || update.isPending || setPricing.isPending}
      isSubmitDisabled={!isReady}
      isDirty={isDirty}
      error={create.error ?? update.error ?? setPricing.error}
      footerStart={
        provider ? (
          <Button
            onPress={runTest}
            isPending={test.isPending}
            isDisabled={!isReady || testInput.trim() === ""}
          >
            Test
          </Button>
        ) : null
      }
    >
      {editing ? (
        <p className="text-caption">
          <code className="font-mono">{editing.name}</code> on{" "}
          {editing.provider}. The name and provider are fixed: the price and
          access lists follow the name, the options the provider.
        </p>
      ) : (
        <Select
          label="Provider"
          value={draft.provider}
          onChange={(next) =>
            // The options and endpoint belong to the provider, so a change
            // drops them rather than carrying them to one that may refuse them.
            set({ provider: next, apiBase: "", options: {}, extraJson: "" })
          }
          placeholder="Choose a provider"
          options={providers.map((entry) => ({
            value: entry.id,
            label:
              kind === "search" && !servesTheModel(entry)
                ? `${entry.id} (search API only)`
                : entry.id,
          }))}
          description={
            provider?.doc_url ? (
              <a
                href={provider.doc_url}
                target="_blank"
                rel="noreferrer"
                className="text-link hover:text-link-hover"
              >
                {provider.id}&rsquo;s API reference
              </a>
            ) : undefined
          }
          autoFocus
        />
      )}

      {provider ? (
        <>
          {kind === "fetch" && provider.reports_final_url === false ? (
            <InfoBanner tone="warning">
              {provider.id} {REDIRECT_WARNING}
            </InfoBanner>
          ) : null}
          {kind === "search" && !servesTheModel(provider) ? (
            <InfoBanner tone="warning">
              The model cannot search with {provider.id} yet; a tool on it
              serves the search API only.
            </InfoBanner>
          ) : null}
          {isEdit ? null : (
            <Field
              label="Name"
              value={draft.name}
              onChange={(next) => set({ name: next })}
              placeholder={suggested}
              description={
                kind === "search"
                  ? `Optional. Callers reach it as /api/v1/search/${name || "<name>"}; leave blank to name it ${suggested}.`
                  : `Optional. web_fetch_default_tool and a search tool's enrichment name it; leave blank to name it ${suggested}.`
              }
              isInvalid={namedProblem !== ""}
              errorMessage={namedProblem || undefined}
            />
          )}
          <SecretField
            label="API key"
            value={draft.apiKey}
            onChange={(apiKey) => set({ apiKey })}
            isRequired={keyRequired}
            placeholder={
              editing?.last4 ? `Keep ••••${editing.last4}` : undefined
            }
            description={keyDescription}
          />
          {baseRequired ? endpoint : null}
          <Field
            label="Price per call, USD"
            value={draft.price}
            onChange={(next) => set({ price: next })}
            inputMode="decimal"
            placeholder="0.00"
            isInvalid={isPriceInvalid}
            errorMessage={
              isPriceInvalid ? "An amount in dollars, such as 0.01." : undefined
            }
            description={
              isEdit && currentPrice !== undefined
                ? "0 prices it as free. Clearing the field keeps the current price."
                : PRICE_HELP[kind]
            }
          />

          <div>
            <Button
              size="sm"
              variant="ghost"
              aria-expanded={isMoreOpen}
              onPress={() => setIsMoreOpen(!isMoreOpen)}
            >
              {`More settings for ${provider.id}`}
              <FiChevronDown
                aria-hidden
                className={`transition-transform ${isMoreOpen ? "rotate-180" : ""}`}
              />
            </Button>
          </div>
          {isMoreOpen ? (
            <div className="flex flex-col gap-4">
              {baseRequired ? null : endpoint}
              <Field
                label="Timeout, seconds"
                value={draft.timeout}
                onChange={(next) => set({ timeout: next })}
                inputMode="decimal"
                placeholder="Gateway default"
                isInvalid={isTimeoutInvalid}
                errorMessage={
                  isTimeoutInvalid ? "A number of seconds above 0." : undefined
                }
              />
              {kind === "search" ? (
                <Select
                  label="Enrich with"
                  value={draft.fetchTool}
                  onChange={(fetchTool) => set({ fetchTool })}
                  placeholder={`The fetch tool (${fetchDefault})`}
                  options={[
                    { value: "", label: `The fetch tool (${fetchDefault})` },
                    ...fetchTools.map((tool) => ({
                      value: tool.name,
                      label: tool.name,
                    })),
                  ]}
                  description={
                    enrichingProvider?.reports_final_url === false
                      ? `Reads the page for a hit that has no text. ${enrichingProvider.id} ${REDIRECT_WARNING}`
                      : "Reads the page for a hit that has no text."
                  }
                />
              ) : null}
              <ProviderOptionFields
                provider={provider}
                values={draft.options}
                errors={built.ok ? {} : built.fieldErrors}
                extraJson={draft.extraJson}
                jsonError={built.ok ? "" : built.jsonError}
                onChange={(options) => set({ options })}
                onExtraJsonChange={(extraJson) => set({ extraJson })}
              />
            </div>
          ) : null}

          <Field
            label={kind === "search" ? "Test query" : "Test URL"}
            value={testInput}
            onChange={setTestInput}
            description={
              isEdit
                ? "Test runs on the saved settings, not the changes above."
                : "Test runs one call on these settings before anything is saved."
            }
          />
          {test.data ? (
            <p role="status" className="text-caption">
              {testOutcome(kind, test.data)}
            </p>
          ) : null}
          <ErrorBanner error={test.error} />
        </>
      ) : null}
    </FormDialog>
  )
}
