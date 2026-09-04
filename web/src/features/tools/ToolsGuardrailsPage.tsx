import { Button } from "@heroui/react"
import { Fragment, useEffect, useRef, useState } from "react"
import { FiCheck } from "react-icons/fi"
import type {
  ManagedTool,
  ToolServiceName,
  ToolSettingField,
  UpdateToolSettingsRequest,
} from "@/client"
import { isDeploymentOperator } from "@/features/organization/roles"
import { OrganizationGuardrailsCard } from "@/features/tools/OrganizationGuardrailsCard"
import { SearchToolsCard } from "@/features/tools/SearchToolsCard"
import { WorkspaceCodeExecutionPolicyCard } from "@/features/tools/WorkspaceCodeExecutionPolicyCard"
import { WorkspaceMcpServersCard } from "@/features/tools/WorkspaceMcpServersCard"
import { WorkspaceWebSearchCard } from "@/features/tools/WorkspaceWebSearchCard"
import {
  useOrganizationContext,
  usePricing,
  useSetPricing,
  useTestService,
  useToolSettings,
  useTools,
  useUpdateToolSettings,
} from "@/shared/api/hooks"
import { Dot, SettingsGroup } from "@/shared/components/surface"
import {
  ErrorBanner,
  errorMessage,
  FilterSelect,
  INPUT_CLASS,
  PageLoading,
} from "@/shared/components/ui"

// One settable field maps onto one key of the update request; cast at this one
// boundary (the keys come from the backend's field list).
function oneField(
  key: string,
  value: boolean | number | string | null,
): UpdateToolSettingsRequest {
  return { [key]: value } as UpdateToolSettingsRequest
}

// The services, in display order, with the fields each owns (url first). Fields
// not listed for a service are still rendered under it via the fallback, but
// this fixes the order and lets us give each service a short blurb.
const SERVICES: {
  key: ToolServiceName
  label: string
  blurb: string
  order: string[]
  // The pricing key for a tool Otari runs itself. Present only for the two
  // gateway-run tools: guardrails is a check, not billable work.
  pricingKey?: string
  // The /v1/tools id whose calling convention is shown under this service.
  // Absent for guardrails, which is not declared in `tools[]` at all.
  toolId?: string
}[] = [
  {
    key: "web_search",
    label: "Web search",
    blurb:
      "Backend for otari_web_search tools. A SearXNG-shaped service at the URL below, or a licensed API (web_search_provider), which needs no URL.",
    pricingKey: "otari:web_search",
    toolId: "otari_web_search",
    order: [
      "web_search_url",
      "web_search_engines",
      "web_search_max_results",
      "web_search_extract",
      "web_search_intercept",
      "web_search_purpose_hint",
    ],
  },
  {
    key: "sandbox",
    label: "Code execution",
    blurb:
      "Backend for otari_code_execution tools (the sandbox that runs generated code).",
    pricingKey: "otari:code_execution",
    toolId: "otari_code_execution",
    order: ["sandbox_url", "sandbox_session_image", "sandbox_purpose_hint"],
  },
  {
    key: "guardrails",
    label: "Guardrails",
    blurb:
      "Default input-guardrails service used when a request does not pass its own guardrail URL.",
    order: ["guardrails_url"],
  },
]

// A short-lived confirmation toast, styled to match the app's ConnectionStatus
// toast (bottom-right), so a save gives visible feedback without a page-level banner.
function useSaveToast(): [string | null, (message: string) => void] {
  const [message, setMessage] = useState<string | null>(null)
  const timer = useRef<number | undefined>(undefined)
  const show = (next: string) => {
    setMessage(next)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setMessage(null), 2500)
  }
  useEffect(() => () => window.clearTimeout(timer.current), [])
  return [message, show]
}

function SaveToast({ message }: { message: string | null }) {
  if (!message) return null
  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed right-4 bottom-4 z-50 flex items-center gap-2 rounded-lg border border-success bg-success-subtle px-4 py-3 text-sm font-medium text-success shadow-elevation-lg"
    >
      <FiCheck aria-hidden="true" className="h-5 w-5" />
      {message}
    </div>
  )
}

// Every field renders as one grid row with three fixed-width tracks:
// label | input (16rem) | actions (10rem). Because the input and action tracks
// have the same width on every row, the boxes and the Save buttons line up in
// columns down a card regardless of which rows also carry a Test button, an
// extra help line, or a narrower numeric input. Below `sm` the grid collapses
// to a single column and the pieces stack.
//
// The actions track is sized for the Save button so Save stays column-aligned
// across every row; on URL rows the trailing Test button is intentionally left
// to overflow the track's right edge (grid tracks don't clip) rather than
// widening the track, which would push Save inward and break that alignment.
const ROW_CLASS =
  "grid gap-x-4 gap-y-1.5 py-4 sm:grid-cols-[minmax(0,1fr)_16rem_10rem] sm:items-start"
const INPUT_CELL = `w-full sm:col-start-2 ${INPUT_CLASS}`
const ACTIONS_CELL = "flex items-center gap-2 sm:col-start-3"
const MESSAGE_CELL = "flex flex-col gap-1 sm:col-span-2 sm:col-start-2"

function SaveError({ message }: { message?: string }) {
  if (!message) return null
  return <span className="break-words text-caption text-danger">{message}</span>
}

function FieldLabel({
  field,
  help,
}: {
  field: ToolSettingField
  help?: string
}) {
  return (
    <div className="min-w-0 sm:col-start-1">
      <code className="font-mono text-body">{field.key}</code>
      {field.description ? (
        // Capped for the reason every full-bleed row's prose is: the row
        // spans the page, the sentence does not.
        <p className="mt-1 max-w-prose text-caption">{field.description}</p>
      ) : null}
      {help ? <p className="mt-1 text-caption">{help}</p> : null}
    </div>
  )
}

// A URL row: a shared draft that Test probes and Save commits. Test runs against
// the typed (possibly unsaved) value, so an operator can verify before saving.
function UrlRow({
  field,
  onSave,
  saveError,
  disabled,
}: {
  field: ToolSettingField
  onSave: (value: string | null) => void
  saveError?: string
  disabled: boolean
}) {
  const committed = typeof field.value === "string" ? field.value : ""
  const [draft, setDraft] = useState(committed)
  // The URL a still-relevant test result belongs to. A pending request for the old
  // URL can resolve after the field is edited; only render a result whose URL still
  // matches the field, so an outcome is never shown against a different URL.
  const [testedUrl, setTestedUrl] = useState<string | null>(null)
  const test = useTestService()

  useEffect(() => {
    setDraft(committed)
  }, [committed])

  const changed = draft.trim() !== committed
  const trimmed = draft.trim()
  const resultMatches = testedUrl !== null && testedUrl === trimmed

  return (
    <div className={ROW_CLASS}>
      <FieldLabel
        field={field}
        help="Leave blank and Save to fall back to the configured default."
      />
      <input
        type="text"
        inputMode="url"
        aria-label={field.key}
        value={draft}
        disabled={disabled}
        placeholder="unset"
        onChange={(event) => {
          setDraft(event.target.value)
          // Drop any prior reachability result so a result for the old URL
          // never sits beside a newly-typed, untested one. The result render is
          // also gated on `resultMatches`, which covers a late-resolving request.
          test.reset()
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
          onPress={() => {
            setTestedUrl(trimmed)
            test.mutate({ service: field.service, url: trimmed })
          }}
        >
          {test.isPending ? "Testing…" : "Test"}
        </Button>
      </div>
      <div className={MESSAGE_CELL}>
        {/* aria-live so the reachability outcome is announced, not just shown. */}
        <span
          role="status"
          aria-live="polite"
          className="block break-words text-xs"
        >
          {test.isPending || !resultMatches ? null : test.error ? (
            <span className="text-danger">{errorMessage(test.error)}</span>
          ) : test.data ? (
            <span
              className={
                test.data.ok ? "font-medium text-success" : "text-danger"
              }
            >
              {test.data.reason}
            </span>
          ) : null}
        </span>
        <SaveError message={saveError} />
      </div>
    </div>
  )
}

// A nullable free-text row (engines, purpose hints). Empty clears to the default.
function TextRow({
  field,
  onSave,
  saveError,
  disabled,
}: {
  field: ToolSettingField
  onSave: (value: string | null) => void
  saveError?: string
  disabled: boolean
}) {
  const committed = typeof field.value === "string" ? field.value : ""
  const [draft, setDraft] = useState(committed)

  useEffect(() => {
    setDraft(committed)
  }, [committed])

  const changed = draft !== committed

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
  )
}

// A nullable integer row (max_results). Empty clears to the backend default.
function NumberRow({
  field,
  onSave,
  saveError,
  disabled,
}: {
  field: ToolSettingField
  onSave: (value: number | null) => void
  saveError?: string
  disabled: boolean
}) {
  const committed = typeof field.value === "number" ? String(field.value) : ""
  const [draft, setDraft] = useState(committed)

  useEffect(() => {
    setDraft(committed)
  }, [committed])

  const trimmed = draft.trim()
  const parsed = Number(trimmed)
  const valid = trimmed === "" || (Number.isInteger(parsed) && parsed >= 1)
  const changed = valid && trimmed !== committed

  return (
    <div className={ROW_CLASS}>
      <FieldLabel
        field={field}
        help="Leave blank to use the backend default."
      />
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
  )
}

// Per-call price for a tool Otari runs itself.
//
// The stored column is `input_price_per_million`, and `flat_request_cost` reads it
// as USD per *million* calls, so charging a cent a search means storing 10000. That
// convention is right for the wire and hostile at a keyboard, so this row speaks in
// dollars per call and does the 1e6 conversion itself. Same reason it lives here
// rather than on the Models page: a tool is not a model, and the Models editor is
// labeled per million tokens throughout.
const PER_MILLION = 1_000_000

function ToolPriceRow({
  pricingKey,
  configured,
  onSave,
  saving,
  saveError,
  disabled,
}: {
  pricingKey: string
  configured: number | null
  onSave: (perCall: number) => void
  saving: boolean
  saveError?: string
  disabled: boolean
}) {
  const committed = configured === null ? "" : String(configured / PER_MILLION)
  const [draft, setDraft] = useState(committed)

  useEffect(() => {
    setDraft(committed)
  }, [committed])

  const trimmed = draft.trim()
  const parsed = Number(trimmed)
  const valid = trimmed !== "" && Number.isFinite(parsed) && parsed >= 0
  const changed = valid && trimmed !== committed

  return (
    <div className={ROW_CLASS}>
      <div className="flex flex-col gap-0.5">
        <span className="text-body">Price per call</span>
        <span className="text-caption">
          {configured === null ? (
            <>
              Not priced. Calls are recorded but billed nothing, and with{" "}
              <code className="font-mono">require_pricing</code> on they are
              refused. Stored as <code className="font-mono">{pricingKey}</code>
              .
            </>
          ) : (
            <>
              Charged per call and added to the request that ran it. Stored as{" "}
              <code className="font-mono">{pricingKey}</code>.
            </>
          )}
        </span>
      </div>
      <div className="flex items-center gap-1.5 sm:col-start-2 sm:justify-self-end">
        <span className="text-caption">USD</span>
        <input
          type="number"
          min="0"
          step="0.0001"
          inputMode="decimal"
          aria-label={`Price per call for ${pricingKey}`}
          value={draft}
          disabled={disabled}
          placeholder="0.00"
          onChange={(event) => setDraft(event.target.value)}
          className={`w-full text-right tabular-nums sm:w-28 ${INPUT_CLASS}`}
        />
      </div>
      <div className={ACTIONS_CELL}>
        <Button
          size="sm"
          variant="primary"
          aria-label={`Save price for ${pricingKey}`}
          isDisabled={disabled || !changed || saving}
          onPress={() => onSave(parsed)}
        >
          {saving ? "Saving…" : "Save"}
        </Button>
      </div>
      {saveError ? (
        <div className={MESSAGE_CELL}>
          <SaveError message={saveError} />
        </div>
      ) : null}
    </div>
  )
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
  field: ToolSettingField
  onSave: (value: boolean | null) => void
  saveError?: string
  disabled: boolean
}) {
  // A tri-state select applies on change (like the toggles on the Settings page),
  // so a discrete choice needs no separate Save; text/number/url rows keep an
  // explicit Save because they have intermediate, typed-but-unsaved states.
  const current =
    field.value === true ? "on" : field.value === false ? "off" : "default"
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
  )
}

// What a client has to send to make the gateway run a tool. This exists because
// the contract is otherwise invisible: nothing on the page told an operator that
// `otari_web_search` is the type to declare, or that turning interception on also
// makes `web_search_20250305` work. Rendered from GET /v1/tools, so it reports
// what this deployment currently honors rather than a static example.
function HowToCallCard({ tool }: { tool: ManagedTool }) {
  const request = {
    model: "anthropic:claude-sonnet-4-6",
    messages: [{ role: "user", content: "..." }],
    tools: [tool.example],
  }
  return (
    <div className="flex flex-col gap-3 py-4">
      <div className="flex flex-wrap items-center gap-2">
        <code className="font-mono text-body">{tool.id}</code>
        {tool.available ? null : (
          // The reason, stated: a subtle dot and a mono fact, because a tool
          // with no backend is unavailable rather than broken.
          <span className="flex items-center gap-2 text-mono-overline text-subtle">
            <Dot className="bg-text-subtle" />
            Unavailable — no backend
          </span>
        )}
      </div>
      <p className="max-w-prose text-sm text-muted">{tool.description}</p>
      <div className="flex flex-col gap-1">
        <span className="text-xs font-medium text-foreground">
          Accepted tools[].type
        </span>
        <div className="flex flex-wrap items-center gap-x-1.5">
          {tool.accepted_types.map((type, index) => (
            // Mono on the page ground with a separator between entries, not a
            // boxed chip: these are values to read, and the box was the only
            // thing making a list of type names look like a set of controls.
            <span key={type} className="flex items-center gap-1.5">
              {index > 0 ? (
                <span aria-hidden className="text-subtle">
                  ·
                </span>
              ) : null}
              <code className="font-mono text-xs text-foreground">{type}</code>
            </span>
          ))}
        </div>
      </div>
      {/* Bounded by rules, not by a border. `border-control-border bg-surface`
          is the floating-surface recipe, which belongs to things that sit above
          the page (a popover, a menu); applied to a block that is part of the
          page it makes a card of it, which is the shape this whole direction
          removes. The rules say where the example starts and stops, and the
          block scrolls inside them. */}
      <pre className="overflow-x-auto border-y border-border py-3 text-xs">
        <code>{`POST /v1/chat/completions\n${JSON.stringify(request, null, 2)}`}</code>
      </pre>
    </div>
  )
}

// What a member is shown: the same field, as a value rather than as a control.
// A disabled input would be the cheaper change and the worse one, since it reads
// as a form that is briefly unavailable rather than as something this caller
// does not set. The service-endpoint fields never reach here at all: the server
// withholds them from a non-operator (otari-ai#1969).
function ReadOnlyRow({ field }: { field: ToolSettingField }) {
  const shown =
    field.value === null || field.value === ""
      ? "Default"
      : field.value === true
        ? "On"
        : field.value === false
          ? "Off"
          : String(field.value)
  return (
    <div className={ROW_CLASS}>
      <FieldLabel field={field} />
      <span className="break-words text-body sm:col-start-2">{shown}</span>
    </div>
  )
}

function ServiceRow({
  field,
  onSave,
  saveError,
  disabled,
  readOnly,
}: {
  field: ToolSettingField
  onSave: (value: boolean | number | string | null) => void
  saveError?: string
  disabled: boolean
  readOnly: boolean
}) {
  if (readOnly) {
    return <ReadOnlyRow field={field} />
  }
  if (field.type === "url") {
    return (
      <UrlRow
        field={field}
        onSave={onSave}
        saveError={saveError}
        disabled={disabled}
      />
    )
  }
  if (field.type === "int") {
    return (
      <NumberRow
        field={field}
        onSave={onSave}
        saveError={saveError}
        disabled={disabled}
      />
    )
  }
  if (field.type === "bool") {
    return (
      <BoolRow
        field={field}
        onSave={onSave}
        saveError={saveError}
        disabled={disabled}
      />
    )
  }
  return (
    <TextRow
      field={field}
      onSave={onSave}
      saveError={saveError}
      disabled={disabled}
    />
  )
}

/**
 * The tool and guardrail service settings, whole or narrowed to one service.
 *
 * `only` is what makes the sidebar's Tools group work: each child route renders
 * this page filtered to its own service rather than scrolling one long page.
 * Omitted, every service renders, which is the /tools route the group's parent
 * still points at.
 */
export function ToolsGuardrailsPage({ only }: { only?: ToolServiceName } = {}) {
  // The Tools group is member-visible for the workspace and organization cards
  // below. Of the deployment-wide reads above them, only the tool settings now
  // answer a tenant, without the service endpoints in them (otari-ai#1969), so
  // that one is asked unconditionally and rendered read-only. The pricing row
  // and the /v1/search tools stay operator-only on the server, so they are still
  // gated on the same answer the sidebar uses rather than fired into a 403.
  const organization = useOrganizationContext()
  const isOperator = isDeploymentOperator(organization.data)
  const query = useToolSettings()
  const tools = useTools(isOperator)
  const pricing = usePricing(isOperator)
  const setPricing = useSetPricing()
  const [pricedTool, setPricedTool] = useState<string | null>(null)
  const [priceErrors, setPriceErrors] = useState<Record<string, string>>({})

  // Latest rate per key. /v1/pricing is history-shaped (one row per effective_at),
  // and the newest row is the one in force.
  const currentRates = new Map<string, number>()
  for (const row of pricing.data ?? []) {
    const seen = currentRates.get(row.model_key)
    if (seen === undefined)
      currentRates.set(row.model_key, row.input_price_per_million)
  }

  const savePrice = (key: string, perCall: number) => {
    setPricedTool(key)
    setPriceErrors((prev) => ({ ...prev, [key]: "" }))
    setPricing.mutate(
      {
        model_key: key,
        // The stored convention is USD per million calls; the row above collects
        // dollars per call, so scale here and nowhere else.
        input_price_per_million: perCall * PER_MILLION,
        output_price_per_million: 0,
      },
      {
        onSuccess: () => {
          setPricedTool(null)
          showToast("Price saved")
        },
        onError: (error: unknown) => {
          setPricedTool(null)
          setPriceErrors((prev) => ({
            ...prev,
            [key]:
              error instanceof Error
                ? error.message
                : "Could not save the price",
          }))
        },
      },
    )
  }
  const update = useUpdateToolSettings()
  const [toast, showToast] = useSaveToast()
  const [errors, setErrors] = useState<Record<string, string>>({})

  const data = query.data
  const disabled = !data || update.isPending

  const byKey = new Map((data?.fields ?? []).map((field) => [field.key, field]))

  const save = (
    field: ToolSettingField,
    value: boolean | number | string | null,
  ) => {
    setErrors((prev) => {
      const { [field.key]: _removed, ...rest } = prev
      return rest
    })
    update.mutate(oneField(field.key, value), {
      onSuccess: () => showToast(`${field.key} saved`),
      onError: (error) =>
        setErrors((prev) => ({ ...prev, [field.key]: errorMessage(error) })),
    })
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="pb-1">
        <h1 className="text-display">
          {only
            ? (SERVICES.find((service) => service.key === only)?.label ??
              "Tools & Guardrails")
            : "Tools & Guardrails"}
        </h1>
        {/* Two readings: an operator configures the service endpoints, and a
            caller who does not is told what the deployment's tools do to their
            requests instead of how to configure a backend they cannot reach. */}
        <p className="mt-1 max-w-[620px] text-sm text-muted">
          {isOperator
            ? "Configure the built-in tool and guardrail service endpoints without a restart. Changes apply immediately and persist. URLs are validated for shape (http/https) and can be tested for reachability before saving; the network-safety gates for these services live on the Settings page."
            : "How this deployment's built-in tools behave on your requests, what your workspace may use of them, and what your organization mandates. The service backends themselves are a deployment operator's to configure."}
        </p>
      </header>

      <ErrorBanner error={query.error} />

      {query.isLoading ? <PageLoading /> : null}

      {SERVICES.filter((service) => !only || service.key === only).map(
        (service) => {
          const ordered = service.order
            .map((key) => byKey.get(key))
            .filter((f): f is ToolSettingField => f !== undefined)
          // Any field the backend reports for this service that isn't in `order`
          // (e.g. a newly added key) still renders, after the ordered ones, so a
          // backend addition is visible without a frontend change.
          const extra = (data?.fields ?? []).filter(
            (f) => f.service === service.key && !service.order.includes(f.key),
          )
          const fields = [...ordered, ...extra]
          // Absent while /v1/tools is still loading, or if it failed: the card is
          // reference material, so its absence must not hide the editable settings.
          const managed = service.toolId
            ? (tools.data?.data ?? []).find(
                (tool) => tool.id === service.toolId,
              )
            : undefined
          return (
            <Fragment key={service.key}>
              {/* The deployment-wide settings list exists only once the
                  operator-gated read produced this service's fields. The
                  workspace and organization cards below sit outside this
                  branch, because their audience is wider than the operator's
                  and they must not disappear with a read their viewer may not
                  make. */}
              {fields.length === 0 ? null : (
                <section className="flex flex-col gap-2">
                  {/* The shared settings list rather than a hand-rolled one, so
                      this cannot pick a separator tier of its own. The heading
                      is dropped on a filtered view, where the page title
                      already names the service and repeating it reads as two
                      headings for the same thing. */}
                  <SettingsGroup
                    title={only ? undefined : service.label}
                    description={service.blurb}
                  >
                    {/* Operator-only inside a list a member now also sees: the
                        rate comes from `/v1/pricing`, whose read is still
                        operator-gated, so a member would get an editable "Not
                        priced" row that can only fail on save. */}
                    {isOperator && service.pricingKey ? (
                      <ToolPriceRow
                        pricingKey={service.pricingKey}
                        configured={
                          currentRates.get(service.pricingKey) ?? null
                        }
                        onSave={(perCall) =>
                          savePrice(service.pricingKey as string, perCall)
                        }
                        saving={pricedTool === service.pricingKey}
                        saveError={
                          priceErrors[service.pricingKey] ||
                          (pricing.error
                            ? "Could not load the current price. Reload before editing."
                            : undefined)
                        }
                        // Also disabled when the load failed: an errored query leaves
                        // `configured` null, which renders as "Not priced" and would
                        // invite an operator to overwrite a rate they cannot see.
                        disabled={pricing.isLoading || Boolean(pricing.error)}
                      />
                    ) : null}
                    {fields.map((field) => (
                      <ServiceRow
                        key={field.key}
                        field={field}
                        onSave={(value) => save(field, value)}
                        saveError={errors[field.key]}
                        disabled={disabled}
                        readOnly={!isOperator}
                      />
                    ))}
                    {managed ? <HowToCallCard tool={managed} /> : null}
                  </SettingsGroup>
                </section>
              )}
              {/* Directly below the in-loop web-search settings, because a searxng
                search tool that declares no backend URL of its own inherits the
                one set just above it. Operator-only, like the settings above:
                its rows are the deployment's own /v1/search credentials. */}
              {/* The workspace card goes below both, because it narrows the
                backend set above it and the /v1/search tools beside it: a
                workspace switched off may use neither. */}
              {service.key === "web_search" ? (
                <>
                  {isOperator ? <SearchToolsCard onSaved={showToast} /> : null}
                  <WorkspaceWebSearchCard onSaved={showToast} />
                </>
              ) : null}
              {/* And directly below the sandbox settings, for the same reason:
                the workspace policy narrows the deployment-wide sandbox set
                just above it, and reads as nonsense apart from it. */}
              {service.key === "sandbox" ? (
                <WorkspaceCodeExecutionPolicyCard onSaved={showToast} />
              ) : null}
              {/* And directly below the guardrail settings, which are the
                deployment-wide half of the same feature: an entry with no
                endpoint of its own is sent to the URL set just above it. */}
              {service.key === "guardrails" ? (
                <OrganizationGuardrailsCard onSaved={showToast} />
              ) : null}
            </Fragment>
          )
        },
      )}

      {/* Beside the services rather than under one of them, and left out of
        every narrowed view, each of which is one service. `/tools/mcp-servers`
        renders the same card. */}
      {only ? null : <WorkspaceMcpServersCard />}

      <SaveToast message={toast} />
    </div>
  )
}
