import { Button, Spinner } from "@heroui/react"
import { Link } from "@tanstack/react-router"
import { type ReactNode, useEffect, useRef, useState } from "react"
import type {
  CreateStoredProviderRequest,
  ProviderHealth,
  ProviderInfo,
  StoredProvider,
  TestProviderResult,
  UpdateStoredProviderRequest,
} from "@/client"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { Dot } from "@/design-system/indicators/Dot"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { Section } from "@/design-system/layout/Section"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import {
  useOrganizationContext,
  useProviderKeyEncryption,
} from "@/shared/api/organizations"
import {
  useCreateStoredProvider,
  useDeleteStoredProvider,
  useProviderDetail,
  useProviderHealth,
  useProviders,
  useRecheckProviderHealth,
  useStoredProviders,
  useTestProviderCredentials,
  useTestStoredProvider,
  useUpdateStoredProvider,
} from "@/shared/api/providers"
import { useSettings, useUpdateSettings } from "@/shared/api/settings"
import { formatRelative } from "@/shared/helpers/format"

import {
  type CredentialFieldValues,
  credentialFieldsFor,
  credentialSpecFor,
  mergeCredentialFields,
  splitClientArgs,
  validateCredentialFields,
} from "./providerCredentialFields"
import {
  ClientArgsField,
  formatClientArgs,
  ProviderComboBox,
  ProviderCredentialFields,
  parseClientArgs,
} from "./providerFields"

// Testing the form's credentials before they are saved, in two nodes because
// they belong in two places: the button sits in the footer beside the submit,
// its outcome at the end of the body. The unverified case below is four lines
// plus the provider's own reply, and a footer that grows shoves the form up
// under the operator's hands; in the body it scrolls with everything else.
type ConnectionTestState = ReturnType<typeof useTestProviderCredentials>

// `getPayload` returns null when the minimum fields for a test are not filled
// in yet, which disables the button.
function ConnectionTestButton({
  test,
  getPayload,
}: {
  test: ConnectionTestState
  getPayload: () => CreateStoredProviderRequest | null
}) {
  const payload = getPayload()
  return (
    <Button
      variant="ghost"
      isDisabled={payload === null || test.isPending}
      onPress={() => {
        if (payload) test.mutate(payload)
      }}
    >
      {test.isPending ? "Testing…" : "Test connection"}
    </Button>
  )
}

function ConnectionTestResult({ test }: { test: ConnectionTestState }) {
  const answered = test.data ?? test.error
  const ref = useRef<HTMLSpanElement | null>(null)
  // The body scrolls, and this is its last child: on an `lg` dialog in an 800px
  // window the custom tab's own fields already fill it, so a verdict rendered
  // here can land below the fold with the footer button back to "Test
  // connection" and nothing else, to a sighted operator, having happened.
  useEffect(() => {
    if (answered) ref.current?.scrollIntoView({ block: "nearest" })
  }, [answered])
  return (
    // No wrapper, and `empty:hidden` rather than a conditional render: the
    // parent's `gap-4` would otherwise reserve a row before a test is ever run,
    // and the live region has to exist before it has something to say.
    <span
      ref={ref}
      role="status"
      aria-live="polite"
      className="flex flex-col gap-1.5 empty:hidden"
    >
      {test.isPending ? null : test.error ? (
        <span className="text-xs text-danger">{errorMessage(test.error)}</span>
      ) : test.data ? (
        test.data.ok ? (
          <span className="text-xs font-medium text-success">
            Connected. {test.data.model_count} model
            {test.data.model_count === 1 ? "" : "s"} available.
          </span>
        ) : test.data.discovery_unsupported ? (
          // No /v1/models on this backend: the test cannot confirm the key, but
          // it is not evidence the key is wrong either (issue #447). The error is
          // kept because this is the form where the operator just typed api_base,
          // and a wrong one 404s exactly like an absent listing endpoint.
          <span className="block max-w-md break-words text-xs text-warning">
            This provider does not list models, so the key could not be verified
            here. Save it and use the provider; declare its model ids under{" "}
            <code>models:</code> to have them show up in the catalog. If you did
            not expect this, check the provider's reply below.
            {test.data.error ? (
              <span className="mt-0.5 block text-muted">{test.data.error}</span>
            ) : null}
          </span>
        ) : (
          <span className="block max-w-md break-words text-caption text-danger">
            {test.data.error ?? "Connection failed."}
          </span>
        )
      ) : null}
    </span>
  )
}

// Add a hosted provider whose endpoint is built into the SDK: pick it, paste a
// key. Name and api_base are only exposed under Advanced.
function KnownProviderForm({
  isOpen,
  onClose,
  tabs,
}: {
  isOpen: boolean
  onClose: () => void
  tabs: ReactNode
}) {
  const create = useCreateStoredProvider()
  const test = useTestProviderCredentials()
  const [providerId, setProviderId] = useState("")
  const [apiKey, setApiKey] = useState("")
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [apiBase, setApiBase] = useState("")
  const [name, setName] = useState("")
  const [clientArgsText, setClientArgsText] = useState("")
  const [credentials, setCredentials] = useState<CredentialFieldValues>({})
  const clientArgs = parseClientArgs(clientArgsText)
  const credentialFields = credentialFieldsFor(providerId)
  const credentialErrors = validateCredentialFields(
    credentialFields,
    credentials,
  )
  const credentialSpec = credentialSpecFor(providerId)

  // Autofill hints are fetched lazily for just the selected provider, so the
  // picker itself never imports every provider SDK (issue #365).
  const detail = useProviderDetail(providerId)
  const selected = detail.data?.id === providerId ? detail.data : undefined
  // Prefill the (editable) API base with the provider's built-in default once its
  // detail loads, so Advanced shows what will be used. Keyed on the selected
  // provider so it fires once per selection and does not clobber later edits.
  useEffect(() => {
    if (selected) setApiBase(selected.default_api_base ?? "")
  }, [selected])
  const envKeyPresent = selected?.env_key_present ?? false
  // The key is only mandatory when the provider needs one and its env var is not
  // already set on the server; any-llm falls back to that env var otherwise.
  const needsKey = (selected?.requires_api_key ?? true) && !envKeyPresent
  const renamed = name.trim() !== "" && name.trim() !== providerId
  const nameHasDelimiter = /[:/]/.test(name)
  // Require the key when the chosen provider says it needs one; keyless local
  // backends (Ollama, llama.cpp) can submit without it.
  // One snapshot of everything the form owns, seeded on mount, rather than a
  // list of fields: the list was two of six, so Advanced's rename, API base,
  // client options and every typed credential (a Bedrock region) were invisible
  // to the guard and went on Escape with nothing asked. `key={addOpenCount}`
  // reseeds it per open. See feedback.md.
  const { isDirty } = useDirtySnapshot({
    providerId,
    apiKey,
    name,
    apiBase,
    clientArgsText,
    credentials,
  })
  const canSubmit =
    providerId !== "" &&
    !nameHasDelimiter &&
    (!needsKey || apiKey.trim() !== "") &&
    clientArgs.ok &&
    Object.keys(credentialErrors).length === 0
  // Hold the section open while something inside it is what's blocking submit,
  // so collapsing it can't leave a disabled button with its reason off screen.
  // A hide requested meanwhile is remembered and applies once the field is fixed.
  const advancedOpen = showAdvanced || !clientArgs.ok || nameHasDelimiter

  // The typed fields and the JSON textarea are two views of one `client_args`
  // object, so the request body is built in one place for both the save and the
  // connection test.
  const buildPayload = (): CreateStoredProviderRequest | null =>
    providerId === "" ||
    !clientArgs.ok ||
    Object.keys(credentialErrors).length > 0
      ? null
      : {
          instance: renamed ? name.trim() : providerId,
          // A renamed instance is no longer named after its provider, so record
          // the provider it is so routing still resolves.
          provider_type: renamed ? providerId : null,
          api_base: apiBase.trim() || null,
          api_key: apiKey.trim() || null,
          client_args: mergeCredentialFields(credentials, clientArgs.value),
        }

  const submit = () => {
    const payload = buildPayload()
    // `create.isPending` is a reason not to send twice, not a reason to draw
    // the primary as refused, so it guards the call rather than `canSubmit`.
    if (!canSubmit || create.isPending || payload === null) return
    create.mutate(payload, { onSuccess: onClose })
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      size="lg"
      title="New provider"
      tabs={tabs}
      submitLabel="Add provider"
      onSubmit={submit}
      isPending={create.isPending}
      isSubmitDisabled={!canSubmit}
      isDirty={isDirty}
      error={create.error}
      footerStart={
        <ConnectionTestButton test={test} getPayload={buildPayload} />
      }
    >
      <ProviderComboBox
        // The known tab is the dialog's default, so this is its first field.
        autoFocus
        label="Provider"
        value={providerId}
        onChange={(id) => {
          setProviderId(id)
          setName("")
          // Clear the API base; the effect above refills it from the provider's
          // built-in default once this provider's detail loads.
          setApiBase("")
          // The typed fields belong to the provider, so a change to it drops
          // values that no longer have a field to sit in.
          setCredentials({})
        }}
        description="Its endpoint is built in."
      />
      <SecretField
        value={apiKey}
        onChange={setApiKey}
        // The registry names the credential where the provider does not call it
        // an API key; the optional suffix still tracks whether one is needed.
        label={
          selected && !needsKey
            ? `${credentialSpec?.apiKeyLabel ?? "API key"} (optional)`
            : (credentialSpec?.apiKeyLabel ?? "API key")
        }
        description={[
          selected
            ? needsKey
              ? `${selected.name}'s endpoint is built in — just add your key.`
              : envKeyPresent
                ? `${selected.env_key} is set on the server, so a key is optional here. Paste one to override it.`
                : `${selected.name} needs no API key.`
            : "Stored encrypted. Requires OTARI_SECRET_KEY on the server.",
          credentialSpec?.apiKeyHelpText,
        ]
          .filter(Boolean)
          .join(" ")}
      />
      {/* Outside the Advanced disclosure below: a required field hidden behind
          a collapsed section is a submit button disabled for a reason off
          screen. */}
      <ProviderCredentialFields
        provider={providerId}
        values={credentials}
        onChange={setCredentials}
        errors={credentialErrors}
      />
      <button
        type="button"
        className="self-start text-xs font-medium text-link hover:text-link-hover"
        onClick={() => setShowAdvanced((v) => !v)}
      >
        {advancedOpen
          ? "Hide advanced"
          : "Advanced (API base, rename, client options)"}
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
                  <span className="text-danger">
                    A name cannot contain “:” or “/”.
                  </span>
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
      <ConnectionTestResult test={test} />
    </FormDialog>
  )
}

// Add a self-hosted or OpenAI-compatible endpoint: name it anything, say what
// API it speaks, and give the base URL (and a key if it needs one).
function CustomProviderForm({
  isOpen,
  onClose,
  tabs,
}: {
  isOpen: boolean
  onClose: () => void
  tabs: ReactNode
}) {
  const create = useCreateStoredProvider()
  const test = useTestProviderCredentials()
  const [name, setName] = useState("")
  const [providerType, setProviderType] = useState("openai-compatible")
  const [apiBase, setApiBase] = useState("")
  const [apiKey, setApiKey] = useState("")
  const [clientArgsText, setClientArgsText] = useState("")
  const clientArgs = parseClientArgs(clientArgsText)

  const nameHasDelimiter = /[:/]/.test(name)
  // Same snapshot as the known tab, for the same reason: this list had missed
  // `providerType` and the client options.
  const { isDirty } = useDirtySnapshot({
    name,
    providerType,
    apiBase,
    apiKey,
    clientArgsText,
  })
  const canSubmit =
    name.trim() !== "" &&
    !nameHasDelimiter &&
    apiBase.trim() !== "" &&
    clientArgs.ok

  const submit = () => {
    if (!canSubmit || create.isPending || !clientArgs.ok) return
    create.mutate(
      {
        instance: name.trim(),
        provider_type: providerType || "openai-compatible",
        api_base: apiBase.trim(),
        api_key: apiKey.trim() || null,
        client_args: clientArgs.value,
      },
      { onSuccess: onClose },
    )
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      size="lg"
      title="New provider"
      tabs={tabs}
      submitLabel="Add provider"
      onSubmit={submit}
      isPending={create.isPending}
      isSubmitDisabled={!canSubmit}
      isDirty={isDirty}
      error={create.error}
      footerStart={
        <ConnectionTestButton
          test={test}
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
      }
    >
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
              <span className="text-danger">
                A name cannot contain “:” or “/”.
              </span>
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
      <ConnectionTestResult test={test} />
    </FormDialog>
  )
}

type ProviderTab = "known" | "custom"

/**
 * The two ways to attach a provider, in one dialog.
 *
 * Each tab keeps its own mutation, its own validity and its own submit, which
 * is what it had as a panel. The two write the same five fields but derive them
 * from different questions: one asks which provider and fills the rest from its
 * built-in detail, the other asks for a name, an API flavor and a base URL. A
 * shared submit would be a switch on the tab, which is the same code with one
 * more place to look.
 *
 * What is shared is the frame. Each half renders the same `FormDialog` with the
 * same title, size and tab row, so switching tabs changes the fields and
 * nothing else, and a half-filled tab still does not survive a switch away from
 * it, which is what it did as a panel.
 */
function AddProviderForm({
  isOpen,
  onClose,
}: {
  isOpen: boolean
  onClose: () => void
}) {
  const [tab, setTab] = useState<ProviderTab>("known")
  const tabs = (
    <TabRow>
      {(
        [
          ["known", "Known provider"],
          ["custom", "Custom endpoint"],
        ] as const
      ).map(([id, label]) => (
        <Tab key={id} isActive={tab === id} onPress={() => setTab(id)}>
          {label}
        </Tab>
      ))}
    </TabRow>
  )

  return tab === "known" ? (
    <KnownProviderForm isOpen={isOpen} onClose={onClose} tabs={tabs} />
  ) : (
    <CustomProviderForm isOpen={isOpen} onClose={onClose} tabs={tabs} />
  )
}

function EditProviderForm({
  provider,
  onClose,
  onSaved,
}: {
  provider: StoredProvider
  onClose: () => void
  // Called with the saved instance when a save succeeds, so the page can retire
  // anything that described the credentials as they were. Distinct from onClose,
  // which also fires on cancel, where nothing was written and an existing verdict
  // still holds.
  onSaved: (instance: string) => void
}) {
  const update = useUpdateStoredProvider()
  const [providerType, setProviderType] = useState(provider.provider_type ?? "")
  const [apiBase, setApiBase] = useState(provider.api_base ?? "")
  const [replacingKey, setReplacingKey] = useState(false)
  const [apiKey, setApiKey] = useState("")
  // An instance keeps its provider's name unless it was renamed, so the
  // instance is what says which provider this is when provider_type is unset.
  // Read the same way below, so the fields rendered are the ones the stored
  // options were split against.
  const providerId = providerType.trim() || provider.instance
  const [stored] = useState(() =>
    splitClientArgs(
      credentialFieldsFor(provider.provider_type?.trim() || provider.instance),
      provider.client_args,
    ),
  )
  const [clientArgsText, setClientArgsText] = useState(() =>
    formatClientArgs(stored.rest),
  )
  const [credentials, setCredentials] = useState<CredentialFieldValues>(
    () => stored.typed,
  )
  const clientArgs = parseClientArgs(clientArgsText)
  const credentialFields = credentialFieldsFor(providerId)
  const credentialErrors = validateCredentialFields(
    credentialFields,
    credentials,
    stored.redacted,
  )

  const submit = () => {
    if (
      update.isPending ||
      !clientArgs.ok ||
      Object.keys(credentialErrors).length > 0
    )
      return
    const body: UpdateStoredProviderRequest = {
      provider_type: providerType.trim() || null,
      api_base: apiBase.trim() || null,
      // Sent on every save, so emptying the field clears the stored options.
      client_args: mergeCredentialFields(
        credentials,
        clientArgs.value,
        stored.redacted,
      ),
      // Guard against clobbering a concurrent edit; a 412 tells the operator to reload.
      expected_updated_at: provider.updated_at,
    }
    if (replacingKey && apiKey.trim()) {
      body.api_key = apiKey.trim()
    }
    update.mutate(
      { instance: provider.instance, body },
      {
        onSuccess: () => {
          onSaved(provider.instance)
          onClose()
        },
      },
    )
  }

  return (
    <Section
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      <div className="text-title">
        Edit <code>{provider.instance}</code>
      </div>
      <ErrorBanner error={update.error} />
      <div className="grid gap-4 sm:grid-cols-2">
        <Field
          label="Provider type"
          value={providerType}
          onChange={setProviderType}
          placeholder="openai"
        />
        <Field
          label="API base"
          value={apiBase}
          onChange={setApiBase}
          placeholder="https://api.openai.com/v1"
        />
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
              className="self-start text-xs font-medium text-link hover:text-link-hover"
              onClick={() => {
                setReplacingKey(false)
                setApiKey("")
              }}
            >
              Keep the current key
            </button>
          </>
        ) : (
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted">
              API key:{" "}
              <code>
                {provider.last4 ? `••••${provider.last4}` : "none set"}
              </code>
            </span>
            <Button
              size="sm"
              variant="ghost"
              onPress={() => setReplacingKey(true)}
            >
              Replace key
            </Button>
          </div>
        )}
      </div>
      <ProviderCredentialFields
        provider={providerId}
        values={credentials}
        onChange={setCredentials}
        errors={credentialErrors}
        redacted={stored.redacted}
      />
      <ClientArgsField
        value={clientArgsText}
        onChange={setClientArgsText}
        error={clientArgs.ok ? null : clientArgs.error}
      />
      <div className="flex gap-2">
        <Button
          variant="primary"
          isDisabled={
            update.isPending ||
            !clientArgs.ok ||
            Object.keys(credentialErrors).length > 0
          }
          onPress={submit}
        >
          {update.isPending ? "Saving…" : "Save changes"}
        </Button>
        <Button variant="ghost" onPress={onClose}>
          Cancel
        </Button>
      </div>
    </Section>
  )
}

interface ProviderRow {
  instance: string
  source: "config" | "stored"
  stored: StoredProvider | undefined
  meta: ProviderInfo | undefined
}

function buildRows(
  meta: ProviderInfo[] | undefined,
  stored: StoredProvider[] | undefined,
): ProviderRow[] {
  const storedByInstance = new Map((stored ?? []).map((p) => [p.instance, p]))
  const metaByInstance = new Map((meta ?? []).map((p) => [p.instance, p]))
  const instances = new Set<string>([
    ...storedByInstance.keys(),
    ...metaByInstance.keys(),
  ])
  return [...instances].sort().map((instance) => {
    const s = storedByInstance.get(instance)
    return {
      instance,
      source: s ? "stored" : "config",
      stored: s,
      meta: metaByInstance.get(instance),
    } as const
  })
}

type TestState =
  | { status: "pending" }
  | ({ status: "done" } & TestProviderResult)

function TestOutcome({ state }: { state: TestState | undefined }) {
  if (!state) return null
  if (state.status === "pending") {
    return (
      <span className="inline-flex items-center gap-1.5 text-caption">
        {/* Hidden rather than left as HeroUI's own role="status": its
            "Loading" name would contradict the "Testing…" beside it. */}
        <Spinner size="sm" aria-hidden="true" /> Testing…
      </span>
    )
  }
  if (state.ok) {
    return (
      <span className="text-xs font-medium text-success">
        Connected. {state.model_count} model{state.model_count === 1 ? "" : "s"}{" "}
        available.
      </span>
    )
  }
  // A backend with no model-listing endpoint cannot be verified this way, but the
  // key is not therefore wrong: say so instead of reporting a failed connection.
  // The provider error stays visible underneath, because a 404 is also what a
  // wrong api_base returns, and that is a misconfiguration to fix, not to reassure
  // away.
  if (state.discovery_unsupported) {
    return (
      <span className="block max-w-xs break-words text-xs text-warning">
        Could not list models, so the key could not be verified. It may still
        work for requests.
        {state.error ? (
          <span className="mt-0.5 block text-muted">{state.error}</span>
        ) : null}
      </span>
    )
  }
  return (
    <span className="block max-w-xs break-words text-caption text-danger">
      {state.error ?? "Connection failed."}
    </span>
  )
}

// A provider's reachability, from the shared model-discovery health path. Config
// providers (no per-row Test button) get a status here too, not just stored ones.
// Semantic status surface: raw Tailwind palette classes, matching TestOutcome and
// ErrorBanner rather than the page's own chrome. A provider that answers no model
// listing (its backend never implemented /v1/models) is not unreachable: only
// discovery is broken, and it may still serve requests, so it gets the amber
// warning state rather than the red one (issue #447).
// A repeated column's words, in their own casing. The uppercase these carried
// was emphasis applied to every row alike, including the reachable ones, which
// is the state a column watching for failures wants to draw the eye to least.
// Same treatment, and same reasoning, as Activity's status column.
const HEALTH_LABELS = {
  ok: "Reachable",
  degraded: "No model discovery",
  unreachable: "Unreachable",
} as const

function HealthPill({ health }: { health: ProviderHealth | undefined }) {
  if (!health) {
    return <span className="text-caption">—</span>
  }
  const degraded = !health.ok && health.discovery_unsupported
  // Only a real failure colors its text. Degraded is a provider that answers
  // requests but lists no models, which is a fact about discovery rather than an
  // outage, so it reads on the muted rung with the danger dot that says "worth
  // noticing" without the ink that says "broken".
  const styles = health.ok || degraded ? "text-muted" : "text-danger"
  const dot = health.ok ? "bg-success" : "bg-danger"
  // The last-checked time lives in the top summary banner; the row just shows the
  // status. The error (and time) stay available on hover as the pill's tooltip.
  const checked = health.checked_at
    ? `Last checked ${formatRelative(health.checked_at)}`
    : "Not checked yet"
  const reason = degraded
    ? `${health.error ?? "This provider does not list models."} Requests to it may still work.`
    : (health.error ?? "Unreachable")
  const title = health.ok ? checked : `${reason} · ${checked}`
  return (
    <span
      title={title}
      className={`flex items-center gap-2 text-mono-caption ${styles}`}
    >
      <Dot className={dot} />
      {HEALTH_LABELS[health.ok ? "ok" : degraded ? "degraded" : "unreachable"]}
    </span>
  )
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
  healthy: number
  degraded: number
  total: number
  checkedAt: string | null
}) {
  const allHealthy = healthy === total
  const dot = allHealthy
    ? "bg-success"
    : healthy + degraded === total
      ? "bg-warning"
      : "bg-danger"
  const recheck = useRecheckProviderHealth()
  return (
    // A band of the page rather than a card, which is what the rest of this page
    // became: the summary is a region between rules, so the rules are what bound
    // it and the fill goes.
    <Section
      className="border-y border-border py-3"
      contentClassName="flex flex-wrap items-center gap-3 text-sm"
    >
      <Dot className={dot} />
      <span className="font-medium text-foreground">
        {healthy} of {total} provider{total === 1 ? "" : "s"} reachable
      </span>
      {degraded > 0 ? (
        <span className="text-warning">{degraded} without model discovery</span>
      ) : null}
      {checkedAt ? (
        <span className="text-muted">
          Last checked {formatRelative(checkedAt)}
        </span>
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
    </Section>
  )
}

function Step({
  n,
  title,
  children,
}: {
  n: number
  title: string
  children: ReactNode
}) {
  return (
    <li className="flex gap-3">
      <span className="flex h-6 w-6 shrink-0 items-center justify-center bg-primary-subtle text-xs font-semibold text-primary-subtle-foreground">
        {n}
      </span>
      <div className="text-sm">
        <div className="font-medium text-foreground">{title}</div>
        <div className="text-muted">{children}</div>
      </div>
    </li>
  )
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
  onAddProvider: () => void
  needsPricing: boolean
  onEnablePricing: () => void
  enabling: boolean
  secretKeyConfigured: boolean
}) {
  return (
    <Section
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      <div className="flex items-start gap-3">
        <Dot className="mt-2.5 bg-accent" />
        <div>
          <h2 className="text-display-sub">Welcome to Otari</h2>
          <p className="mt-1 text-sm text-muted">
            You are signed in. Add a provider to start serving models: three
            quick steps.
          </p>
        </div>
      </div>
      <ol className="flex flex-col gap-3">
        <Step n={1} title="Add a provider">
          Enter a provider name (like <code>openai</code>) and its API key. Keys
          are encrypted at rest.
        </Step>
        <Step n={2} title="Test the connection">
          Use <strong>Test</strong> on the provider row to confirm the key works
          and see how many models it serves.
        </Step>
        <Step n={3} title="Send your first request">
          Once a provider exists, the <strong>Overview</strong> page usually
          offers a setup guide that hands you an API key and the call to make.
          Either way, point your app at <code>/v1</code> on this gateway with an
          API key (the one printed in the server logs starts <code>gw-…</code>).
          See the{" "}
          {/* /welcome is served by the gateway itself, not by a client route, so this
                stays a plain path anchor: a router Link would resolve to /#/welcome, which
                the catch-all route sends back to the overview. It leaves the SPA, so open a
                new tab and the operator keeps the dashboard (as the guide's links do). */}
          <a
            href="/welcome"
            target="_blank"
            rel="noreferrer"
            className="font-medium text-link hover:text-link-hover"
          >
            quickstart
          </a>
          .
        </Step>
      </ol>
      {needsPricing ? (
        <p className="text-sm text-muted">
          Tip: <code>require_pricing</code> is on, so requests are rejected
          until pricing is set.{" "}
          <button
            type="button"
            className="font-medium text-link hover:text-link-hover disabled:opacity-(--disabled-opacity)"
            disabled={enabling}
            onClick={onEnablePricing}
          >
            Enable default pricing
          </button>{" "}
          to meter new models with public rates.
        </p>
      ) : null}
      <div>
        {/* Disabled at 0.4 rather than hidden. Blocked is not the same as
              empty: an operator whose server has no secret key has to see the
              action they are being denied, or the page reads as though adding a
              provider is not a thing this product does. */}
        <Button
          variant="primary"
          isDisabled={!secretKeyConfigured}
          onPress={onAddProvider}
        >
          Add your first provider
        </Button>
      </div>
    </Section>
  )
}

export function ProvidersPage() {
  const meta = useProviders()
  const stored = useStoredProviders()
  const context = useOrganizationContext()
  const settings = useSettings()
  const health = useProviderHealth()
  const deleteProvider = useDeleteStoredProvider()
  const testProvider = useTestStoredProvider()
  const updateSettings = useUpdateSettings()

  const [addOpen, setAddOpen] = useState(false)
  const [addOpenCount, setAddOpenCount] = useState(0)
  const [editing, setEditing] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<string>()
  const [tests, setTests] = useState<Record<string, TestState>>({})
  // `addOpenCount` above is bumped on each open, and the add form is keyed on
  // it, so the draft (a pasted provider key included) is fresh every time and
  // untouched through the exit: the dialog keeps its content while it animates
  // out, so clearing on the way out would blank the body in front of the
  // operator. See feedback.md, "A draft is fresh on every open and untouched
  // through the exit". Both openers on this page go through here.
  const openAdd = () => {
    setEditing(null)
    setAddOpenCount((n) => n + 1)
    setAddOpen(true)
  }

  const rows = buildRows(meta.data?.providers, stored.data)
  const healthByInstance = new Map(
    (health.data?.providers ?? []).map((item) => [item.instance, item]),
  )
  const loading = meta.isLoading || stored.isLoading
  const editingProvider =
    stored.data?.find((p) => p.instance === editing) ?? null
  const needsPricing =
    settings.data?.require_pricing === true &&
    settings.data.default_pricing === false
  // Gate adding providers on the server having OTARI_SECRET_KEY, which the
  // membership context reports and `/settings` no longer answers for every
  // caller who reaches this page (#839).
  const secretKeyConfigured = useProviderKeyEncryption()
  // Not gated on `addOpen`: unmounting the first-run panel when the dialog
  // opens takes away the node react-aria restores focus to, so closing drops
  // focus to `<body>`. The heading's action is ungated for the same reason.
  const showOnboarding = !loading && rows.length === 0

  // Which test run each row is currently showing. A row's result is only worth
  // recording while it is still the answer to the newest thing the operator asked
  // for, and neither the pending marker nor the Test button's disabled state can
  // establish that on its own: the button is per-row and keys off the marker
  // (below), so clearing the marker re-enables it in the same instant, and a
  // retest can start while the previous request is still in flight. A monotonic
  // id per instance is the thing that actually settles it. In a ref because a
  // late callback must read the current value, not the one its render closed over.
  const testRuns = useRef<Record<string, number>>({})
  const startTestRun = (instance: string) => {
    const run = (testRuns.current[instance] ?? 0) + 1
    testRuns.current[instance] = run
    return run
  }

  // A test verdict describes the credentials as they were when it ran, so drop
  // it once the provider changes underneath it: otherwise the row keeps showing
  // a failure from the previous configuration, contradicting the status pill
  // right above it (issue #464). Bumping the run id retires any request still in
  // flight, so a slow test cannot write the old verdict back afterwards.
  const clearTest = (instance: string) => {
    startTestRun(instance)
    setTests((prev) => {
      // Own-property check: `in` also reports inherited keys, so an instance
      // named after one (`toString`) would read as having a verdict it does not.
      if (!Object.hasOwn(prev, instance)) return prev
      const next = { ...prev }
      delete next[instance]
      return next
    })
  }

  // Record a result only if its run is still the current one for that row, which
  // rules out both a verdict retired by an edit or delete and one superseded by a
  // later test on the same row.
  const settleTest = (instance: string, run: number, state: TestState) => {
    if (testRuns.current[instance] !== run) return
    setTests((prev) => ({ ...prev, [instance]: state }))
  }

  // Resolve each row's test from its own promise. One useMutation observer serves
  // every row, and TanStack Query detaches it from the previous mutation as soon
  // as the next `mutate` lands, discarding that call's onSuccess/onError: testing
  // a second provider while the first was still in flight left the first row
  // spinning on "Testing…", with its Test button disabled, for the life of the
  // page. The promise from mutateAsync settles regardless of that detach.
  const runTest = async (instance: string) => {
    const run = startTestRun(instance)
    setTests((prev) => ({ ...prev, [instance]: { status: "pending" } }))
    try {
      const result = await testProvider.mutateAsync(instance)
      settleTest(instance, run, { status: "done", ...result })
    } catch (error) {
      settleTest(instance, run, {
        status: "done",
        ok: false,
        model_count: 0,
        error: errorMessage(error),
        discovery_unsupported: false,
      })
    }
  }

  const columns: DataTableColumn<ProviderRow>[] = [
    {
      id: "provider",
      header: "Provider",
      isRowHeader: true,
      cell: (row) => (
        <Link
          to="/models"
          search={{ provider: row.instance }}
          className="font-medium text-foreground hover:text-link hover:underline"
        >
          {row.instance}
        </Link>
      ),
    },
    {
      id: "type",
      header: "Type",
      cell: (row) => (
        <span className="text-muted">
          {row.meta?.provider_type ?? row.stored?.provider_type ?? row.instance}
        </span>
      ),
    },
    {
      id: "source",
      header: "Source",
      cell: (row) => (
        // A CATEGORY rather than a state: the vocabulary is closed (stored or
        // config) and a row's answer never changes on its own. Categories carry
        // no dot, and this one's was worse than decorative: an accent mark in a
        // column of statuses reads as a state the column does not carry. The
        // uppercase stays, which is what separates a category from the states
        // in the column beside it.
        <span className="text-mono-caption text-muted">
          {row.source === "stored" ? "STORED" : "CONFIG"}
        </span>
      ),
    },
    {
      id: "api_key",
      header: "API key",
      cell: (row) => (
        <span className="text-muted">
          {row.source === "stored" ? (
            row.stored && !row.stored.decryptable ? (
              <span
                className="text-warning"
                title="This key can't be decrypted with the current OTARI_SECRET_KEY. Replace the key, or restore the original OTARI_SECRET_KEY."
              >
                ⚠ key unreadable
              </span>
            ) : (
              <code>
                {row.stored?.last4 ? `••••${row.stored.last4}` : "none set"}
              </code>
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
    {
      id: "status",
      header: "Status",
      cell: (row) => <HealthPill health={healthByInstance.get(row.instance)} />,
    },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) =>
        row.source === "stored" ? (
          <div className="flex flex-col items-end gap-1.5">
            <RowActionRow>
              <RowAction
                // A row whose key can't be decrypted can't be tested; Edit/Delete still recover it.
                isDisabled={
                  tests[row.instance]?.status === "pending" ||
                  row.stored?.decryptable === false
                }
                onPress={() => void runTest(row.instance)}
              >
                Test
              </RowAction>
              <RowAction
                onPress={() => {
                  setAddOpen(false)
                  setEditing(row.instance)
                }}
              >
                Edit
              </RowAction>
              <RowAction onPress={() => setPendingDelete(row.instance)}>
                Delete
              </RowAction>
            </RowActionRow>
            <TestOutcome state={tests[row.instance]} />
          </div>
        ) : (
          <span className="block text-right text-caption">
            managed in config.yml
          </span>
        ),
    },
  ]

  return (
    <div className="flex flex-col gap-6">
      <PageIntro
        title="Providers"
        action={
          <Button
            // Visible while the dialog is open and beside the first-run
            // panel's own copy of it: the dialog is over the page. Disabled
            // rather than hidden without a server secret key, which is the
            // rule for a control that carries its own reason nearby.
            variant="primary"
            isDisabled={!secretKeyConfigured}
            onPress={openAdd}
          >
            Add provider
          </Button>
        }
      >
        Add provider API keys here to serve models without editing config.yml.
        Keys are encrypted at rest.
      </PageIntro>

      {/* `settings.error` is deliberately absent. The page reads that endpoint
          only for the pricing hint below, and it is operator-only, so a caller
          who may manage providers but is not a deployment operator would carry
          a permanent "Not authorized" alert for a read nothing here depends on.
          `context.error` takes its place, because the add control now gates on
          the membership context; a write that fails still reports itself
          through `updateSettings.error`.

          The cost is that a genuine 500 from that endpoint is silent too, and
          `needsPricing` then reads false, so the pricing hint disappears with
          nothing saying why. Taken deliberately: the alternative shows every
          non-operator a permanent error for a read they were never entitled to
          make, and the hint is an advisory nudge rather than a control. Telling
          the two apart needs the page to distinguish a 403 from a 5xx, which is
          worth doing when the hint earns it. */}
      <ErrorBanner
        error={
          meta.error ??
          stored.error ??
          context.error ??
          health.error ??
          updateSettings.error
        }
      />

      {/* A band of the page, not a tinted box: a square danger dot, the setting
          named in mono so it is copyable by eye, and the consequence in muted
          prose. No fill, no border, no radius. */}
      {/* `context.data &&` as well as the flag: a context read that failed
          leaves the flag falsy, and claiming the key is unset would be a guess,
          and the wrong one whenever the deployment has one. The error banner
          above already names the read that actually failed. */}
      {context.data && !secretKeyConfigured ? (
        <Section
          className="border-y border-border py-3"
          contentClassName="flex items-start gap-3 text-sm"
        >
          <Dot className="mt-2 bg-danger" />
          <p className="text-muted">
            <span className="text-mono-caption text-foreground">
              OTARI_SECRET_KEY
            </span>{" "}
            is not set, so provider keys can&rsquo;t be encrypted at rest and
            adding providers from the dashboard is disabled. Set it on the
            server and restart to add providers here. Providers defined in{" "}
            <span className="text-mono-caption">config.yml</span> keep working
            without it.
          </p>
        </Section>
      ) : null}

      {showOnboarding ? (
        <OnboardingPanel
          onAddProvider={openAdd}
          needsPricing={needsPricing}
          onEnablePricing={() =>
            updateSettings.mutate({ default_pricing: true })
          }
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
      <AddProviderForm
        key={addOpenCount}
        isOpen={addOpen && secretKeyConfigured}
        onClose={() => setAddOpen(false)}
      />
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
        <TableScrollFrame className="otari-providers-table">
          <DataTable
            ariaLabel="Providers"
            columns={columns}
            rows={rows}
            getRowKey={(row) => row.instance}
            isLoading={loading}
            emptyContent="No providers yet. Add your first provider to start serving models."
          />
        </TableScrollFrame>
      )}

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        // Cleared on the way out: a refusal otherwise sits on the mutation and
        // greets the next row's confirm as if that row had failed.
        onOpenChange={(open) => {
          if (open) return
          setPendingDelete(undefined)
          deleteProvider.reset()
        }}
        heading="Delete provider"
        body={
          pendingDelete
            ? `${pendingDelete} and the credential stored with it are removed. A request routed to it fails until another provider serves its models.`
            : null
        }
        confirmLabel="Delete provider"
        isPending={deleteProvider.isPending}
        error={deleteProvider.error}
        onConfirm={() => {
          if (pendingDelete === undefined) return
          deleteProvider.mutate(pendingDelete, {
            // Clear the verdict too: a provider re-added under the same name is
            // a different provider, and would otherwise inherit it.
            onSuccess: () => {
              clearTest(pendingDelete)
              setPendingDelete(undefined)
            },
          })
        }}
      />
    </div>
  )
}
