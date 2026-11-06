import { useState } from "react"
import { FiChevronDown } from "react-icons/fi"

import type {
  CreateOrgWebSearchKeyRequest,
  OrgWebSearchKey,
  SearchProviderInfo,
  UpdateOrgWebSearchKeyRequest,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { ProviderOptionFields } from "@/features/webSearch/ProviderOptionFields"
import {
  buildOptions,
  sameOptions,
  splitOptions,
  type WebToolKind,
} from "@/features/webSearch/webTools"
import {
  useCreateOrgWebSearchKey,
  useUpdateOrgWebSearchKey,
} from "@/shared/api/webSearchKeys"

import { KIND_COPY, offeredProviders } from "./webSearchKeys"

// Adds one of the organization's search or fetch keys, or edits one. The
// provider comes first and decides the rest: what it requires, then its
// endpoint and options under "More settings", typed from the catalog's schema.

interface Draft {
  provider: string
  name: string
  apiKey: string
  apiBase: string
  options: Record<string, string>
  extraJson: string
}

function initialDraft(
  editing: OrgWebSearchKey | undefined,
  provider: SearchProviderInfo | undefined,
): Draft {
  const split = splitOptions(provider, editing?.options ?? {})
  return {
    provider: editing?.provider ?? "",
    name: editing?.name ?? "",
    // Never prefilled: the gateway returns only the last four characters.
    apiKey: "",
    apiBase: editing?.api_base ?? "",
    options: split.typed,
    extraJson: split.extraJson,
  }
}

export function WebSearchKeyDialog({
  kind,
  editing,
  providers,
  isOpen = true,
  onClose,
}: {
  kind: WebToolKind
  /** The key being edited; absent to add one. */
  editing?: OrgWebSearchKey
  /** The provider catalog for this kind. */
  providers: readonly SearchProviderInfo[]
  /** Kept mounted while closed, so the dialog plays its exit. */
  isOpen?: boolean
  onClose: () => void
}) {
  const isEdit = editing !== undefined
  const copy = KIND_COPY[kind]
  const create = useCreateOrgWebSearchKey()
  const update = useUpdateOrgWebSearchKey()
  const [draft, setDraft] = useState<Draft>(() =>
    initialDraft(
      editing,
      providers.find((entry) => entry.id === editing?.provider),
    ),
  )
  const [isMoreOpen, setIsMoreOpen] = useState(false)
  const { isDirty } = useDirtySnapshot(draft)
  const set = (patch: Partial<Draft>) => setDraft({ ...draft, ...patch })

  const provider = providers.find((entry) => entry.id === draft.provider)
  // A key on a provider the catalog does not describe, such as a Tavily key
  // added through the API, keeps only its name and secret editable here.
  const isUncataloged = isEdit && provider === undefined
  const apiBase = draft.apiBase.trim()
  const movesHost =
    isEdit && apiBase !== "" && apiBase !== (editing.api_base ?? "")
  // The next call sends the stored secret to the new host, so the gateway
  // refuses the move unless the secret comes with it.
  const needsKeyAgain = movesHost && Boolean(editing?.last4)
  const keyRequired =
    (!isEdit && Boolean(provider?.requires_api_key)) || needsKeyAgain
  // A provider that requires an endpoint has none of its own. For an operator,
  // the catalog gives it the endpoint the deployment's tools inherit, which a
  // key never does, so the key needs its own.
  const baseRequired = Boolean(provider?.requires_api_base)
  const defaultBase = baseRequired ? "" : (provider?.default_api_base ?? "")
  const built = buildOptions(provider, draft.options, draft.extraJson)
  const isReady =
    (isEdit || provider !== undefined) &&
    draft.name.trim() !== "" &&
    (!keyRequired || draft.apiKey !== "") &&
    (!baseRequired || apiBase !== "") &&
    (isUncataloged || built.ok)

  const options = built.ok ? built.options : {}

  const submit = () => {
    if (!editing) {
      const body: CreateOrgWebSearchKeyRequest = {
        kind,
        provider: draft.provider,
        name: draft.name.trim(),
        api_key: draft.apiKey === "" ? null : draft.apiKey,
        api_base: apiBase === "" ? null : apiBase,
        options: Object.keys(options).length > 0 ? options : null,
      }
      create.mutate(body, { onSuccess: onClose })
      return
    }
    // Only what changed: the gateway checks what a request sets, so an edit
    // never trips over a stored option a library upgrade left behind.
    const body: UpdateOrgWebSearchKeyRequest = {}
    if (draft.name.trim() !== editing.name) body.name = draft.name.trim()
    if (draft.apiKey !== "") body.api_key = draft.apiKey
    if (!isUncataloged) {
      if (apiBase !== (editing.api_base ?? "")) {
        body.api_base = apiBase === "" ? null : apiBase
      }
      if (!sameOptions(options, editing.options ?? {})) {
        body.options = Object.keys(options).length > 0 ? options : null
      }
    }
    update.mutate({ keyId: editing.id, body }, { onSuccess: onClose })
  }

  const endpoint = (
    <Field
      label="API base URL"
      value={draft.apiBase}
      onChange={(next) => set({ apiBase: next })}
      isRequired={baseRequired}
      placeholder={defaultBase || "https://…"}
      description={[
        defaultBase
          ? `Optional. Blank uses ${defaultBase}.`
          : baseRequired
            ? "This provider has no endpoint of its own, so it needs one here."
            : "Optional.",
        isEdit
          ? "A new URL needs the API key, and any hidden option below, entered again, because the key will be sent there."
          : "",
      ]
        .filter(Boolean)
        .join(" ")}
    />
  )

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      size="lg"
      title={isEdit ? `Edit ${copy.noun}` : `Add ${copy.noun}`}
      description={isEdit ? editing.name : copy.serves}
      submitLabel={isEdit ? "Save" : `Add ${copy.noun}`}
      onSubmit={submit}
      isPending={create.isPending || update.isPending}
      isSubmitDisabled={!isReady}
      isDirty={isDirty}
      error={create.error ?? update.error}
    >
      {editing ? (
        <p className="text-caption">
          A {kind} key on <code className="font-mono">{editing.provider}</code>.
          Its kind and provider are fixed: add another key to use another
          provider.
        </p>
      ) : (
        <Select
          label="Provider"
          value={draft.provider}
          onChange={(next) =>
            // The endpoint and options belong to the provider, so a change
            // clears them: another provider may refuse them.
            set({ provider: next, apiBase: "", options: {}, extraJson: "" })
          }
          placeholder="Choose a provider"
          options={offeredProviders(providers).map((entry) => ({
            value: entry.id,
            label: entry.id,
          }))}
          description={
            provider?.doc_url ? (
              <a
                href={provider.doc_url}
                target="_blank"
                rel="noreferrer"
                className="text-link hover:text-link-hover"
              >
                {provider.id}&rsquo;s API documentation
              </a>
            ) : undefined
          }
          autoFocus
        />
      )}

      {isEdit || provider ? (
        <>
          {kind === "fetch" && provider?.reports_final_url === false ? (
            <InfoBanner tone="warning">
              {provider.id} does not report where a redirect led, so a
              workspace&rsquo;s allowed and blocked domains are not checked past
              one.
            </InfoBanner>
          ) : null}
          <Field
            label="Name"
            value={draft.name}
            onChange={(next) => set({ name: next })}
            isRequired
            placeholder="production"
            description={`Unique per provider, so a second ${draft.provider} key needs another name.`}
          />
          <SecretField
            label="API key"
            value={draft.apiKey}
            onChange={(apiKey) => set({ apiKey })}
            isRequired={keyRequired}
            placeholder={
              editing?.last4 ? `Keep ••••${editing.last4}` : undefined
            }
            description={
              needsKeyAgain
                ? "Enter the API key again: it goes to the new URL."
                : isEdit
                  ? "Encrypted at rest and never shown again. Leave blank to keep the current key."
                  : "Encrypted at rest and never shown again; only the last 4 characters come back."
            }
          />
          {isUncataloged ? (
            <p className="text-caption">
              This provider runs on the older search code, which takes no
              endpoint or options.
            </p>
          ) : (
            <>
              {baseRequired ? endpoint : null}
              <div>
                <Button
                  size="sm"
                  variant="ghost"
                  aria-expanded={isMoreOpen}
                  onPress={() => setIsMoreOpen(!isMoreOpen)}
                >
                  {`More settings for ${draft.provider}`}
                  <FiChevronDown
                    aria-hidden
                    className={`transition-transform ${isMoreOpen ? "rotate-180" : ""}`}
                  />
                </Button>
              </div>
              {isMoreOpen ? (
                <div className="flex flex-col gap-4">
                  {baseRequired ? null : endpoint}
                  <ProviderOptionFields
                    provider={provider}
                    values={draft.options}
                    errors={built.ok ? {} : built.fieldErrors}
                    extraJson={draft.extraJson}
                    jsonError={built.ok ? "" : built.jsonError}
                    onChange={(next) => set({ options: next })}
                    onExtraJsonChange={(extraJson) => set({ extraJson })}
                  />
                </div>
              ) : null}
            </>
          )}
        </>
      ) : null}
    </FormDialog>
  )
}
