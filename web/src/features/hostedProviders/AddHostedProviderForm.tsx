import { useState } from "react"

import type { TestHostedProviderRequest } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import {
  ConnectionTestButton,
  ConnectionTestResult,
} from "@/features/providers/ConnectionTest"
import {
  BYO_UNSUPPORTED_PROVIDERS,
  type CredentialFieldValues,
  credentialFieldsFor,
  credentialSpecFor,
  mergeCredentialFields,
  validateCredentialFields,
} from "@/features/providers/providerCredentialFields"
import {
  ClientArgsField,
  ProviderComboBox,
  ProviderCredentialFields,
  parseClientArgs,
} from "@/features/providers/providerFields"
import {
  useCreateHostedProvider,
  useTestHostedProvider,
} from "@/shared/api/hostedProviders"
import { useProviderDetail } from "@/shared/api/providers"

import { ServesUnlistedModelsField } from "./ServesUnlistedModelsField"
import { SupportedCountriesField } from "./SupportedCountriesField"
import { parseCountries } from "./supportedCountries"

// The form that configures a new hosted provider, in the shape of the
// organization Providers page's key form: the provider picked from the catalog,
// the key label spelled the provider's way, its typed credential fields, and the
// `client_args` JSON escape hatch. The key rule mirrors the server's, which
// calls a provider pointed at its own endpoint with a placeholder key, so the
// form never holds back a save the API would accept.
//
// On submit the deployment asks the provider what it serves on the new key and
// offers all of it, so the create can take a few seconds and arrives with its
// models already listed. A provider config.yml already defines can be added,
// and the stored one then serves instead, which the form says before it is
// saved.

interface ProviderDraft {
  provider: string
  name: string
  apiKey: string
  apiBase: string
  credentials: CredentialFieldValues
  clientArgs: string
  servesUnlistedModels: boolean
  supportedCountries: string
}

const EMPTY_DRAFT: ProviderDraft = {
  provider: "",
  name: "",
  apiKey: "",
  apiBase: "",
  credentials: {},
  clientArgs: "",
  servesUnlistedModels: false,
  supportedCountries: "",
}

export function AddHostedProviderForm({
  isOpen,
  onClose,
  configuredProviders = [],
}: {
  isOpen: boolean
  onClose: () => void
  /** The names config.yml defines, which a stored one would override. */
  configuredProviders?: string[]
}) {
  const create = useCreateHostedProvider()
  const test = useTestHostedProvider()
  const [draft, setDraft] = useState<ProviderDraft>(EMPTY_DRAFT)
  const detail = useProviderDetail(draft.provider)
  const selected = detail.data?.id === draft.provider ? detail.data : undefined
  const name = draft.name.trim()
  const apiBase = draft.apiBase.trim()
  const needsKey = (selected?.requires_api_key ?? true) && apiBase === ""
  const nameHasDelimiter = /[:/]/.test(name)
  const effectiveName = name || draft.provider.trim()

  const spec = credentialSpecFor(draft.provider)
  const parsedClientArgs = parseClientArgs(draft.clientArgs)
  const countries = parseCountries(draft.supportedCountries)
  const credentialErrors = validateCredentialFields(
    credentialFieldsFor(draft.provider),
    draft.credentials,
  )
  const { isDirty } = useDirtySnapshot(draft)
  const canSubmit =
    draft.provider !== "" &&
    !nameHasDelimiter &&
    (!needsKey || draft.apiKey !== "") &&
    parsedClientArgs.ok &&
    countries.ok &&
    Object.keys(credentialErrors).length === 0

  // One body for the save and the test, so the test tries what would be saved.
  const buildPayload = (): TestHostedProviderRequest | null => {
    if (
      draft.provider === "" ||
      nameHasDelimiter ||
      !parsedClientArgs.ok ||
      Object.keys(credentialErrors).length > 0
    ) {
      return null
    }
    const clientArgs = mergeCredentialFields(
      draft.credentials,
      parsedClientArgs.value,
    )
    return {
      provider: draft.provider.trim(),
      ...(name && name !== draft.provider.trim() ? { name } : {}),
      ...(draft.apiKey ? { api_key: draft.apiKey } : {}),
      ...(apiBase ? { api_base: apiBase } : {}),
      ...(clientArgs ? { client_args: clientArgs } : {}),
    }
  }

  const submit = () => {
    const payload = buildPayload()
    if (!canSubmit || payload === null || !countries.ok) return
    create.mutate(
      {
        ...payload,
        serves_unlisted_models: draft.servesUnlistedModels,
        ...(countries.value === null
          ? {}
          : { supported_countries: countries.value }),
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
      title="Add hosted provider"
      description="Requests that bring no credential of their own are served on this key. Every model the provider lists on it is offered at once."
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
        label="Provider"
        value={draft.provider}
        // The typed fields belong to the provider, so a change to it drops
        // values that no longer have a field to sit in.
        onChange={(provider) =>
          setDraft({ ...draft, provider, credentials: {} })
        }
        excludeIds={BYO_UNSUPPORTED_PROVIDERS}
        description="The implementation that serves it, so it has to be one this build can dispatch to."
      />
      <Field
        label="Name"
        value={draft.name}
        onChange={(value) => setDraft({ ...draft, name: value })}
        placeholder={draft.provider || "openai-eu"}
        description={
          draft.provider
            ? `Optional. Requests reach it as ${effectiveName}:<model>. Leave blank to name it ${draft.provider}; give a name of its own to serve a second ${draft.provider} endpoint beside the first.`
            : "Optional. Requests reach it as <name>:<model>, and it defaults to the provider."
        }
        isInvalid={nameHasDelimiter}
        errorMessage={
          nameHasDelimiter ? "A name cannot contain ':' or '/'." : undefined
        }
      />
      {effectiveName !== "" && configuredProviders.includes(effectiveName) ? (
        <InfoBanner tone="warning">
          <p role="status">
            {effectiveName} is already defined in config.yml. The provider added
            here overrides it until it is removed.
          </p>
        </InfoBanner>
      ) : null}
      <SecretField
        label={spec?.apiKeyLabel ?? "API key"}
        value={draft.apiKey}
        onChange={(apiKey) => setDraft({ ...draft, apiKey })}
        isRequired={needsKey}
        description={[
          needsKey
            ? ""
            : "Optional for this provider or endpoint, which is called without one.",
          "Encrypted at rest and never shown again; only the last 4 characters come back.",
          spec?.apiKeyHelpText,
        ]
          .filter(Boolean)
          .join(" ")}
      />
      <Field
        label="API base URL"
        value={draft.apiBase}
        onChange={(apiBase) => setDraft({ ...draft, apiBase })}
        placeholder="https://api.example.com/v1"
        description="Optional. Only needed for a self-hosted or regional endpoint, which may then go without a key."
      />
      <ServesUnlistedModelsField
        isSelected={draft.servesUnlistedModels}
        onChange={(servesUnlistedModels) =>
          setDraft({ ...draft, servesUnlistedModels })
        }
      />
      <SupportedCountriesField
        value={draft.supportedCountries}
        parsed={countries}
        onChange={(supportedCountries) =>
          setDraft({ ...draft, supportedCountries })
        }
      />
      <ProviderCredentialFields
        provider={draft.provider}
        values={draft.credentials}
        onChange={(credentials) => setDraft({ ...draft, credentials })}
        errors={credentialErrors}
      />
      <ClientArgsField
        isEncryptedAtRest
        value={draft.clientArgs}
        onChange={(clientArgs) => setDraft({ ...draft, clientArgs })}
        error={parsedClientArgs.ok ? null : parsedClientArgs.error}
      />
      <ConnectionTestResult test={test} />
    </FormDialog>
  )
}
