import { useState } from "react"

import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
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
import { useCreateHostedProvider } from "@/shared/api/hostedProviders"

// The form that configures a new hosted provider, in the shape of the
// organization Providers page's key form: the provider picked from the catalog,
// the key label spelled the provider's way, its typed credential fields, and the
// `client_args` JSON escape hatch. No name, because a hosted provider's
// implementation is its identity, one row each.
//
// On submit the deployment asks the provider what it serves on the new key and
// offers all of it, so the create can take a few seconds and arrives with its
// models already listed. A provider config.yml already defines can be added,
// and the stored one then serves instead, which the form says before it is
// saved.

interface ProviderDraft {
  provider: string
  apiKey: string
  apiBase: string
  credentials: CredentialFieldValues
  clientArgs: string
}

const EMPTY_DRAFT: ProviderDraft = {
  provider: "",
  apiKey: "",
  apiBase: "",
  credentials: {},
  clientArgs: "",
}

export function AddHostedProviderForm({
  isOpen,
  onClose,
  configuredProviders = [],
}: {
  isOpen: boolean
  onClose: () => void
  /** The providers config.yml defines, which a stored one would override. */
  configuredProviders?: string[]
}) {
  const create = useCreateHostedProvider()
  const [draft, setDraft] = useState<ProviderDraft>(EMPTY_DRAFT)

  const spec = credentialSpecFor(draft.provider)
  const parsedClientArgs = parseClientArgs(draft.clientArgs)
  const credentialErrors = validateCredentialFields(
    credentialFieldsFor(draft.provider),
    draft.credentials,
  )
  const { isDirty } = useDirtySnapshot(draft)
  const canSubmit =
    draft.provider !== "" &&
    draft.apiKey !== "" &&
    parsedClientArgs.ok &&
    Object.keys(credentialErrors).length === 0

  const submit = () => {
    if (!canSubmit || !parsedClientArgs.ok) return
    const clientArgs = mergeCredentialFields(
      draft.credentials,
      parsedClientArgs.value,
    )
    const apiBase = draft.apiBase.trim()
    create.mutate(
      {
        provider: draft.provider.trim(),
        api_key: draft.apiKey,
        // An empty box means the provider's default endpoint, which the API
        // spells as an absent field rather than an empty string.
        ...(apiBase ? { api_base: apiBase } : {}),
        ...(clientArgs ? { client_args: clientArgs } : {}),
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
      {configuredProviders.includes(draft.provider.trim()) ? (
        <InfoBanner tone="warning">
          <p role="status">
            {draft.provider} is already defined in config.yml. The provider
            added here overrides it until it is removed.
          </p>
        </InfoBanner>
      ) : null}
      <SecretField
        label={spec?.apiKeyLabel ?? "API key"}
        value={draft.apiKey}
        onChange={(apiKey) => setDraft({ ...draft, apiKey })}
        description={[
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
        description="Optional. Only needed for a self-hosted or regional endpoint."
      />
      <ProviderCredentialFields
        provider={draft.provider}
        values={draft.credentials}
        onChange={(credentials) => setDraft({ ...draft, credentials })}
        errors={credentialErrors}
      />
      <ClientArgsField
        value={draft.clientArgs}
        onChange={(clientArgs) => setDraft({ ...draft, clientArgs })}
        error={parsedClientArgs.ok ? null : parsedClientArgs.error}
      />
    </FormDialog>
  )
}
