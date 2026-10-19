import { useState } from "react"

import type { HostedProvider, TestHostedProviderRequest } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import {
  ConnectionTestButton,
  ConnectionTestResult,
} from "@/features/providers/ConnectionTest"
import {
  type CredentialFieldValues,
  credentialFieldsFor,
  credentialSpecFor,
  mergeCredentialFields,
  splitClientArgs,
  validateCredentialFields,
} from "@/features/providers/providerCredentialFields"
import {
  ClientArgsField,
  formatClientArgs,
  ProviderCredentialFields,
  parseClientArgs,
} from "@/features/providers/providerFields"
import {
  useTestHostedProvider,
  useUpdateHostedProvider,
} from "@/shared/api/hostedProviders"

import { movesToAnotherOrigin } from "./endpointOrigin"

// The editor for one hosted provider. Its name and implementation are immutable
// (the update body cannot change them), so they are named in the description.
// The key box
// starts empty and stays optional, which is the whole shape of a write-only
// credential. A stored typed secret arrives masked, so blank there means "keep
// it" and the mask is echoed back for the server to restore.
//
// The organization restriction is not edited here: the body omits it, which the
// API reads as "keep it".

interface EditDraft {
  apiKey: string
  apiBase: string
  credentials: CredentialFieldValues
  clientArgs: string
  /** Typed secrets already stored, so blank means "keep it", not "clear it". */
  redacted: string[]
}

function draftFrom(provider: HostedProvider): EditDraft {
  const { typed, rest, redacted } = splitClientArgs(
    credentialFieldsFor(provider.provider),
    provider.client_args,
  )
  return {
    apiKey: "",
    apiBase: provider.api_base ?? "",
    credentials: typed,
    clientArgs: formatClientArgs(rest),
    redacted,
  }
}

export function EditHostedProviderForm({
  provider,
  isOpen,
  onClose,
}: {
  provider: HostedProvider
  isOpen: boolean
  onClose: () => void
}) {
  const update = useUpdateHostedProvider()
  const test = useTestHostedProvider()
  const [draft, setDraft] = useState<EditDraft>(() => draftFrom(provider))

  const spec = credentialSpecFor(provider.provider)
  const identity =
    provider.name === provider.provider
      ? provider.name
      : `${provider.name} (${provider.provider})`
  const parsedClientArgs = parseClientArgs(draft.clientArgs)
  const credentialErrors = validateCredentialFields(
    credentialFieldsFor(provider.provider),
    draft.credentials,
    draft.redacted,
  )
  const { isDirty } = useDirtySnapshot(draft)
  // A provider stored without a key has none to send anywhere.
  const needsKey =
    provider.has_api_key &&
    draft.apiKey === "" &&
    movesToAnotherOrigin(provider.api_base ?? "", draft.apiBase)
  const canSubmit =
    parsedClientArgs.ok &&
    Object.keys(credentialErrors).length === 0 &&
    !needsKey

  const buildTest = (): TestHostedProviderRequest | null => {
    if (!canSubmit || !parsedClientArgs.ok) return null
    const clientArgs = mergeCredentialFields(
      draft.credentials,
      parsedClientArgs.value,
      draft.redacted,
    )
    return {
      provider: provider.provider,
      name: provider.name,
      ...(draft.apiKey === "" ? {} : { api_key: draft.apiKey }),
      api_base: draft.apiBase.trim() || null,
      ...(provider.client_args === null && clientArgs === null
        ? {}
        : { client_args: clientArgs }),
    }
  }

  const submit = () => {
    if (!canSubmit || !parsedClientArgs.ok) return
    // A stored secret the form was never shown rides back as the mask, which
    // the server reads as "keep what is stored". Omitted when the API showed
    // none and none was entered: null would clear arguments it could not read.
    const clientArgs = mergeCredentialFields(
      draft.credentials,
      parsedClientArgs.value,
      draft.redacted,
    )
    update.mutate(
      {
        provider: provider.name,
        body: {
          ...(draft.apiKey === "" ? {} : { api_key: draft.apiKey }),
          api_base: draft.apiBase.trim(),
          ...(provider.client_args === null && clientArgs === null
            ? {}
            : { client_args: clientArgs }),
        },
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
      title="Edit hosted provider"
      description={`${identity}. ${
        provider.api_key_last4
          ? `The stored key ends in ${provider.api_key_last4}.`
          : "No key is stored for this provider."
      }`}
      submitLabel="Save"
      onSubmit={submit}
      isPending={update.isPending}
      isSubmitDisabled={!canSubmit}
      isDirty={isDirty}
      error={update.error}
      footerStart={<ConnectionTestButton test={test} getPayload={buildTest} />}
    >
      <SecretField
        label={spec?.apiKeyLabel ?? "API key"}
        value={draft.apiKey}
        onChange={(apiKey) => setDraft({ ...draft, apiKey })}
        placeholder="Leave blank to keep the stored key"
        description="Typing a new key here replaces the stored one. There is no way to read the old one back."
        isRequired={needsKey}
        isInvalid={needsKey}
        errorMessage={
          needsKey
            ? "This endpoint is on another host, scheme or port, so enter the key again."
            : undefined
        }
      />
      <Field
        label="API base URL"
        value={draft.apiBase}
        onChange={(apiBase) => setDraft({ ...draft, apiBase })}
        placeholder="Leave blank for the provider's default endpoint"
        description="Pointing this at another host needs the key retyped above."
      />
      <ProviderCredentialFields
        provider={provider.provider}
        values={draft.credentials}
        onChange={(credentials) => setDraft({ ...draft, credentials })}
        errors={credentialErrors}
        redacted={draft.redacted}
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
