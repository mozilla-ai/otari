import type { EndpointAuth } from "@/client"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { ExpiryFields, toDatetimeLocal } from "@/features/keys/ExpiryFields"

/** A key's endpoint credential, as the form holds it. */
export interface EndpointAuthDraft {
  /** Blank means the key carries no endpoint credential. */
  headerName: string
  /** Blank keeps the stored value, as the API key field does. */
  secret: string
  /** The datetime-local shape `ExpiryFields` reads, or "" when none is set. */
  expiresAt: string
}

export const EMPTY_ENDPOINT_AUTH: EndpointAuthDraft = {
  headerName: "",
  secret: "",
  expiresAt: "",
}

export function endpointAuthDraftFrom(
  auth: EndpointAuth | null | undefined,
): EndpointAuthDraft {
  if (!auth) return EMPTY_ENDPOINT_AUTH
  return {
    headerName: auth.header_name,
    // Never prefilled: only the last four characters come back.
    secret: "",
    expiresAt: toDatetimeLocal(auth.expires_at),
  }
}

/**
 * Whether this draft can be saved, given what the key already holds.
 *
 * A stored value stands in for a retyped one, except where the key is moving to
 * another address: the gateway refuses to send a value to a host the person
 * saving it never proved they hold, so asking for it here is what keeps the
 * dialog from offering a save the gateway will refuse.
 *
 * Only the structure is checked. Which header names a key may carry is the
 * gateway's rule, and it answers with the name of what is wrong, so a copy of
 * the list here would be a second authority on it that drifts from the first.
 */
export function isEndpointAuthComplete(
  draft: EndpointAuthDraft,
  {
    hasStoredSecret,
    addressMoved,
  }: { hasStoredSecret: boolean; addressMoved: boolean },
): boolean {
  if (draft.headerName.trim() === "") return true
  if (draft.expiresAt === "") return false
  if (draft.secret !== "") return true
  return hasStoredSecret && !addressMoved
}

/**
 * The credential an edge proxy in front of a key's base URL requires.
 *
 * Shown only for a key that has a base URL of its own, since a proxy is
 * something the organization put in front of an address it chose.
 */
export function EndpointAuthFields({
  value,
  onChange,
  stored,
}: {
  value: EndpointAuthDraft
  onChange: (value: EndpointAuthDraft) => void
  /** The credential already held, or null for a key that carries none. */
  stored: EndpointAuth | null
}) {
  const last4 = stored?.last4 ?? ""
  const isSet = value.headerName.trim() !== ""

  return (
    <div className="flex flex-col gap-4">
      <Field
        label="Endpoint credential header"
        value={value.headerName}
        onChange={(headerName) => onChange({ ...value, headerName })}
        placeholder="Fastly-Key"
        description="Optional. The header a CDN or identity-aware proxy in front of this address expects, sent on every request. Leave blank when the address needs no credential of its own."
      />

      {isSet ? (
        <>
          <SecretField
            label="Endpoint credential"
            value={value.secret}
            onChange={(secret) => onChange({ ...value, secret })}
            description={
              last4
                ? `Encrypted at rest. Ending ${last4}. Leave blank to keep it, and retype it when you move this key to a different address.`
                : "Encrypted at rest and never shown again; only the last 4 characters come back."
            }
          />

          <ExpiryFields
            label="Credential expires"
            value={value.expiresAt}
            onChange={(expiresAt) => onChange({ ...value, expiresAt })}
            description="Required. When you expect to rotate it. Otari marks the key as the date nears and once it passes, and never refuses a request because of it."
          />
        </>
      ) : null}
    </div>
  )
}
