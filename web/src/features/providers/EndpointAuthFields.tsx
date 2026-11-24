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
 * The rotation date to send, given what the key already holds.
 *
 * The inputs hold minutes, so a stored date that carried seconds does not
 * survive the round trip through the draft. A save that did not touch the
 * field has to send the instant already stored: converting the draft back
 * would move a rotation date nobody edited, and the gateway re-checks the date
 * whenever the credential changes, so one inside the current minute would move
 * into the past and be refused.
 */
export function endpointAuthExpiresAt(
  draft: EndpointAuthDraft,
  stored: EndpointAuth | null,
): string {
  if (stored && draft.expiresAt === toDatetimeLocal(stored.expires_at)) {
    return stored.expires_at
  }
  return new Date(draft.expiresAt).toISOString()
}

/** The shortest value the gateway stores, mirroring the request schema's floor. */
const SECRET_MIN_LENGTH = 8

/**
 * What is wrong with this draft's value, or undefined when nothing is.
 *
 * A blank value is not wrong, because it keeps whatever is already stored.
 */
export function endpointAuthSecretError(
  draft: EndpointAuthDraft,
): string | undefined {
  // Code points, which is what the gateway counts. `String.length` counts
  // UTF-16 units, so four astral characters would pass here and be refused
  // there, which is the round trip this exists to pre-empt.
  if (draft.secret === "" || [...draft.secret].length >= SECRET_MIN_LENGTH) {
    return undefined
  }
  return `At least ${SECRET_MIN_LENGTH} characters.`
}

/**
 * Why this address cannot carry the drafted endpoint credential, or undefined.
 *
 * The client half of the gateway's own rule, not a second authority: it says on
 * the field what is wrong, rather than leaving a save to reach the refusal.
 */
export function endpointAuthAddressError(
  apiBase: string,
  draft: EndpointAuthDraft,
): string | undefined {
  if (draft.headerName.trim() === "") return undefined
  const address = apiBase.trim()
  if (address === "") {
    return "An endpoint credential needs an address of its own. Enter one, or clear the credential header below."
  }
  if (!/^https:\/\//i.test(address)) {
    return "Must be HTTPS, since the endpoint credential travels with it."
  }
  return undefined
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
  if (endpointAuthSecretError(draft) !== undefined) return false
  if (draft.secret !== "") return true
  return hasStoredSecret && !addressMoved
}

/** What the header field says, which is where a credential is entered and removed. */
const HEADER_DESCRIPTION =
  "Optional. The header a CDN or identity-aware proxy in front of this address expects, sent on every request. The base URL has to be HTTPS, since the credential travels with it. Leave blank when the address needs no credential of its own."

const UNREADABLE_DESCRIPTION =
  "This key stores a credential this deployment cannot read, so none of it is shown. Save with this blank to remove it, or enter the header and the value again to replace it."

/**
 * What the value field says, for the one state the form is in.
 *
 * One sentence per state rather than one paragraph covering them all, because
 * the paragraph offered a blank save in the state where a blank is refused.
 */
function secretDescription({
  hasStored,
  addressMoved,
}: {
  hasStored: boolean
  addressMoved: boolean
}): string {
  if (addressMoved && hasStored) {
    return "Enter the credential again: it goes to the new address."
  }
  if (hasStored) {
    return "Encrypted at rest and never shown again. Leave blank to keep it."
  }
  return `At least ${SECRET_MIN_LENGTH} characters, encrypted at rest and never shown again.`
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
  storedIsUnreadable,
  addressMoved,
}: {
  value: EndpointAuthDraft
  onChange: (value: EndpointAuthDraft) => void
  /** The credential already held, or null for a key that carries none. */
  stored: EndpointAuth | null
  /** Whether the key stores a credential this deployment cannot read. */
  storedIsUnreadable: boolean
  /** Whether this save moves the key to a different address. */
  addressMoved: boolean
}) {
  const last4 = stored?.last4 ?? ""
  const isSet = value.headerName.trim() !== ""
  const secretError = endpointAuthSecretError(value)
  // An unreadable credential is removed by saving this blank, so it is the one
  // stored value the field does not ask for.
  const secretRequired =
    !storedIsUnreadable && (stored === null || addressMoved)

  return (
    <div className="flex flex-col gap-4">
      <Field
        label="Endpoint credential header"
        value={value.headerName}
        onChange={(headerName) => onChange({ ...value, headerName })}
        placeholder="Fastly-Key"
        description={
          storedIsUnreadable ? UNREADABLE_DESCRIPTION : HEADER_DESCRIPTION
        }
      />

      {isSet ? (
        <>
          <SecretField
            label="Endpoint credential"
            value={value.secret}
            onChange={(secret) => onChange({ ...value, secret })}
            isRequired={secretRequired}
            placeholder={last4 ? `Keep ••••${last4}` : undefined}
            description={secretDescription({
              hasStored: last4 !== "",
              addressMoved,
            })}
            isInvalid={secretError !== undefined}
            errorMessage={secretError}
          />

          <ExpiryFields
            label="Credential expires"
            value={value.expiresAt}
            onChange={(expiresAt) => onChange({ ...value, expiresAt })}
            isRequired
            description="When you expect to rotate it. Otari marks the key near that date and never refuses a request because of it."
          />
        </>
      ) : null}
    </div>
  )
}
