import {
  Button,
  Card,
  Description,
  Input,
  Label,
  TextField,
} from "@heroui/react"
import { useState } from "react"
import { FiKey, FiSmartphone } from "react-icons/fi"

import type { Passkey } from "@/client"
import {
  useDeletePasskey,
  usePasskeys,
  useRegisterPasskey,
  useRenamePasskey,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { ErrorBanner } from "@/shared/components/ui"
import { RowActions } from "@/shared/components/ui/RowActions"
import { formatDateTime } from "@/shared/helpers/format"
import {
  MAX_PASSKEY_NAME_LENGTH,
  PasskeyCancelledError,
  supportsPasskeys,
} from "@/shared/helpers/webauthn"
import {
  useDeployment,
  useOfferPasskeySignIn,
} from "@/shared/hooks/useDeployment"

/**
 * One registered passkey: what it is called, what kind it is, and when it was
 * last used.
 *
 * A list rather than a `DataTable`. There are at most a handful of rows, every
 * column but the name is a sentence fragment rather than a value worth sorting,
 * and a table would owe an answer below `md` that this shape does not need.
 */
function PasskeyRow({
  passkey,
  onRename,
  onDelete,
  isBusy,
}: {
  passkey: Passkey
  onRename: () => void
  onDelete: () => void
  isBusy: boolean
}) {
  return (
    <li className="flex flex-col gap-3 border-b border-border py-4 last:border-b-0 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex min-w-0 items-start gap-3">
        {/* Which icon is a hint, not a claim: "backed up" is what the
            authenticator reported about syncing, and it is the difference
            between losing this passkey with a device and not. */}
        {passkey.backed_up ? (
          <FiSmartphone
            aria-hidden
            className="mt-0.5 size-4 shrink-0 text-muted"
          />
        ) : (
          <FiKey aria-hidden className="mt-0.5 size-4 shrink-0 text-muted" />
        )}
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-foreground">
            {passkey.name}
          </p>
          <p className="text-xs text-muted">
            {passkey.backed_up
              ? "Synced to your credential manager"
              : "Stored on one device"}
            {" · "}
            {passkey.last_used_at
              ? `Last used ${formatDateTime(passkey.last_used_at)}`
              : "Never used"}
          </p>
          {/* An orphan: registered under a relying-party ID this deployment no
              longer uses, so it cannot sign anybody in and nothing but deleting
              it will help. Said on the row rather than hidden, because a list
              that quietly dropped it would leave the operator with no
              explanation and no way to clean up. */}
          {passkey.is_usable ? null : (
            <p className="text-xs text-warning">
              Registered for a different address than this dashboard now uses,
              so it can no longer sign you in. Delete it and add a new one.
            </p>
          )}
        </div>
      </div>
      <RowActions>
        <Button
          variant="ghost"
          size="sm"
          isDisabled={isBusy}
          onPress={onRename}
        >
          Rename
        </Button>
        <Button
          variant="ghost"
          size="sm"
          isDisabled={isBusy}
          onPress={onDelete}
        >
          Delete
        </Button>
      </RowActions>
    </li>
  )
}

/**
 * The passkeys this identity can sign in to the dashboard with: register one,
 * rename one, remove one.
 *
 * A passkey is a key pair whose private half never leaves the authenticator, so
 * nothing this page can show is a credential: the list is labels and dates, and
 * the ceremony that creates the pair happens in the browser between two calls
 * (`useRegisterPasskey`).
 *
 * Three states this card refuses to conflate, because each has a different way
 * out:
 *
 * - **The browser cannot do passkeys at all** (no `PublicKeyCredential`, or an
 *   insecure context). Nothing here would work, so the register button is not
 *   offered and the reason is said. Checked before the query, because it does
 *   not depend on the answer.
 * - **The deployment is not configured for them.** `GET /v1/auth/webauthn/credentials`
 *   answers 503 naming the setting an operator has to fill in, and that message
 *   is shown rather than replaced with a friendlier one that hides the fix.
 * - **The prompt was dismissed.** Not a failure: `PasskeyCancelledError` is
 *   swallowed, because reporting "that passkey could not be verified" at
 *   somebody who pressed Escape is the dashboard telling them their hardware
 *   broke.
 */
export function PasskeysCard() {
  // Two independent conditions, and the card says something different for each.
  // `passkeys_ready` is the deployment's (it has a relying-party ID at all),
  // read off the bootstrap rather than discovered from a 503, which is the same
  // rule otari#648 settled for mail-dependent surfaces. Deliberately not
  // `sign_in_methods.includes("passkey")`: that one is narrower and also
  // requires a passkey to already exist, so gating on it would hide this card
  // from the person about to register the first one.
  const { passkeys_ready } = useDeployment()
  const canUsePasskeys = supportsPasskeys()
  const passkeys = usePasskeys()
  const register = useRegisterPasskey()
  const rename = useRenamePasskey()
  const remove = useDeletePasskey()
  const offerPasskeySignIn = useOfferPasskeySignIn()

  const [newName, setNewName] = useState("")
  const [renaming, setRenaming] = useState<Passkey | null>(null)
  const [renamedTo, setRenamedTo] = useState("")
  const [deleting, setDeleting] = useState<Passkey | null>(null)

  const rows = passkeys.data?.data ?? []
  const isBusy = register.isPending || rename.isPending || remove.isPending

  const startRegistration = () => {
    if (register.isPending) {
      return
    }
    register.mutate(newName.trim() || undefined, {
      onSuccess: () => {
        setNewName("")
        // The gateway publishes `passkey` in `sign_in_methods` exactly while one
        // could answer, so the first registration changes that answer. Reported
        // rather than refetched: the bootstrap is a context, not a query.
        offerPasskeySignIn(true)
      },
      onError: (error) => {
        // A dismissed prompt is a decision, not a refusal. Clearing the
        // mutation's error is what keeps the banner from appearing for it.
        if (error instanceof PasskeyCancelledError) {
          register.reset()
        }
      },
    })
  }

  const confirmRename = () => {
    const name = renamedTo.trim()
    // `isRequired` on the field sets aria-required, which announces the rule
    // but does not stop the dialog's Save: that is a button with a handler, not
    // a form submit. Without this guard, clearing the box and pressing Save
    // sends a blank name and the server answers 422 from inside a dialog that
    // looks like it worked.
    if (!renaming || rename.isPending || !name) {
      return
    }
    rename.mutate(
      { id: renaming.id, name },
      { onSuccess: () => setRenaming(null) },
    )
  }

  const confirmDelete = () => {
    if (!deleting || remove.isPending) {
      return
    }
    remove.mutate(deleting.id, {
      onSuccess: () => {
        // Deleting the last one stops the deployment offering the method.
        if (rows.length <= 1) {
          offerPasskeySignIn(false)
        }
        setDeleting(null)
      },
    })
  }

  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold text-foreground">Passkeys</h2>
      <Card>
        <Card.Content className="flex flex-col gap-4 px-5 py-5">
          <p className="max-w-3xl text-sm text-muted">
            Sign in with your device instead of typing a password. The key stays
            on the device or in your credential manager; this gateway only ever
            stores the public half, so there is nothing here anybody could sign
            in with. Your password still works.
          </p>

          {!passkeys_ready ? (
            <p className="text-sm text-muted">
              This gateway is not set up for passkeys yet. An operator needs to
              set <code className="font-mono text-xs">public_base_url</code> to
              the address this dashboard is served on, and restart.
            </p>
          ) : !canUsePasskeys ? (
            <p className="text-sm text-muted">
              This browser cannot use passkeys. They need a recent browser on a
              secure (HTTPS) connection.
            </p>
          ) : null}

          {/* The query's own error carries the gateway's message, which for the
              unconfigured case names the setting to fill in. */}
          <ErrorBanner error={passkeys.error} />
          <ErrorBanner error={register.error} />
          <ErrorBanner error={remove.error} />

          {passkeys.isPending && !passkeys.data ? (
            <p className="text-sm text-muted" role="status">
              Loading passkeys…
            </p>
          ) : rows.length > 0 ? (
            <ul className="flex flex-col">
              {rows.map((passkey) => (
                <PasskeyRow
                  key={passkey.id}
                  passkey={passkey}
                  isBusy={isBusy}
                  onRename={() => {
                    setRenaming(passkey)
                    setRenamedTo(passkey.name)
                    rename.reset()
                  }}
                  onDelete={() => {
                    setDeleting(passkey)
                    remove.reset()
                  }}
                />
              ))}
            </ul>
          ) : passkeys.isSuccess && passkeys_ready ? (
            <p className="text-sm text-muted">You have no passkeys yet.</p>
          ) : null}

          {/* Registration is the one action that needs a live ceremony, so it
              is the only one gated. Listing, renaming and deleting stay
              available either way: a deployment that lost its relying-party ID
              is exactly when somebody has orphans to clear out. */}
          {passkeys_ready && canUsePasskeys && !passkeys.isError ? (
            <form
              className="flex flex-col gap-3 sm:flex-row sm:items-end"
              onSubmit={(event) => {
                event.preventDefault()
                startRegistration()
              }}
            >
              <TextField
                value={newName}
                onChange={setNewName}
                className="flex max-w-md flex-1 flex-col gap-1"
              >
                <Label className="text-sm font-medium text-foreground">
                  Name
                </Label>
                <Input
                  placeholder="Work laptop"
                  maxLength={MAX_PASSKEY_NAME_LENGTH}
                />
                <Description className="text-xs text-muted">
                  Optional. It is only a label, so you can tell this passkey
                  from the others.
                </Description>
              </TextField>
              <div className="sm:pb-6">
                <Button
                  type="submit"
                  variant="primary"
                  isPending={register.isPending}
                >
                  Add a passkey
                </Button>
              </div>
            </form>
          ) : null}
        </Card.Content>
      </Card>

      <ConfirmDialog
        isOpen={renaming !== null}
        onOpenChange={(open) => {
          if (!open) {
            setRenaming(null)
          }
        }}
        heading="Rename this passkey"
        confirmLabel="Save"
        confirmVariant="primary"
        isPending={rename.isPending}
        error={rename.error}
        onConfirm={confirmRename}
        body={
          <TextField
            value={renamedTo}
            onChange={setRenamedTo}
            isRequired
            className="flex flex-col gap-1"
          >
            <Label className="text-sm font-medium text-foreground">Name</Label>
            <Input autoFocus maxLength={MAX_PASSKEY_NAME_LENGTH} />
          </TextField>
        }
      />

      <ConfirmDialog
        isOpen={deleting !== null}
        onOpenChange={(open) => {
          if (!open) {
            setDeleting(null)
          }
        }}
        heading="Delete this passkey?"
        confirmLabel="Delete"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={confirmDelete}
        body={
          <>
            <strong className="text-foreground">{deleting?.name}</strong> will
            stop signing you in. The passkey itself stays on your device until
            you remove it there too. You can still sign in with your password.
          </>
        }
      />
    </section>
  )
}
