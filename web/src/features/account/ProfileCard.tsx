import { useState } from "react"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Skeleton } from "@/design-system/feedback/Skeleton"
import { Field } from "@/design-system/forms/Field"
import { Section } from "@/design-system/layout/Section"
import { useUpdateProfile } from "@/shared/api/auth"
import { useOrganizationContext } from "@/shared/api/organizations"

/**
 * The name you are known by on this deployment, and the only thing about the
 * signed-in identity that is not a credential.
 *
 * It is read from the membership context rather than a request of its own: that
 * is the call the shell already makes to draw the account control, and
 * `caller.full_name` is the very field this writes (otari#832). An identity an
 * admin added to the roster by address carries none until it is supplied here,
 * so before this card such an account was drawn by its address permanently.
 *
 * Clearing it is a real choice and not a broken form: it puts the identity back
 * in the state it was added in, and every surface that draws a person falls back
 * to the address. The address itself is not editable, here or anywhere: changing
 * one is a credential change with a verification flow behind it, and
 * `PUT /v1/auth/password` refuses one for the same reason.
 */
// The client half of the column width the gateway bounds `full_name` at
// (`MAX_FULL_NAME_LENGTH` in `models/tenancy.py`), not a second authority: it
// says so in the field rather than round-tripping to a 422 nobody can read.
// Counted in code points, because that is what the server's `max_length`
// counts and `String.length` counts UTF-16 units, so an emoji is two here and
// one there.
const MAX_DISPLAY_NAME_LENGTH = 255

export function ProfileCard() {
  const context = useOrganizationContext()
  const update = useUpdateProfile()
  const savedName = context.data?.caller?.full_name?.trim() ?? ""

  return (
    <Section
      aria-labelledby="account-name-title"
      className="border-t border-border pt-6 pb-5"
      contentClassName="flex flex-col gap-4"
    >
      <h2 id="account-name-title" className="text-title">
        Your name
      </h2>

      <p className="max-w-3xl text-sm text-muted">
        How you are named on this deployment: in the account menu, on the
        organization roster, and wherever your requests are listed. Clear it and
        you are listed by your sign-in address instead.
      </p>

      {update.isSuccess ? (
        <p
          role="status"
          aria-live="polite"
          className="max-w-3xl text-sm text-success"
        >
          {update.data.full_name
            ? `Saved. You are shown as ${update.data.full_name}.`
            : "Saved. You are shown by your sign-in address from now on."}
        </p>
      ) : null}

      {context.isPending && !context.data ? (
        <Skeleton
          className="h-20 w-full max-w-md"
          ariaLabel="Loading your name"
        />
      ) : context.error && !context.data ? (
        <ErrorBanner error={context.error} />
      ) : (
        // Keyed on what the server holds, so the draft is re-seeded when that
        // moves and never while somebody is typing into it.
        <NameForm
          key={savedName}
          savedName={savedName}
          isPending={update.isPending}
          error={update.error}
          onEdit={() => {
            update.reset()
          }}
          onSubmit={(name) => {
            update.mutate({ full_name: name === "" ? null : name })
          }}
        />
      )}
    </Section>
  )
}

function NameForm({
  savedName,
  isPending,
  error,
  onEdit,
  onSubmit,
}: {
  savedName: string
  isPending: boolean
  error: unknown
  onEdit: () => void
  onSubmit: (name: string) => void
}) {
  const [name, setName] = useState(savedName)
  const trimmed = name.trim()
  const isChanged = trimmed !== savedName
  const isTooLong = [...trimmed].length > MAX_DISPLAY_NAME_LENGTH
  const canSubmit = isChanged && !isTooLong

  return (
    <form
      className="flex flex-col gap-4"
      onSubmit={(event) => {
        event.preventDefault()
        // A form submits on Enter as well as through the button, so the guard
        // that stops a second concurrent write has to be here rather than only
        // on the button's disabled state.
        if (!canSubmit || isPending) {
          return
        }
        onSubmit(trimmed)
      }}
    >
      <Field
        label="Display name"
        value={name}
        onChange={(next) => {
          setName(next)
          // The line reporting the last save describes a name that is no longer
          // the one in the field, so typing clears it. Never mid-request:
          // `reset()` would clear the `isPending` the submit guard reads and let
          // a keystroke open a second write.
          if (!isPending) {
            onEdit()
          }
        }}
        description="Leave this empty to be listed by your sign-in address."
        isInvalid={isTooLong}
        errorMessage={`A name can be at most ${MAX_DISPLAY_NAME_LENGTH} characters.`}
      />

      <ErrorBanner error={error} />

      <div>
        <Button
          type="submit"
          variant="primary"
          isPending={isPending}
          isDisabled={!canSubmit}
        >
          Save name
        </Button>
      </div>
    </form>
  )
}
